// Fused GEMM -> AllReduce, one process per rank, peer memory over IPC.
//
// gemm_rs_mp.hip does the reduce-scatter form, which is what a sequence-parallel
// TP path wants. This file does the all-reduce form, which is what vLLM's and
// SGLang's default (non-SP) RowParallelLinear wants -- and, more importantly,
// it is the only one of the two that decode can use at all.
//
//                --- why all-reduce shards along N, not M ---
//
// Reduce-scatter has no choice: the framework shards the output by token, so
// the epilogue must decide ownership from the row. All-reduce ends with every
// rank holding the whole output, so how the reduction is divided up in the
// middle is entirely ours to pick, and picking N buys two things.
//
// It makes M unconstrained. The M-sharded epilogue needs shard_rows to be a
// multiple of the warp tile, which rejects every decode shape outright -- M=1
// cannot be row-sharded across two ranks, let alone eight. Sharding along N
// leaves M alone, so one epilogue serves prefill and decode.
//
// And for world > 2 it moves less. Reducing along N and then gathering moves
// 2(w-1)/w of C; having every rank push its whole partial to every other moves
// (w-1). At w=8 that is 1.75 against 7.
//
// The constraint that replaces `shard_rows % REG_BLOCK_M` is `N/world % 16`,
// because a 16-column warp-tile chunk must not straddle two owners. Every
// config in gemm.cpp has SPLIT_N == 16, and Qwen3's N=5120 gives 2560/1280/640
// for TP=2/4/8, so this one is satisfied everywhere the other was not.
//
//                --- the shape of an iteration ---
//
//   GEMM          each rank computes a full M x N partial over its K slice.
//                 The epilogue keeps the columns this rank owns in c_own and
//                 pushes the other world-1 column blocks into the owning
//                 rank's inbox, at the slot indexed by this rank so senders
//                 never collide. For a split-K config the epilogue has fp32
//                 partials rather than an answer, so the same scatter happens
//                 one kernel later, in the reduction pass -- see
//                 reduce_splitk_scatter_n in gemm.cpp.
//   barrier       my combine reads slots that peers' GEMMs write.
//   combine+gather
//                 sum my inbox slots into my column shard, which completes it,
//                 then push the completed shard into every rank's output at my
//                 column offset. That is the all-gather, and it is fused into
//                 the combine because the data is already in registers there.
//   barrier       nobody may overwrite an inbox a peer is still reading, and
//                 nobody may read an output a peer is still writing.
//
// Two barriers, same as the RS path, ~11 us against a millisecond iteration.
//
// Running this:
//
//   HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./gemm_ar_mp
//   HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./gemm_ar_mp 4096 5120 6144
//
// Every rank must see every device -- hipIpcOpenMemHandle has to reach the
// exporting one -- so HIP_VISIBLE_DEVICES lists all of them and rank r drives
// visible device r. The node is shared; pin it.
#define HK_GEMM_NO_PYBIND 1
#include "../gemm/bf16fp32/gemm.cpp"

#include "ipc_heap.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

using namespace hk_dist;

// Globals for the N-sharded all-reduce epilogue.
//
// c is the final output, full M x N, and it is symmetric because the gather
// writes into peers' copies of it. c_own is my column shard of my own partial,
// which the combine turns into the finished shard. c_sym is my slot of the
// inbox; translated into an owner's address space it becomes that owner's slot
// for me, so no sender needs to know which slot it is writing.
//
// ws is the split-K scratch. Left null: launch() fills it from its own
// process-static allocator, which is the right owner here because which config
// -- and therefore whether there is a split at all -- is the dispatcher's
// decision, not this file's.
struct micro_globals_ar {
    _gl_A a;
    _gl_B b;
    _gl_C c;              // (M, N) final output, symmetric
    sym_view v;           // the inbox heap's view
    bf16 *c_own;          // (M, shard_cols) my columns of my partial
    bf16 *c_sym;          // my slot of the inbox, (M, shard_cols)
    int shard_cols;
    int my_rank;
    float *ws;
    hipStream_t stream;

    static constexpr bool FUSED_RS = false;
    static constexpr bool FUSED_AR = true;
};

// Generators, identical to gemm_rs_mp.hip's and for the same reasons: built
// from *global* indices so a rank can materialise its own K-slice and the whole
// matrix for its reference with no transfers; values are multiples of 1/8 and
// small, hence exact in bf16, so the two paths differ only in summation order;
// and the mixing is non-separable, because a separable one makes every output
// of a long K come out ~0 and the comparison then measures nothing.
__device__ __forceinline__ unsigned mix(unsigned x, unsigned y, unsigned salt) {
    unsigned h = x * 73856093u ^ y * 19349663u ^ salt;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}
__device__ __forceinline__ float gen_a(int row, int k) {
    return (float)((int)(mix(row, k, 0x9e3779b9u) % 7) - 3) * 0.125f;
}
__device__ __forceinline__ float gen_b(int col, int k) {
    return (float)((int)(mix(col, k, 0x85ebca6bu) % 5) - 2) * 0.125f;
}
__global__ void fill_a(bf16 *p, int rows, int k_local, int k_off) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n = (size_t)rows * k_local;
    for (; i < n; i += stride)
        p[i] = __float2bfloat16(gen_a((int)(i / k_local), k_off + (int)(i % k_local)));
}
__global__ void fill_b(bf16 *p, int rows, int k_local, int k_off) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n = (size_t)rows * k_local;
    for (; i < n; i += stride)
        p[i] = __float2bfloat16(gen_b((int)(i / k_local), k_off + (int)(i % k_local)));
}

// The inbox loads are nontemporal and that is a correctness requirement, not a
// hint: the inbox is in my HBM but written by a peer, and on gfx1100 that write
// does not invalidate my L2, so an ordinary load can return the zeros I memset
// there myself. __builtin_nontemporal_load emits the glc/slc/dlc bypass bits.
// No locality is lost -- each inbox element is read exactly once.
__device__ __forceinline__ float inbox_load(const bf16 *p) {
    unsigned short w = __builtin_nontemporal_load(
        reinterpret_cast<const unsigned short *>(p));
    bf16 v; __builtin_memcpy(&v, &w, sizeof(v));
    return __bfloat162float(v);
}
__device__ __forceinline__ void peer_store(bf16 *p, bf16 v) {
    unsigned short w; __builtin_memcpy(&w, &v, sizeof(w));
    __builtin_nontemporal_store(w, reinterpret_cast<unsigned short *>(p));
}

// Finish my column shard and hand it to everyone.
//
// own is (M, ns): my partial of my own columns. inbox is [world][M][ns]. out is
// (M, N), symmetric, and v translates it into any rank's address space.
//
// The accumulation is in fp32. The partials are bf16 because that is what the
// GEMM epilogue emits, and summing world-1 of them in bf16 would throw away
// most of what little mantissa there is; the pass is entirely memory-bound, so
// the wider accumulator is free. Slot me is skipped rather than zeroed -- my
// own columns went straight to `own`.
//
// Writing my own copy through the same translate() as the peers' keeps one code
// path; translate(p, me) is the identity, so it costs nothing.
__global__ void combine_and_gather(sym_view v, bf16 *out, const bf16 *own,
                                   const bf16 *inbox, int M, int N, int ns,
                                   int world, int me) {
    const size_t n = (size_t)M * ns;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float acc = __bfloat162float(own[i]);
        for (int r = 0; r < world; r++)
            if (r != me) acc += inbox_load(inbox + (size_t)r * n + i);
        const bf16 val = __float2bfloat16(acc);
        // (row, local col) -> the same element of every rank's full output.
        const int row = (int)(i / ns), lc = (int)(i % ns);
        const size_t off = (size_t)row * N + (size_t)me * ns + lc;
        for (int r = 0; r < world; r++) {
            bf16 *dst = v.translate(out, r) + off;
            if (r == me) *dst = val; else peer_store(dst, val);
        }
    }
    // The gather stores have to retire before the wave ends, or the barrier
    // flag that follows can get out ahead of them -- the tiles go over PCIe to
    // a peer's HBM while the flag goes to a host page, two paths with nothing
    // ordering them.
    __threadfence_system();
}

// Same, restricted to a column band of a row-major M x N. The fused path has
// two independent halves -- the columns this rank computed and stored locally,
// and the columns a peer computed and pushed -- and a single whole-buffer
// number cannot tell which of them is wrong.
__global__ void err_scale_cols(const bf16 *got, const bf16 *ref, int M, int N,
                               int c0, int c1, unsigned *out) {
    const int w = c1 - c0;
    const size_t n = (size_t)M * w;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float worst = 0.f, scale = 0.f;
    for (; i < n; i += stride) {
        const size_t o = (i / w) * N + c0 + (i % w);
        const float g = __bfloat162float(got[o]), r = __bfloat162float(ref[o]);
        worst = fmaxf(worst, fabsf(g - r));
        scale = fmaxf(scale, fabsf(r));
    }
    atomicMax(out + 0, __float_as_uint(worst));
    atomicMax(out + 1, __float_as_uint(scale));
}

__global__ void err_scale(const bf16 *got, const bf16 *ref, size_t n,
                          unsigned *out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float worst = 0.f, scale = 0.f;
    for (; i < n; i += stride) {
        const float g = __bfloat162float(got[i]), r = __bfloat162float(ref[i]);
        worst = fmaxf(worst, fabsf(g - r));
        scale = fmaxf(scale, fabsf(r));
    }
    // atomicMax over float bit patterns is exact for non-negative floats:
    // IEEE-754 orders them the same way as their unsigned representations.
    atomicMax(out + 0, __float_as_uint(worst));
    atomicMax(out + 1, __float_as_uint(scale));
}

// Compare an (M, ns) shard buffer against the band [c0, c0+ns) of a row-major
// M x N. The fused epilogue's local branch writes the first; the plain epilogue
// writes the second, and test.py already covers that one. Both come from the
// same accumulators and differ only in the destination address, so they must
// agree bit for bit -- a disagreement puts the bug in the store, not the GEMM,
// and the first bad (row, col) says which tile.
__global__ void cmp_shard(const bf16 *shard, const bf16 *full, int M, int N,
                          int ns, int c0, unsigned *out) {
    const size_t n = (size_t)M * ns;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const int r = (int)(i / ns), c = (int)(i % ns);
        const float a = __bfloat162float(shard[i]);
        const float b = __bfloat162float(full[(size_t)r * N + c0 + c]);
        if (!(fabsf(a - b) <= 1e-3f * fmaxf(1.f, fabsf(b)))) {
            atomicAdd(out + 0, 1u);
            atomicMin(out + 1, (unsigned)i);   // lowest bad linear index
            // Which sub-tile. big_config's warp tile is REG_BLOCK_N=64 wide and
            // splits into N_SPLIT=4 columns of SPLIT_N=16, and is REG_BLOCK_M=32
            // tall with WARP_ROWS=4 of them per block. If the damage lines up
            // with either index the store is the culprit, not the math.
            atomicAdd(out + 2 + ((c % 64) / 16), 1u);
            // Was the tile stored with wrong values, or not stored at all? The
            // caller zeroes the destination first, so a bad element that is
            // still exactly zero means the store never happened.
            if (a == 0.f) atomicAdd(out + 6, 1u);
            atomicMax(out + 7, __float_as_uint(fabsf(a)));
        }
    }
}

#define CHK(x) HKD_CHECK(x)
#define SPIN_TICK __asm__ __volatile__("" ::: "memory")

// An N-way device-side barrier on the stream, with a deadline. Identical to
// gemm_rs_mp.hip's; the long argument for host flags rather than heap flags,
// and for wall_clock64() rather than clock64(), is over there.
__global__ void dev_barrier(uint32_t *flags, uint32_t epoch, int world, int me,
                            long long budget, unsigned *timed_out) {
    const int r = threadIdx.x;
    if (r >= world) return;
    if (r == me)
        __hip_atomic_store(flags + me, epoch, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    const long long t0 = wall_clock64();
    while (__hip_atomic_load(flags + r, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        SPIN_TICK;
        if (wall_clock64() - t0 > budget) { atomicExch(timed_out, epoch); return; }
    }
}

// This run's mode of the fabric, measured in this process -- see the long note
// at the top of gemm_rs_mp.hip. The node is bimodal per launch, so a fused time
// from one run and a bandwidth from another describe different hardware states.
__global__ void push_bytes(sym_view v, float4 *dst, const float4 *src, int n4,
                           int peer) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 *remote = v.translate(dst, peer);
    for (; i < (size_t)n4; i += stride) remote[i] = src[i];
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const char *rank_s = getenv("HK_RANK"), *world_s = getenv("HK_WORLD");
    if (!rank_s || !world_s) {
        fprintf(stderr, "HK_RANK/HK_WORLD unset -- launch with ./run_mp.sh\n");
        return 2;
    }
    const int rank = atoi(rank_s), world = atoi(world_s);

    int avail = 0; CHK(hipGetDeviceCount(&avail));
    if (avail < world) {
        fprintf(stderr, "rank %d: world %d but only %d visible devices. Every "
                        "rank must see every device.\n", rank, world, avail);
        return 2;
    }

    file_bootstrap bs(rank, world);

    // Qwen3.8-27B under TP=`world`. H=5120 is hidden_size and therefore N;
    // attn_out K=6144 is 24 heads x 256; mlp_down K=17408 is intermediate_size.
    const int H = 5120, I = 17408, ATTN = 6144;
    // M_pad is what the GEMM actually computes. The kernel writes whole
    // BLOCK_M-row tiles, so a decode batch of one token is computed as 16 and
    // 15 rows are thrown away -- which is free, because at these shapes the
    // cost is reading B and B does not depend on M at all. It is reported so
    // that nobody reads the M=1 row as a measurement of one row of work.
    struct shape { const char *phase; int conc; const char *op; int M, N, K; };
    std::vector<shape> shapes;
    if (argc > 3) {
        shapes.push_back({"manual", 0, "manual", atoi(argv[1]), atoi(argv[2]),
                          atoi(argv[3])});
    } else {
        // Decode is one token per sequence, so M is the concurrency.
        for (int c : {1, 2, 4, 8, 16, 32}) {
            shapes.push_back({"decode", c, "attn_out", c, H, ATTN});
            shapes.push_back({"decode", c, "mlp_down", c, H, I});
        }
        // Prefill batches C requests of 2048 tokens into min(C*2048, 8192)
        // tokens per forward -- the chunk cap most engines default to, which is
        // why the shape stops growing past C=4.
        for (int c : {1, 2, 4, 8, 16, 32}) {
            const long long tok = (long long)c * 2048;
            const int M = (int)(tok < 8192 ? tok : 8192);
            shapes.push_back({"prefill", c, "attn_out", M, H, ATTN});
            shapes.push_back({"prefill", c, "mlp_down", M, H, I});
        }
    }

    // The M every shape is rounded up to. Below one thin block there is nothing
    // to tile with; gemm.cpp's plain entry point stages through a padded copy
    // for that case, but the fused epilogues push into peer inboxes whose
    // geometry this file owns, so the padding has to happen here instead.
    const int PAD = 16;
    auto padded = [&](int M) { return M < PAD ? PAD : M; };

    size_t max_a = 0, max_b = 0, max_c = 0, max_own = 0;
    for (const auto &sh : shapes) {
        const size_t kl = (size_t)sh.K / world, mp = padded(sh.M);
        max_a  = std::max(max_a, mp * kl);
        max_b  = std::max(max_b, (size_t)sh.N * kl);
        max_c  = std::max(max_c, mp * sh.N);
        max_own = std::max(max_own, mp * (size_t)(sh.N / world));
    }
    // world slots of one column shard each -- world*shard == N, so the inbox is
    // exactly one full output regardless of TP, same as the RS path's.
    const size_t max_inbox = max_c;
    const size_t WIRE_BYTES = 64u << 20;
    const size_t wire_elems = WIRE_BYTES / sizeof(bf16);
    const size_t heap_bytes =
        (max_a + max_b + max_c + max_own + 2 * wire_elems) * sizeof(bf16)
        + (16u << 20);
    const size_t inbox_bytes = max_inbox * sizeof(bf16) + (4u << 20);

    // Which of the two correct treatments of the stale-L2 problem to use. The
    // RS path measured the cached-plus-host-sync route as the faster one on
    // prefill (2.09 ms against 2.57), because the sync costs one host round
    // trip per op while uncached memory is paid on every 2-byte peer store.
    // That ranking is expected to flip on decode, where the op is ~0.15 ms and
    // a host round trip is a visible fraction of it -- which is exactly why
    // both routes are still here and selectable in one build.
    const char *uc = getenv("HK_INBOX_UNCACHED");
    const bool inbox_cached = !(uc && atoi(uc));

    fprintf(stderr, "rank %d: heap %zu MB + inbox %zu MB (%s)\n", rank,
            heap_bytes >> 20, inbox_bytes >> 20,
            inbox_cached ? "cached, host sync before combine" : "uncached");
    ipc_heap heap(heap_bytes, rank, bs);
    ipc_heap ibox(inbox_bytes, rank, bs, /*uncached=*/!inbox_cached);
    bf16 *A     = heap.allocate<bf16>(max_a);
    bf16 *B     = heap.allocate<bf16>(max_b);
    bf16 *Cout  = heap.allocate<bf16>(max_c);      // (M, N) final, symmetric
    bf16 *Cown  = heap.allocate<bf16>(max_own);    // (M, N/world) my columns
    bf16 *Inbox = ibox.allocate<bf16>(max_inbox);  // [world][M][N/world]
    bf16 *WireSrc = heap.allocate<bf16>(wire_elems);
    bf16 *WireDst = heap.allocate<bf16>(wire_elems);
    // In one process the heap enforces that every rank ran the same allocation
    // sequence; one process per rank can only check it, and an unchecked
    // violation gives valid-looking pointers into the wrong object.
    heap.check_symmetry();
    ibox.check_symmetry();

    host_flags hf(MAX_RANKS + 1, bs);
    uint32_t *const flags = hf.device();
    unsigned *const tmo = reinterpret_cast<unsigned *>(flags + MAX_RANKS);
    fprintf(stderr, "rank %d: heap ready\n", rank);

    const hipStream_t S = heap.stream();
    const sym_view V = heap.view();
    // The inbox is in its own heap, so peer addresses for it come from that
    // heap's view; mixing the two would translate through the wrong base and
    // land somewhere plausible but wrong. Cout lives in the main heap, so the
    // gather uses V and the epilogue's push uses VI.
    const sym_view VI = ibox.view();
    int wall_khz = 0;
    CHK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate,
                              heap.device()));
    const long long BUDGET = (long long)wall_khz * 2000;   // 2 s, to terminate

    uint32_t epoch = 0;
    bool barrier_failed = false;
    auto sync_ranks = [&]() {
        hipLaunchKernelGGL(dev_barrier, dim3(1), dim3(MAX_RANKS), 0, S,
                           flags, ++epoch, world, rank, BUDGET, tmo);
    };
    auto wait_local = [&]() { CHK(hipStreamSynchronize(S)); };
    // Drain, then rendezvous on the host. The file bootstrap costs milliseconds
    // -- the same order as the GEMMs being timed -- so this belongs *between*
    // measurements and never inside one.
    auto settle = [&]() {
        heap.barrier();
        const unsigned t = hf.host()[MAX_RANKS];
        if (t && !barrier_failed) {
            barrier_failed = true;
            fprintf(stderr, "rank %d: device barrier timed out at epoch %u -- a "
                            "peer is gone or wedged; results below are void\n",
                    rank, t);
        }
    };

    double barrier_us = 0.0;
    {
        fprintf(stderr, "rank %d: barrier probe\n", rank);
        const int reps = 100;
        for (int w = 0; w < 10; w++) sync_ranks();
        settle();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) sync_ranks();
        wait_local();
        const auto t1 = std::chrono::steady_clock::now();
        barrier_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
        settle();
        fprintf(stderr, "rank %d: barrier probe done, %.1f us\n", rank, barrier_us);
    }

    hipDeviceProp_t prop; CHK(hipGetDeviceProperties(&prop, heap.device()));
    if (rank == 0)
        printf("=== fused GEMM->AllReduce (N-sharded), %d ranks x %s (%s), "
               "one process per rank, peer memory over IPC ===\n",
               world, prop.name, prop.gcnArchName);

    double bw = 0.0;
    {
        const int n4 = (int)(WIRE_BYTES / 16);
        const int blocks = std::min((n4 + 255) / 256, 1024);
        const int peer = (rank + 1) % world;
        const int reps = 50;
        auto once = [&]() {
            hipLaunchKernelGGL(push_bytes, dim3(blocks), dim3(256), 0, S, V,
                               (float4 *)WireDst, (const float4 *)WireSrc, n4,
                               peer);
        };
        for (int w = 0; w < 5; w++) once();
        settle();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) once();
        wait_local();
        const auto t1 = std::chrono::steady_clock::now();
        settle();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        bw = WIRE_BYTES / (ms * 1e-3) / 1e9;
        if (rank == 0)
            printf("\nwire, measured in this process: %.1f GB/s per rank "
                   "(ring, %d MB pushes) -- %s mode.\nThis node is bimodal per "
                   "launch, so rows below are only comparable to other rows "
                   "measured in it.\n",
                   bw, (int)(WIRE_BYTES >> 20), bw < 20.0 ? "SLOW" : "FAST");
    }
    if (rank == 0)
        printf("cross-rank device barrier: %.1f us each, back to back.\n",
               barrier_us);

    // The reference: the whole GEMM over the full K on this rank's own device.
    // Each rank builds its own rather than one rank shipping slices -- the
    // generators are deterministic, and a local allocation is cheaper than a
    // transfer plus the code to orchestrate it. Off-heap: nobody else names it.
    bf16 *rA = nullptr, *rB = nullptr, *rC = nullptr;
    {
        size_t ra = 0, rb = 0;
        for (const auto &sh : shapes) {
            ra = std::max(ra, (size_t)padded(sh.M) * sh.K);
            rb = std::max(rb, (size_t)sh.N * sh.K);
        }
        CHK(hipMalloc(&rA, ra * sizeof(bf16)));
        CHK(hipMalloc(&rB, rb * sizeof(bf16)));
        CHK(hipMalloc(&rC, max_c * sizeof(bf16)));
    }

    if (rank == 0) {
        printf("\ngemm_ms is the %d GEMMs with no communication and no reduce "
               "at all -- not a\ncorrect implementation, just the floor overlap "
               "is aiming at. fus/gemm of 1.00\nwould mean the transfer "
               "vanished completely behind the math. M_pad is what the\nGEMM "
               "computes: the kernel writes whole 16-row tiles.\n\n");
        printf("%-8s %4s %-9s %6s %6s %7s %6s %5s %8s %8s %8s %8s %6s  %s\n",
               "phase", "conc", "op", "M", "M_pad", "K", "K_loc", "N",
               "gemm_ms", "seq_ms", "fus_ms", "speedup", "f/g", "check");
    }

    bool all_ok = true;
    for (const auto &sh : shapes) {
        const int M = padded(sh.M), N = sh.N, K = sh.K;
        const int K_local = K / world, ns = N / world;
        const size_t c_elems = (size_t)M * N, own_elems = (size_t)M * ns;

        // What the N-sharded epilogue actually needs. A whole warp tile must
        // not straddle two owners -- the epilogue decides ownership once per
        // warp tile rather than once per 16-column chunk, because deciding it
        // per chunk spills registers and silently corrupts the accumulators
        // (see the comment on the FUSED_AR branch in gemm.cpp). The widest warp
        // tile any config in gemm.cpp produces is 64 columns, and a static
        // assert there holds that to the same number. The GEMM still needs its
        // own N and K_local to tile; M is unconstrained, which is the entire
        // point of this file.
        const char *why = nullptr;
        if (K % world || N % world)            why = "K or N does not split across ranks";
        else if (ns % 64)                      why = "the column shard boundary falls inside a warp tile";
        else if (N % thin_config::BLOCK_N)     why = "N does not tile";
        else if (K_local % big_config::K_STEP) why = "K_local does not tile";
        if (why) {
            if (rank == 0)
                printf("%-8s %4d %-9s %6d %6d %7d %6d %5d   -- %s\n",
                       sh.phase, sh.conc, sh.op, sh.M, M, K, K_local, N, why);
            continue;
        }

        CHK(hipMemsetAsync(Inbox, 0, c_elems * sizeof(bf16), S));
        // Zeroed so that "wrong" and "never written" are distinguishable; the
        // epilogue is supposed to cover all of it, and if it does this costs
        // one memset per shape outside the timed loop.
        CHK(hipMemsetAsync(Cown, 0, own_elems * sizeof(bf16), S));
        hipLaunchKernelGGL(fill_a, dim3(1024), dim3(256), 0, S, A, M, K_local,
                           rank * K_local);
        hipLaunchKernelGGL(fill_b, dim3(1024), dim3(256), 0, S, B, N, K_local,
                           rank * K_local);
        hipLaunchKernelGGL(fill_a, dim3(1024), dim3(256), 0, S, rA, M, K, 0);
        hipLaunchKernelGGL(fill_b, dim3(1024), dim3(256), 0, S, rB, N, K, 0);
        CHK(hipStreamSynchronize(S));
        {
            micro_globals g{_gl_A(rA, 1, 1, M, K), _gl_B(rB, 1, 1, N, K),
                            _gl_C(rC, 1, 1, M, N), S};
            dispatch_any(g);
            CHK(hipStreamSynchronize(S));
        }

        // Designated initializers rather than assignment: kittens::gl has no
        // default constructor, so the struct is not default-constructible and
        // has to be built in one expression. Naming the fields also means a
        // reordering of the struct cannot silently swap two ints.
        micro_globals_ar gf{
            .a = _gl_A(A, 1, 1, M, K_local),
            .b = _gl_B(B, 1, 1, N, K_local),
            .c = _gl_C(Cout, 1, 1, M, N),
            .v = VI,
            .c_own = Cown,
            .c_sym = Inbox + (size_t)rank * own_elems,
            .shard_cols = ns,
            .my_rank = rank,
            .ws = nullptr,
            .stream = S,
        };
        // The unfused path's GEMM writes a full local partial and nothing else.
        micro_globals gs{_gl_A(A, 1, 1, M, K_local), _gl_B(B, 1, 1, N, K_local),
                         _gl_C(Cout, 1, 1, M, N), S};

        // Two syncs around the barrier, doing different jobs. The first is a
        // release: draining my stream from the host is what gets my pushes out
        // of my own L2 and onto the wire, so the flag I raise next does not
        // outrun them. The second is an acquire: the runtime puts a
        // system-scope acquire on the first dispatch after a host sync, and on
        // gfx1100 that is the only thing that invalidates my L2 -- so it has to
        // sit immediately before the combine, because the acquire rides on the
        // next dispatch only and sync_ranks() would otherwise consume it.
        auto run_fused = [&]() {
            dispatch_any(gf);
            if (inbox_cached) CHK(hipStreamSynchronize(S));
            sync_ranks();
            if (inbox_cached) CHK(hipStreamSynchronize(S));
            hipLaunchKernelGGL(combine_and_gather, dim3(1024), dim3(256), 0, S,
                               V, Cout, Cown, Inbox, M, N, ns, world, rank);
            // Not optional: without it a peer's next GEMM pushes into my inbox
            // while my combine is still reading it, and its gather overwrites
            // my Cout while I am still checking it. Nothing else orders those,
            // since the only thing between them is my own stream. The way to
            // remove it is double-buffering, not deleting it.
            sync_ranks();
        };

        // The unfused path: the same GEMM producing a full local partial, then
        // move every other rank's column shard of it across, then the same
        // combine. The copies sit on my own stream behind my own GEMM, which is
        // exactly the serialization fusing removes.
        //
        // hipMemcpy2DAsync because a column shard of a row-major M x N is
        // strided -- ns contiguous elements every N. This is a real
        // disadvantage of sharding along N rather than M and it is charged to
        // the unfused path honestly: the fused epilogue has the tile in
        // registers and does not care, but a framework doing the copy
        // afterwards pays for the stride.
        auto run_sequential = [&]() {
            dispatch_any(gs);
            for (int p = 0; p < world; p++) {
                if (p == rank) continue;
                CHK(hipMemcpy2DAsync(
                        VI.translate(Inbox + (size_t)rank * own_elems, p),
                        ns * sizeof(bf16),
                        Cout + (size_t)p * ns, N * sizeof(bf16),
                        ns * sizeof(bf16), M, hipMemcpyDeviceToDevice, S));
            }
            // My own columns, locally, so the combine sees the same layout it
            // does in the fused path.
            CHK(hipMemcpy2DAsync(Cown, ns * sizeof(bf16),
                                 Cout + (size_t)rank * ns, N * sizeof(bf16),
                                 ns * sizeof(bf16), M, hipMemcpyDeviceToDevice, S));
            if (inbox_cached) CHK(hipStreamSynchronize(S));
            sync_ranks();
            if (inbox_cached) CHK(hipStreamSynchronize(S));
            hipLaunchKernelGGL(combine_and_gather, dim3(1024), dim3(256), 0, S,
                               V, Cout, Cown, Inbox, M, N, ns, world, rank);
            sync_ranks();
        };

        // The ceiling: the same GEMM with no communication and no combine. Not
        // a correct implementation of anything -- it drops the reduce -- but
        // without it a speedup over the unfused path cannot be read as progress
        // toward anything, only as a ratio between two arbitrary numbers.
        auto run_gemm_only = [&]() { dispatch_any(gs); };

        auto verify = [&](double *out_rel) {
            unsigned *d; CHK(hipMalloc(&d, 8)); CHK(hipMemset(d, 0, 8));
            hipLaunchKernelGGL(err_scale, dim3(1024), dim3(256), 0, S, Cout, rC,
                               c_elems, d);
            CHK(hipStreamSynchronize(S));
            unsigned h[2] = {0, 0};
            CHK(hipMemcpy(h, d, 8, hipMemcpyDeviceToHost));
            CHK(hipFree(d));
            float fh[2]; memcpy(fh, h, 8);
            // A reference with no magnitude makes the ratio meaningless and
            // would read as a perfect score. The terms are order 0.1 and there
            // are thousands of them, so max|ref| below 1 means the generator is
            // cancelling rather than the GEMM working.
            if (!(fh[1] > 1.f)) { *out_rel = -1.0; return false; }
            *out_rel = (double)fh[0] / fh[1];
            // bf16 has 8 significand bits, so one rounding is up to 3.9e-3
            // relative; this path rounds each of `world` partials and then
            // their sum. 5e-2 is ~4x headroom over that and still 20x below
            // what any routing bug costs -- a misrouted tile is wrong by order
            // 100% of scale, not by parts per hundred.
            return *out_rel < 5e-2;
        };

        // HK_DIAG=1 splits the fused result by column band. Each band is
        // produced by exactly one rank, so this says which side is wrong.
        auto diag_cols = [&]() {
            if (!getenv("HK_DIAG")) return;
            // max|x| over a buffer: err_scale's second output is max|ref|,
            // so passing the same pointer twice gives exactly that.
            auto mag = [&](const bf16 *p, size_t n) {
                unsigned *d; CHK(hipMalloc(&d, 8)); CHK(hipMemset(d, 0, 8));
                hipLaunchKernelGGL(err_scale, dim3(1024), dim3(256), 0, S, p, p, n, d);
                CHK(hipStreamSynchronize(S));
                unsigned h[2] = {0, 0};
                CHK(hipMemcpy(h, d, 8, hipMemcpyDeviceToHost));
                CHK(hipFree(d));
                float fh[2]; memcpy(fh, h, 8);
                return (double)fh[1];
            };
            fprintf(stderr, "rank %d: max|Cown| %.4g\n", rank, mag(Cown, own_elems));
            for (int r = 0; r < world; r++)
                fprintf(stderr, "rank %d: max|inbox slot %d| %.4g\n", rank, r,
                        mag(Inbox + (size_t)r * own_elems, own_elems));
            for (int r = 0; r < world; r++) {
                unsigned *d; CHK(hipMalloc(&d, 8)); CHK(hipMemset(d, 0, 8));
                hipLaunchKernelGGL(err_scale_cols, dim3(1024), dim3(256), 0, S,
                                   Cout, rC, M, N, r * ns, (r + 1) * ns, d);
                CHK(hipStreamSynchronize(S));
                unsigned h[2] = {0, 0};
                CHK(hipMemcpy(h, d, 8, hipMemcpyDeviceToHost));
                CHK(hipFree(d));
                float fh[2]; memcpy(fh, h, 8);
                fprintf(stderr, "rank %d: band %d (cols %d..%d, owner %d): "
                                "max|err| %.4g  max|ref| %.4g\n",
                        rank, r, r * ns, (r + 1) * ns - 1, r, fh[0], fh[1]);
            }
        };

        double rel_f = 0, rel_s = 0;
        // Nobody may push before everybody has zeroed: the memset above is on
        // my stream and says nothing about a peer's, so without this the rank
        // that starts late wipes pushes the early rank already delivered and
        // its combine sums one partial short. Startup-only -- the steady state
        // is covered by the barriers inside run_fused.
        settle();
        run_fused(); settle();
        diag_cols();
        bool ok = verify(&rel_f);
        // Isolate the fused epilogue's *local* branch. Cown was just written by
        // store_at() out of the same accumulators the plain path stores with
        // store(); re-running the plain GEMM into Cout gives the same partial
        // under an epilogue that test.py covers, so the two must agree.
        if (getenv("HK_DIAG")) {
            dispatch_any(gs);
            auto cmp = [&](const char *tag, const bf16 *shard, int sns, int sc0) {
                unsigned *d; CHK(hipMalloc(&d, 40));
                unsigned init[10] = {0}; init[1] = ~0u;
                CHK(hipMemcpy(d, init, 40, hipMemcpyHostToDevice));
                hipLaunchKernelGGL(cmp_shard, dim3(1024), dim3(256), 0, S,
                                   shard, Cout, M, N, sns, sc0, d);
                CHK(hipStreamSynchronize(S));
                unsigned h[10]; CHK(hipMemcpy(h, d, 40, hipMemcpyDeviceToHost));
                CHK(hipFree(d));
                if (h[0]) {
                    float mx; memcpy(&mx, h + 7, 4);
                    fprintf(stderr, "rank %d: %s: %u/%zu bad, first at "
                                    "row %u col %u; by n_split %u %u %u %u; "
                                    "%u of them still zero, max|got| %.4g\n",
                            rank, tag, h[0],
                            (size_t)M * sns, h[1] / sns, h[1] % sns,
                            h[2], h[3], h[4], h[5], h[6], mx);
                }
                else
                    fprintf(stderr, "rank %d: %s: clean\n", rank, tag);
            };
            cmp("Cown vs plain", Cown, ns, rank * ns);
        }
        CHK(hipMemsetAsync(Cout, 0, c_elems * sizeof(bf16), S));
        settle();
        run_sequential(); settle();
        ok = verify(&rel_s) && ok;

        // Enough iterations to put ~4 TFLOP through each shape, so decode and
        // an 8192-token prefill are timed over comparable windows instead of
        // one of them being three launches long.
        const double flops = 2.0 * M * N * K;
        int iters = (int)(4e12 / flops);
        iters = std::max(5, std::min(200, iters));

        // Host wall clock, not hip events: the work spans several ranks and the
        // device barriers couple them, so an event on one rank's stream would
        // time a subset. The ranks enter each region together (settle() is a
        // host barrier) and are pinned inside it by sync_ranks().
        auto time_it = [&](auto &&fn) {
            for (int w = 0; w < 3; w++) fn();
            settle();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) fn();
            wait_local();
            const auto t1 = std::chrono::steady_clock::now();
            settle();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        };

        const double ms_gem = time_it(run_gemm_only);
        const double ms_seq = time_it(run_sequential);
        const double ms_fus = time_it(run_fused);

        all_ok = all_ok && ok && !barrier_failed;
        // Only rank 0 prints, so a check failing on any other rank would
        // otherwise show up as nothing but a non-zero exit code.
        if (!ok)
            fprintf(stderr, "rank %d: %s M=%d N=%d K=%d FAILED -- fused rel %.3g, "
                            "sequential rel %.3g (tolerance 5e-2; negative means "
                            "the reference had no magnitude)\n",
                    rank, sh.op, M, N, K, rel_f, rel_s);
        if (rank == 0)
            printf("%-8s %4d %-9s %6d %6d %7d %6d %5d %8.3f %8.3f %8.3f %7.2fx "
                   "%6.2f  %s\n",
                   sh.phase, sh.conc, sh.op, sh.M, M, K, K_local, N, ms_gem,
                   ms_seq, ms_fus, ms_seq / ms_fus, ms_fus / ms_gem,
                   barrier_failed ? "VOID"
                     : ok ? "ok" : (rel_f < 0 || rel_s < 0 ? "VACUOUS" : "MISMATCH"));
    }

    if (rank == 0)
        printf("\nwire this run: %.1f GB/s per rank (%s mode).\n", bw,
               bw < 20.0 ? "SLOW" : "FAST");

    (void)hipFree(rA); (void)hipFree(rB); (void)hipFree(rC);
    heap.barrier();   // nobody tears down a heap a peer is still mapping
    return all_ok ? 0 : 2;
}
