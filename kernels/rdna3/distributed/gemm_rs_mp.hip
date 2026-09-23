// Fused GEMM -> ReduceScatter, one process per rank, peer memory over IPC.
//
// gemm_rs.hip measures the same thing with one process driving every GPU. That
// is not a process model any inference framework uses -- vLLM and SGLang run
// one process per rank -- so this file re-measures in the model that would
// actually be deployed. Same device code, same shapes; the only difference is
// where the peer pointers come from.
//
//                --- why this file also measures the wire ---
//
// This node's peer bandwidth is *bimodal per process launch*: about 15.5 GB/s
// or about 24-27, nothing in between, stable to three decimals inside a run and
// flipping between runs. The same pair of GPUs, the same binary, the same
// direction, eight runs: {15.5, 27.1, 15.5, 15.5, 15.5, 27.1, 27.0, 24.0}.
// Iteration count, heap alignment, the other tenants' GPU utilisation and the
// choice of GPU pair have all been ruled out -- every pair is bimodal, PCIe is
// Gen4 x16 on both ends either way.
//
// The consequence is structural, not statistical: a fused time measured in one
// run and a bandwidth measured in another have a coin-flip chance of sitting in
// different modes, so their ratio describes the fabric's mood rather than the
// kernel. Every run therefore measures its own wire first and prints the mode
// it got in the header, and every row below it is only comparable to rows under
// the same mode.
//
// (This also settles an older question. iris_p2p.hip had read 14.5 GB/s for an
// IPC mapping against p2p_bw.hip's 27.0 for hipDeviceEnablePeerAccess, and the
// gap survived controlling for heap type, store width, buffer size, consumer
// spin and eager peer-access -- so the mapping looked like the only thing left,
// and the conclusion was a 1.9x penalty for the deployment process model. That
// was wrong. Those are simply the two modes. IPC reaches 24.0 GB/s here.)
//
// Running this:
//
//   HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./gemm_rs_mp
//   HIP_VISIBLE_DEVICES=0,3 ./run_mp.sh 2 ./gemm_rs_mp 4096 5120 6144
//
// Every rank must see every device -- hipIpcOpenMemHandle has to reach the
// exporting one -- so HIP_VISIBLE_DEVICES lists all of them and rank r drives
// visible device r. The node is shared; pin it.
#define HK_GEMM_NO_PYBIND 1
#include "../gemm/bf16fp32/gemm.cpp"

#include "ipc_heap.cuh"
#include "fused.cuh"   // micro_globals_rs, combine_n, dev_barrier

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

using namespace hk_dist;

// The generators from gemm_rs.hip, unchanged. A and B are built from *global*
// indices, so a rank can materialise its own K-slice and also the whole matrix
// for its reference with no transfers and no ambiguity about which is which.
// The values are multiples of 1/8 and small, hence exact in bf16: the two paths
// then differ only in summation order.
//
// The mixing has to be non-separable and that is not a detail. The first
// version used a = f(row, k mod 7), b = g(col, k mod 5); over each 35-element
// period the pairs (k mod 7, k mod 5) cover every combination exactly once, the
// sum factors into two marginals that are each exactly zero, and every output
// of a long K came out ~0. The comparison was measuring nothing at all.
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
    // atomicMax over the float bit patterns is exact for non-negative floats,
    // because IEEE-754 orders them the same way as their unsigned
    // representations. The version this replaced reduced (unsigned)(v*1000),
    // which wraps once these sums pass 4294 -- and the ratio of two wrapped
    // numbers came out as a clean, entirely fictitious 0.00e+00.
    atomicMax(out + 0, __float_as_uint(worst));
    atomicMax(out + 1, __float_as_uint(scale));
}

#define CHK(x) HKD_CHECK(x)

// ---------------------------------------------------------------------------
// An N-way device-side barrier, on the stream, with a deadline.
//
// A host barrier would be correct and useless: the file bootstrap costs
// milliseconds, which is the same order as the GEMMs being timed, so putting
// one inside the timing loop would measure the bootstrap. This one rides the
// stream, so a whole iteration -- GEMM, barrier, combine -- is enqueued ahead
// of the device and the host stays out of the loop, which is also the situation
// 64 transformer layers are in.
//
// Epochs are monotone, so the flags never need resetting and a late rank cannot
// be mistaken for an early one. flags[i] is the epoch rank i has reached; I
// publish mine and then wait for everyone's to catch up.
//
// The flags are in *host* memory, not in the symmetric heap, and that is the
// whole ballgame. Over a peer IPC mapping a device flag does not work at all:
// bar_probe.hip runs this exact kernel against both media, back to back, 20
// barriers each, and gets 0.008 ms on host flags against a 200 ms timeout on
// device flags. Seven other arrangements of a device-flag barrier -- one kernel
// or two, same stream or a stream each, chained by an event or by a host sync,
// signalled by a release store or by an exchange -- all cost between 1 and 14
// *seconds* per barrier. The reason is in the ISA: a system-scope acquire load
// compiles to `global_load_b32 glc` + `buffer_gl1_inv` + `buffer_gl0_inv`,
// which invalidate L0 and L1 and nothing below, so a peer's store sits in cache
// until something happens to evict it. Host memory is uncached by the GPU and
// both ranks reach the same physical page over PCIe. (Bulk data is unaffected:
// it is flushed at the end of the kernel that wrote it, which is why the fused
// epilogue's output is correct. Only spinning on a flag mid-kernel is broken.)
//
// One block, `world` threads: it spins for the whole length of the GEMM it is
// waiting behind, so it must cost the GEMM as close to nothing as possible. One
// workgroup on one WGP is the floor. Whatever it does cost is charged honestly,
// since the gemm_only baseline runs without it.
//
// dev_barrier itself is in fused.cuh, with the argument for host flags.
// The deadline is not optional. The failure mode of a rank that died is every
// survivor spinning in a kernel nothing will ever satisfy, and on this node the
// cure for that is a device reset. A bounded wait turns it into a printed
// error. The bound is in wall_clock64() ticks -- *not* clock64(), which on
// gfx1100 is a 20-bit shader-cycle counter that wraps every half millisecond
// with its high word forced to zero, making every deadline built on it
// meaningless. wall_clock64() is a real 64-bit counter at
// hipDeviceAttributeWallClockRate.

// Two things were tried here and did not work, both worth not retrying.
//
// A PCIe read-flush -- one dword read back from each peer before signalling, on
// the rule that a read completion may not pass a posted write -- changed
// nothing. A 40 MB scatter leaves across many fabric ports and one read only
// orders the one it travels on.
//
// Nontemporal peer stores (flat_store_d16_hi_b16 ... glc slc dlc, confirmed in
// the ISA) also changed nothing on their own. They are still there, and they do
// no harm, but slc is an L2 *policy* hint on gfx11, not a forced bypass.
//
// What works is keeping the inbox out of L2 entirely (uncached heap) or forcing
// the command processor to invalidate it (a host sync before the combine).

// ---------------------------------------------------------------------------
// This run's mode of the fabric, measured in this process.
//
// Ring: every rank pushes to (rank+1)%world at the same time, which is the
// traffic pattern a fused epilogue produces at world=2 and the one that loads
// the shared root complex the way the real kernel will. The number reported is
// per rank.
// ---------------------------------------------------------------------------
__global__ void push_bytes(sym_view v, float4 *dst, const float4 *src, int n4,
                           int peer) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 *remote = v.translate(dst, peer);
    for (; i < (size_t)n4; i += stride) remote[i] = src[i];
}

int main(int argc, char **argv) {
    // Unbuffered: this runs under an external timeout and a full stdio buffer
    // dies with the process.
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
    // attn_out K=6144 is 24 heads x 256 (and 48 linear-attention value heads x
    // 128, the same number); mlp_down K=17408 is intermediate_size.
    const int H = 5120, I = 17408, ATTN = 6144;
    struct shape { const char *phase; int conc; const char *op; int M, N, K; };
    std::vector<shape> shapes;
    if (argc > 3) {
        shapes.push_back({"manual", 0, "manual", atoi(argv[1]), atoi(argv[2]),
                          atoi(argv[3])});
    } else {
        // Prefill batches C requests of 2048 tokens into min(C*2048, 8192)
        // tokens per forward -- the chunk cap most engines default to, which is
        // why the shape stops growing past C=4 and concurrency turns into more
        // forwards instead. Decode shapes are omitted here: M < BLOCK_M has no
        // path through this GEMM yet, and reporting six identical "cannot tile"
        // rows crowds out the numbers that exist.
        for (int c : {1, 2, 4, 8, 16, 32}) {
            const long long tok = (long long)c * 2048;
            const int M = (int)(tok < 8192 ? tok : 8192);
            shapes.push_back({"prefill", c, "attn_out", M, H, ATTN});
            shapes.push_back({"prefill", c, "mlp_down", M, H, I});
        }
    }

    // One heap for the whole run, sized for the largest shape, with every
    // buffer allocated once and reused. A heap per shape would work too and was
    // the obvious first shape of this program, but each construction is an
    // allgather of IPC handles plus a fine-grained allocation, and doing that
    // twelve times is twelve more chances for a rank to be left behind.
    //
    // Sizes are taken from the shape list rather than hardcoded, because the
    // maximum moves with `world`: K_local shrinks as the world grows.
    size_t max_a = 0, max_b = 0, max_c = 0, max_s = 0;
    for (const auto &sh : shapes) {
        const size_t kl = (size_t)sh.K / world;
        max_a = std::max(max_a, (size_t)sh.M * kl);
        max_b = std::max(max_b, (size_t)sh.N * kl);
        max_c = std::max(max_c, (size_t)sh.M * sh.N);           // full local M x N
        max_s = std::max(max_s, (size_t)(sh.M / world) * sh.N); // one shard
    }
    // The inbox is world slots of one shard each -- world*shard == M, so it is
    // exactly one full output regardless of TP.
    const size_t max_inbox = max_c;
    // The wire probe gets its own pair of buffers at a fixed size rather than
    // borrowing Cfull and Inbox. Borrowing was the first version and it was
    // wrong twice over: it overran them on any shape whose output is under
    // 64 MB, and it would have made the probe's volume depend on the shape
    // list -- when the entire purpose of the probe is to be comparable across
    // runs, and to ipc_heap_test, which also pushes 64 MB.
    const size_t WIRE_BYTES = 64u << 20;
    const size_t wire_elems = WIRE_BYTES / sizeof(bf16);
    const size_t heap_bytes =
        (max_a + max_b + max_c + max_s + 2 * wire_elems) * sizeof(bf16)
        + (16u << 20);
    // The inbox gets a heap of its own. It is the only buffer a peer writes and
    // I read from inside a kernel, and on gfx1100 that combination needs special
    // handling either way -- see the long note in ipc_heap.cuh. Keeping it
    // separate is what makes the uncached option affordable at all: with A and B
    // in the same uncached heap the fused speedup fell from 1.28x to 1.02x.
    const size_t inbox_bytes = max_inbox * sizeof(bf16) + (4u << 20);

    // Default: an ordinary cached inbox, made visible by a host sync before the
    // combine. HK_INBOX_UNCACHED=1 takes the other correct route instead --
    // keep the inbox out of L2 and drop the sync. Measured against each other
    // on 4096x5120x6144, TP=2, both numerically correct:
    //
    //   cached + host sync    fused 2.09 ms   1.25-1.32x   f/g 1.22
    //   uncached, no sync     fused 2.57 ms   1.07x        f/g 1.49
    //
    // The sync wins because it costs one host round trip per op, while uncached
    // memory is paid on every one of the 2-byte peer stores the epilogue emits
    // -- there is no write combining left to do them in lines. Keep both: the
    // sync's cost is fixed per op rather than per byte, so decode, where the op
    // itself is ~0.1 ms, may well flip the ranking.
    const char *uc = getenv("HK_INBOX_UNCACHED");
    const bool inbox_cached = !(uc && atoi(uc));

    fprintf(stderr, "rank %d: heap %zu MB + inbox %zu MB (%s)\n", rank,
            heap_bytes >> 20, inbox_bytes >> 20,
            inbox_cached ? "cached, host sync before combine" : "uncached");
    ipc_heap heap(heap_bytes, rank, bs);
    ipc_heap ibox(inbox_bytes, rank, bs, /*uncached=*/!inbox_cached);
    bf16 *A     = heap.allocate<bf16>(max_a);
    bf16 *B     = heap.allocate<bf16>(max_b);
    bf16 *Cfull = heap.allocate<bf16>(max_c);       // the sequential path's local M x N
    bf16 *Cout  = heap.allocate<bf16>(max_s);       // my shard of the answer
    bf16 *Inbox = ibox.allocate<bf16>(max_inbox);   // [world][shard][N]
    bf16 *WireSrc = heap.allocate<bf16>(wire_elems);
    bf16 *WireDst = heap.allocate<bf16>(wire_elems);
    // Every rank must have run that exact sequence. In one process the heap
    // enforces it; in one process per rank it can only be checked, and an
    // unchecked violation produces valid-looking pointers into the wrong
    // object rather than a crash.
    heap.check_symmetry();
    ibox.check_symmetry();

    // The barrier flags, plus one word for the timeout report, in host memory.
    // The timeout word lives here too so the host can read it without a
    // hipMemcpy -- it is polled between every timed region.
    host_flags hf(MAX_RANKS + 1, bs);
    uint32_t *const flags = hf.device();
    unsigned *const tmo  = reinterpret_cast<unsigned *>(flags + MAX_RANKS);
    fprintf(stderr, "rank %d: heap ready\n", rank);

    const hipStream_t S = heap.stream();
    const sym_view V = heap.view();
    // The inbox lives in its own heap, so peer addresses for it come from
    // that heap's view. Mixing the two up would translate an inbox pointer
    // through the wrong base and land somewhere plausible but wrong.
    const sym_view VI = ibox.view();
    // 2 s in wall-clock ticks: ~200x the slowest barrier that could still be
    // called working. Loose on purpose -- it exists to terminate, not to be
    // accurate.
    int wall_khz = 0;
    CHK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate,
                              heap.device()));
    const long long BUDGET = (long long)wall_khz * 2000;

    // One kernel, on the work stream, no second stream and no event. Both of
    // those were tried, and with host flags neither is needed: the barrier
    // returns in microseconds, so nothing is gained by keeping the work stream
    // unblocked while it runs.
    uint32_t epoch = 0;
    bool barrier_failed = false;
    auto sync_ranks = [&]() {
        hipLaunchKernelGGL(dev_barrier, dim3(1), dim3(MAX_RANKS), 0, S,
                           flags, ++epoch, world, rank, BUDGET, tmo);
    };
    // Drain my stream, nothing more. This is what ends a timed region: it is
    // the last point that is still about this rank's work.
    auto wait_local = [&]() { CHK(hipStreamSynchronize(S)); };
    // Drain, then rendezvous on the host. The file bootstrap costs milliseconds
    // -- the same order as the GEMMs being timed -- so this belongs *between*
    // measurements and never inside one. An earlier version had it closing the
    // timed region and it was quietly inflating every number by the bootstrap's
    // poll interval.
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

    // What the cross-rank barrier itself costs, back to back with nothing
    // between. It is part of the fused path's time -- the combine cannot start
    // until every peer's pushes have landed -- so it has to be known separately
    // from the transfer, or a slow barrier reads as a slow wire.
    double barrier_us = 0.0;
    {
        // Progress to stderr, because this block is before the first stdout
        // line and a hang here otherwise looks like a hang in the heap.
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
        printf("=== fused GEMM->ReduceScatter, %d ranks x %s (%s), "
               "one process per rank, peer memory over IPC ===\n",
               world, prop.name, prop.gcnArchName);

    // ------------------------------------------------- this run's fabric mode
    // Measured before anything else and in this process, because that is the
    // only way the number below and the numbers further down describe the same
    // hardware state.
    double bw = 0.0;
    {
        const size_t bytes = WIRE_BYTES;
        const int n4 = (int)(bytes / 16);
        const int blocks = std::min((n4 + 255) / 256, 1024);
        const int peer = (rank + 1) % world;
        const int reps = 50;   // 10 is a sample of a distribution, not a measurement
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
        bw = bytes / (ms * 1e-3) / 1e9;
        if (rank == 0)
            printf("\nwire, measured in this process: %.1f GB/s per rank "
                   "(ring, %d MB pushes).\nThis node is bimodal per launch -- "
                   "~15.5 or ~24-27 GB/s, nothing between -- so this is the\n"
                   "%s mode, and the rows below are only comparable to other "
                   "rows measured in it.\n",
                   bw, (int)(bytes >> 20), bw < 20.0 ? "SLOW" : "FAST");
    }
    if (rank == 0)
        printf("cross-rank device barrier: %.1f us each, back to back.\n",
               barrier_us);

    // The reference: the whole GEMM over the full K, on this rank's own device.
    // Each rank builds it independently rather than one rank building it and
    // shipping slices, because the generators are deterministic and a local
    // 0.5 GB allocation is cheaper than a cross-rank transfer plus the code to
    // orchestrate it. Off-heap: nobody else has to name it.
    bf16 *rA = nullptr, *rB = nullptr, *rC = nullptr;
    {
        size_t ra = 0, rb = 0;
        for (const auto &sh : shapes) {
            ra = std::max(ra, (size_t)sh.M * sh.K);
            rb = std::max(rb, (size_t)sh.N * sh.K);
        }
        CHK(hipMalloc(&rA, ra * sizeof(bf16)));
        CHK(hipMalloc(&rB, rb * sizeof(bf16)));
        CHK(hipMalloc(&rC, max_c * sizeof(bf16)));
    }

    if (rank == 0) {
        printf("\ngemm_ms is the %d GEMMs with no communication and no reduce at "
               "all -- not a\ncorrect implementation, just the floor overlap is "
               "aiming at. fus/gemm of 1.00\nwould mean the transfer vanished "
               "completely behind the math.\n\n", world);
        printf("%-8s %4s %-9s %6s %7s %6s %5s %8s %8s %8s %8s %6s %7s  %s\n",
               "phase", "conc", "op", "M", "K", "K_loc", "N", "gemm_ms",
               "seq_ms", "fus_ms", "speedup", "f/g", "TFLOPs", "check");
    }

    bool all_ok = true;
    for (const auto &sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K;
        const int K_local = K / world, shard = M / world;
        const size_t c_elems = (size_t)M * N, s_elems = (size_t)shard * N;

        // What is left after the M-remainder path. The GEMM no longer needs
        // M % BLOCK_M -- the last block backs up to M - BLOCK_M and recomputes
        // the overlap idempotently -- but it does still need M >= BLOCK_M for
        // that backup to stay in range, and the fused epilogue still assigns
        // ownership one warp tile at a time, so neither the shard boundary nor
        // the backed-up M start may fall inside a tile. m_base inherits its
        // alignment from M, hence the M % REG_BLOCK_M check.
        constexpr int BM = big_config::BLOCK_M, RBM = big_config::REG_BLOCK_M;
        const char *why = nullptr;
        if (K % world || M % world)            why = "K or M does not split across ranks";
        else if (M < BM)                       why = "M < BLOCK_M (decode regime)";
        else if (M % RBM)                      why = "M is not a multiple of REG_BLOCK_M";
        else if (shard % RBM)                  why = "the shard boundary falls inside a warp tile";
        else if (N % big_config::BLOCK_N)      why = "N does not tile";
        else if (K_local % big_config::K_STEP) why = "K_local does not tile";
        if (why) {
            if (rank == 0)
                printf("%-8s %4d %-9s %6d %7d %6d %5d   -- %s\n",
                       sh.phase, sh.conc, sh.op, M, K, K_local, N, why);
            continue;
        }

        CHK(hipMemsetAsync(Inbox, 0, c_elems * sizeof(bf16), S));
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

        // c_sym is my *slot* of the inbox. Translated into the owner's address
        // space that is the owner's slot for me, so world-1 senders never
        // collide and no rank has to know which slot it is writing.
        micro_globals_rs gf{_gl_A(A, 1, 1, M, K_local),
                            _gl_B(B, 1, 1, N, K_local),
                            _gl_C(Cout, 1, 1, shard, N), VI,
                            Inbox + (size_t)rank * s_elems, shard, rank, S};
        micro_globals gs{_gl_A(A, 1, 1, M, K_local), _gl_B(B, 1, 1, N, K_local),
                         _gl_C(Cfull, 1, 1, M, N), S};

        // The barrier between the GEMMs and the combine is load-bearing: my
        // combine reads inbox slots that other ranks' GEMMs write, and stream
        // order on my device says nothing about a kernel on theirs. What it
        // costs is the tail of the slowest rank, not the transfer -- the pushes
        // themselves are already spread over the whole GEMM, which is the point.
        //
        // The trailing barrier is the other half of the same argument, and it
        // is not optional either: without it a peer's *next* GEMM can start
        // pushing into my inbox while my combine is still reading it. Nothing
        // orders those two, because the only thing between them is my own
        // stream. Two barriers cost ~11 us against a 1.8 ms iteration. The way
        // to get rid of the second one is double-buffered inboxes, not
        // dropping it.
        // Neither barrier is what made the combine read garbage, and it took a
        // while to stop blaming them. The barrier is a *control* rendezvous: it
        // proves the peer's GEMM finished. It cannot make the peer's bytes
        // visible to my combine, because on gfx1100 a peer's write lands in my
        // HBM without invalidating my L2, and no instruction a kernel can issue
        // invalidates L2. The only thing that does is the command processor's
        // system-scope acquire at dispatch, which the runtime emits after a host
        // sync -- so a bare hipStreamSynchronize here fixes the read and a
        // nontemporal load inside combine_n does not, even though it compiles to
        // the bypass bits. The alternative is to keep the inbox out of L2 in the
        // first place, which is what the uncached ibox heap does; then no sync
        // is needed. Both are correct, they cost differently, and
        // HK_INBOX_CACHED picks between them so the two can be measured against
        // each other in one build.
        auto run_fused = [&]() {
            CHK(hipMemsetAsync(Cout, 0, s_elems * sizeof(bf16), S));
            dispatch_any(gf);
            // Two syncs, and they do different jobs -- one alone is not enough,
            // which showed up as an intermittent failure on the first shape of
            // a run. The first is a *release*: draining my stream from the host
            // is what gets my pushes out of my own L2 and onto the wire, so the
            // flag I am about to raise does not outrun them. The second is an
            // *acquire*: the runtime puts a system-scope acquire on the first
            // dispatch after a host sync, and that is the only thing on gfx1100
            // that invalidates my L2, so it has to sit immediately before the
            // combine rather than anywhere earlier -- the acquire rides on the
            // next dispatch only, and sync_ranks() would otherwise consume it.
            if (inbox_cached) CHK(hipStreamSynchronize(S));
            sync_ranks();
            if (inbox_cached) CHK(hipStreamSynchronize(S));
            hipLaunchKernelGGL(combine_n, dim3(1024), dim3(256), 0, S,
                               Cout, Inbox, s_elems, world, rank);
            sync_ranks();
        };

        // The unfused path: a full local M x N, then move every other rank's
        // shard of it across, then the same combine. The copies sit on my own
        // stream behind my own GEMM, which is exactly the serialization fusing
        // removes. hipMemcpyAsync and not hipMemcpyPeerAsync -- the destination
        // is an IPC mapping, an ordinary device pointer in my address space.
        auto run_sequential = [&]() {
            dispatch_any(gs);
            for (int p = 0; p < world; p++) {
                if (p == rank) continue;
                CHK(hipMemcpyAsync(VI.translate(Inbox + (size_t)rank * s_elems, p),
                                   Cfull + (size_t)p * shard * N,
                                   s_elems * sizeof(bf16),
                                   hipMemcpyDeviceToDevice, S));
            }
            sync_ranks();
            // My own shard is already sitting inside Cfull, so this reduces in
            // place there rather than into Cout.
            hipLaunchKernelGGL(combine_n, dim3(1024), dim3(256), 0, S,
                               Cfull + (size_t)rank * shard * N, Inbox, s_elems,
                               world, rank);
            sync_ranks();   // same inbox-reuse hazard as the fused path
        };

        // The ceiling: the same GEMMs with no communication and no combine.
        // Not a correct implementation of anything -- it drops the reduce -- but
        // without it a speedup over the unfused path cannot be read as progress
        // toward anything, only as a ratio between two arbitrary numbers.
        auto run_gemm_only = [&]() { dispatch_any(gs); };

        auto verify = [&](const bf16 *got, double *out_rel) {
            unsigned *d; CHK(hipMalloc(&d, 8)); CHK(hipMemset(d, 0, 8));
            hipLaunchKernelGGL(err_scale, dim3(1024), dim3(256), 0, S, got,
                               rC + (size_t)rank * shard * N, s_elems, d);
            CHK(hipStreamSynchronize(S));
            unsigned h[2] = {0, 0};
            CHK(hipMemcpy(h, d, 8, hipMemcpyDeviceToHost));
            CHK(hipFree(d));
            float fh[2]; memcpy(fh, h, 8);
            // A reference with no magnitude makes the ratio meaningless and
            // would read as a perfect score -- which is exactly what the
            // separable generator produced. The terms are order 0.1 and there
            // are thousands of them, so max|ref| below 1 means the generator is
            // cancelling rather than the GEMM working.
            if (!(fh[1] > 1.f)) { *out_rel = -1.0; return false; }
            *out_rel = (double)fh[0] / fh[1];
            // bf16 has 8 significand bits, so one rounding is up to 2^-8 =
            // 3.9e-3 relative. The reference rounds once; the distributed path
            // rounds each of `world` partials and then their sum, and where the
            // K-slices cancel the partials are larger than the answer. 5e-2 is
            // ~4x headroom over that and still 20x below what any routing bug
            // costs: a misrouted tile delivers some other rank's partial, wrong
            // by order 100% of scale, not by parts per hundred.
            return *out_rel < 5e-2;
        };

        // max|x| over a buffer, for locating a failure. err_scale's second
        // output is max|ref|, so passing the same pointer twice gives exactly
        // that and needs no extra kernel.
        auto magnitude = [&](const bf16 *p, size_t n) {
            unsigned *d; CHK(hipMalloc(&d, 8)); CHK(hipMemset(d, 0, 8));
            hipLaunchKernelGGL(err_scale, dim3(1024), dim3(256), 0, S, p, p, n, d);
            CHK(hipStreamSynchronize(S));
            unsigned h[2] = {0, 0};
            CHK(hipMemcpy(h, d, 8, hipMemcpyDeviceToHost));
            CHK(hipFree(d));
            float fh[2]; memcpy(fh, h, 8);
            return (double)fh[1];
        };

        double rel_f = 0, rel_s = 0;
        // Nobody may push before everybody has zeroed. The memset above is on
        // my stream and says nothing about a peer's, so without this the rank
        // that starts late wipes the pushes the rank that started early has
        // already delivered -- and its combine then sums one partial short.
        // That is a startup-only race (the steady state is covered by the
        // barriers inside run_fused), and it showed up as exactly one rank
        // failing its check with rel ~0.58 while the other passed.
        settle();
        run_fused();  settle();
        bool ok = verify(Cout, &rel_f);
        // The diagnostics that used to sit here -- inbox magnitudes, inbox vs
        // the peer's recomputed partial, locally-kept half vs my own partial --
        // are gone now that they have answered their question. Every one of them
        // read rel 0 on bytes the combine had just summed wrongly, which is what
        // established that the inbox was correct on arrival and the combine was
        // reading a stale L2 copy of it. They also re-ran two GEMMs and pushed
        // into a peer already past its own check, so they could not stay.
        run_sequential(); settle();
        ok = verify(Cfull + (size_t)rank * shard * N, &rel_s) && ok;

        // Enough iterations to put ~4 TFLOP through each shape, so a 2048-token
        // attn_out and an 8192-token mlp_down are timed over comparable windows
        // instead of one of them being three launches long.
        const double flops = 2.0 * M * N * K;
        int iters = (int)(4e12 / flops);
        iters = std::max(5, std::min(100, iters));

        // Host wall clock, not hip events: the work spans several ranks and the
        // device barriers couple them, so an event on one rank's stream would
        // time a subset. The ranks enter each timed region together (settle()
        // is a host barrier) and are pinned together inside it by sync_ranks().
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
        // Only rank 0 prints the table, so a check that fails on any other rank
        // would otherwise show up as nothing but a non-zero exit code.
        if (!ok)
            fprintf(stderr, "rank %d: %s M=%d N=%d K=%d FAILED -- fused rel %.3g, "
                            "sequential rel %.3g (tolerance 5e-2; negative means "
                            "the reference had no magnitude)\n",
                    rank, sh.op, M, N, K, rel_f, rel_s);
        if (rank == 0)
            printf("%-8s %4d %-9s %6d %7d %6d %5d %8.3f %8.3f %8.3f %7.2fx "
                   "%6.2f %7.1f  %s\n",
                   sh.phase, sh.conc, sh.op, M, K, K_local, N, ms_gem, ms_seq,
                   ms_fus, ms_seq / ms_fus, ms_fus / ms_gem,
                   flops / (ms_fus * 1e-3) / 1e12,
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
