// Fused GEMM -> ReduceScatter on RDNA3, two ranks, on the shapes an actual
// TP=2 deployment produces.
//
// The shape of the problem. In a tensor-parallel row-parallel linear layer, K
// is split across ranks: rank r holds A[:, K_r] and B[:, K_r], computes a full
// M x N partial product, and the answer is the sum of the partials with each
// rank keeping only its own row shard of that sum. Done the obvious way that is
// a GEMM, then a barrier, then a reduce-scatter -- the wire is idle while the
// GEMM runs and the CUs are idle while the wire runs.
//
// The fused version writes each output tile to its owner the moment it leaves
// the accumulator: tiles in my shard go to my own output, tiles in the peer's
// shard go straight into the peer's inbox across PCIe. The transfer is then
// spread over the whole GEMM instead of following it, and all that is left at
// the end is a local add of the inbox into my shard.
//
// Two ranks, not eight, and that is a measurement rather than a simplification.
// symmem_test puts the aggregate push bandwidth of this node at ~36 GB/s and
// *flat* -- 2 ranks get ~17 GB/s each, 4 ranks get ~9 -- so the ceiling is the
// shared root complex and per-rank bandwidth falls as the ring grows. Against a
// 77 TFLOP/s GEMM the break-even arithmetic intensity is ~4300 FLOP/byte at 2
// ranks and ~17000 at 8, and for a GEMM whose output is reduce-scattered that
// number is exactly K_local. At 8 ranks this node is fabric-bound however the
// kernel is written.
//
// What is measured, per shape:
//
//   fused       GEMM with the reduce-scatter in its epilogue, then combine
//   sequential  GEMM writing a full local M x N, then a bulk peer copy of the
//               peer's shard, then the same combine -- identical work, not
//               overlapped
//
// Both are verified against a single-GPU GEMM over the full K. That is the same
// kernel, so what is under test is the sharding and the peer stores, not the
// math.
//
//                         --- the shapes ---
//
// Qwen3.8-27B (Qwen3_5ForConditionalGeneration), from its config.json:
// hidden_size 5120, intermediate_size 17408, 64 layers in a 3:1 linear/full
// attention pattern, num_attention_heads 24 x head_dim 256 = 6144, and linear
// attention with 48 value heads x 128 = 6144 as well. Under TP=2 the layers
// that need a reduce-scatter are the row-parallel ones, and there are exactly
// two distinct shapes, both with N = hidden_size = 5120:
//
//   attn_out   K = 6144   (o_proj; the same 6144 for full and linear layers)
//   mlp_down   K = 17408  (down_proj)
//
// M is the number of tokens in the batch, which is where concurrency enters:
// prefill batches C requests of L tokens into min(C*L, chunk cap) tokens per
// forward, decode batches C requests into exactly C tokens.
//
// Running this: two idle GPUs, pinned. HIP_VISIBLE_DEVICES=0,3 ./gemm_rs, under
// an external timeout. The node is shared.
#define HK_GEMM_NO_PYBIND 1
#include "../gemm/bf16fp32/gemm.cpp"

#include "symmem.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace hk_dist;

static constexpr int WORLD = 2;

// Globals for the fused epilogue. Same a and b as micro_globals; c is now only
// my shard rather than the whole output, plus where the peer's half goes.
struct micro_globals_rs {
    _gl_A a;
    _gl_B b;
    _gl_C c;        // my shard          (shard_rows, N), local
    _gl_C c_peer;   // the peer's inbox  (shard_rows, N), on the peer's GPU
    int shard_rows;
    int my_rank;
    hipStream_t stream;

    static constexpr bool FUSED_RS = true;
};

// A and B are generated from their *global* indices rather than copied from a
// host buffer, so a rank can build its own K-slice and the reference can build
// the whole thing with no transfers and no question about which is which. The
// values are multiples of 1/8 and small, so they are exact in bf16 and the two
// paths differ only in summation order.
//
// The mixing matters, and the first attempt here got it wrong in a way worth
// recording. It used a = ((row*31 + k*17) % 7 - 3)/8 and b = ((col*13 + k*29) %
// 5 - 2)/8, which are separable: a depends on k only through k mod 7 and b only
// through k mod 5. Over each 35-element period the pairs (k mod 7, k mod 5)
// cover every combination exactly once, so the sum factors into two marginals
// that are each exactly zero. Every output of an 8192-long K was therefore ~0,
// max|ref| came out as 0.125, and the comparison was measuring nothing at all.
// A hash that mixes row and k together has no such structure, and the resulting
// outputs have a scale of a few, as a sum of ~8192 mean-zero terms should.
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

// out += inbox, through fp32. The partials are bf16 because that is what the
// GEMM epilogue emits; adding them as bf16 would throw away most of what little
// mantissa there is, and going through float costs nothing in a pass that is
// entirely memory-bound.
__global__ void combine(bf16 *out, const bf16 *inbox, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = __float2bfloat16(__bfloat162float(out[i]) + __bfloat162float(inbox[i]));
}

// Error against the reference, as max|diff| and max|ref| separately.
//
// Per-element relative error would be the obvious thing and is wrong here: the
// two K-halves partially cancel, so individual outputs land arbitrarily close
// to zero and a tiny absolute error there reads as an enormous relative one.
// What bf16 rounding actually bounds is the error relative to the *scale of the
// data*, so that is what gets reported.
//
// The reduction is an atomicMax over the float bit patterns, which is exact for
// non-negative floats because IEEE-754 orders them the same way as their
// unsigned representations. The first version reduced (unsigned)(v * 1000)
// instead and that has a range: at K=8192 these sums pass 4294, the scaled
// integer wraps, and the ratio of two wrapped numbers came out as a clean and
// entirely fictitious 0.00e+00. A verification that can silently report success
// is worse than no verification.
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
    atomicMax(out + 0, __float_as_uint(worst));
    atomicMax(out + 1, __float_as_uint(scale));
}

#define CHK(x) HKD_CHECK(x)

// ---------------------------------------------------------------------------
// One shape, both paths, verified and timed. Prints one table row.
// ---------------------------------------------------------------------------
static bool run_shape(const char *phase, int conc, const char *op,
                      int M, int N, int K) {
    const int K_local = K / WORLD;
    const int shard   = M / WORLD;

    // What this GEMM can express. Two separate limits, worth distinguishing
    // because only one of them is about the fusion:
    //
    //   * M % BLOCK_M -- the GEMM itself has no M-remainder path, so a batch of
    //     fewer than BLOCK_M tokens cannot run at all, fused or not. This is
    //     what rules out every decode shape.
    //   * shard % REG_BLOCK_M -- the fused epilogue assigns ownership one warp
    //     tile at a time, so a shard boundary inside a tile would split it.
    //     Implied by the first for WORLD=2, checked anyway.
    constexpr int BM = big_config::BLOCK_M, RBM = big_config::REG_BLOCK_M;
    if (M % BM || shard % RBM) {
        printf("%-8s %4d %-9s %6d %7d %6d %5d   -- M=%d is not a multiple of "
               "BLOCK_M=%d; this GEMM has no M-remainder path\n",
               phase, conc, op, M, K, K_local, N, M, BM);
        return true;   // a limitation to report, not a failure to verify
    }
    if (N % big_config::BLOCK_N || K_local % big_config::K_STEP) {
        printf("%-8s %4d %-9s %6d %7d %6d %5d   -- N or K_local does not tile\n",
               phase, conc, op, M, K, K_local, N);
        return true;
    }

    const size_t a_elems = (size_t)M * K_local;
    const size_t b_elems = (size_t)N * K_local;
    const size_t c_elems = (size_t)M * N;          // full local output, baseline
    const size_t s_elems = (size_t)shard * N;      // one shard

    // One symmetric heap holds everything a peer has to be able to name. Only
    // the inbox strictly has to be symmetric, but allocating in the same order
    // on every rank is what makes translate() a subtraction, so it all goes in.
    const size_t per_rank =
        (a_elems + b_elems + c_elems + 2 * s_elems) * sizeof(bf16) + (16u << 20);
    sym_heap heap(per_rank, WORLD);

    auto A     = heap.allocate_all<bf16>(a_elems);
    auto B     = heap.allocate_all<bf16>(b_elems);
    auto Cfull = heap.allocate_all<bf16>(c_elems);   // the baseline's local M x N
    auto Cout  = heap.allocate_all<bf16>(s_elems);   // my shard of the answer
    auto Inbox = heap.allocate_all<bf16>(s_elems);   // the peer's partial of it

    for (int r = 0; r < WORLD; r++) {
        CHK(hipSetDevice(r));
        hipLaunchKernelGGL(fill_a, dim3(1024), dim3(256), 0, heap.stream(r),
                           A[r], M, K_local, r * K_local);
        hipLaunchKernelGGL(fill_b, dim3(1024), dim3(256), 0, heap.stream(r),
                           B[r], N, K_local, r * K_local);
    }
    heap.sync_all();

    // ---------------------------------------------------------------- reference
    bf16 *rA, *rB, *rC;
    CHK(hipSetDevice(0));
    CHK(hipMalloc(&rA, (size_t)M * K * sizeof(bf16)));
    CHK(hipMalloc(&rB, (size_t)N * K * sizeof(bf16)));
    CHK(hipMalloc(&rC, c_elems * sizeof(bf16)));
    hipLaunchKernelGGL(fill_a, dim3(1024), dim3(256), 0, 0, rA, M, K, 0);
    hipLaunchKernelGGL(fill_b, dim3(1024), dim3(256), 0, 0, rB, N, K, 0);
    CHK(hipDeviceSynchronize());
    {
        micro_globals g{_gl_A(rA, 1, 1, M, K), _gl_B(rB, 1, 1, N, K),
                        _gl_C(rC, 1, 1, M, N), nullptr};
        dispatch_micro(g);
        CHK(hipDeviceSynchronize());
    }

    // ------------------------------------------------------------------- fused
    // c_peer is built once, on the host: the peer's Inbox address translated
    // into this rank's address space. Nothing on the device does any address
    // arithmetic -- the epilogue's peer store is an ordinary vectorized global
    // store that happens to land on another GPU.
    std::vector<micro_globals_rs> gr;
    for (int r = 0; r < WORLD; r++) {
        const int peer = 1 - r;
        bf16 *peer_inbox = heap.view(r).translate(Inbox[r], peer);
        gr.push_back(micro_globals_rs{
            _gl_A(A[r], 1, 1, M, K_local), _gl_B(B[r], 1, 1, N, K_local),
            _gl_C(Cout[r], 1, 1, shard, N), _gl_C(peer_inbox, 1, 1, shard, N),
            shard, r, heap.stream(r)});
    }

    // The sync between the GEMMs and the combines is a real barrier, not an
    // oversight: rank 0's combine reads an inbox that rank 1's GEMM writes, and
    // stream order on rank 0 says nothing about a kernel on rank 1. Kernel
    // completion plus a host sync is what makes those peer stores visible.
    // Inside the GEMM the pushes are still fully overlapped with the math,
    // which is the whole point; what this barrier costs is the tail of the
    // slower rank, not the transfer.
    auto run_fused = [&]() {
        for (int r = 0; r < WORLD; r++) { CHK(hipSetDevice(r)); dispatch_any(gr[r]); }
        heap.sync_all();
        for (int r = 0; r < WORLD; r++) {
            CHK(hipSetDevice(r));
            hipLaunchKernelGGL(combine, dim3(1024), dim3(256), 0, heap.stream(r),
                               Cout[r], Inbox[r], s_elems);
        }
        heap.sync_all();
    };

    // -------------------------------------------------------------- sequential
    // GEMM into a full local M x N, then move the peer's shard across, then the
    // same combine. The copy sits on the rank's own stream behind its own GEMM,
    // which is exactly the serialization the fused version removes.
    std::vector<micro_globals> gs;
    for (int r = 0; r < WORLD; r++)
        gs.push_back(micro_globals{_gl_A(A[r], 1, 1, M, K_local),
                                   _gl_B(B[r], 1, 1, N, K_local),
                                   _gl_C(Cfull[r], 1, 1, M, N), heap.stream(r)});

    // The ceiling: the same two GEMMs with no communication at all and no
    // combine. This is not a correct implementation of anything -- it drops the
    // reduce -- it is the floor time that any overlap scheme is trying to reach.
    // Without it a speedup over the unfused path cannot be read as progress
    // toward anything, only as a ratio between two arbitrary numbers.
    auto run_gemm_only = [&]() {
        for (int r = 0; r < WORLD; r++) { CHK(hipSetDevice(r)); dispatch_any(gs[r]); }
        heap.sync_all();
    };

    auto run_sequential = [&]() {
        for (int r = 0; r < WORLD; r++) { CHK(hipSetDevice(r)); dispatch_any(gs[r]); }
        for (int r = 0; r < WORLD; r++) {
            const int peer = 1 - r;
            CHK(hipSetDevice(r));
            // My copy of the peer's shard starts at row peer*shard.
            CHK(hipMemcpyPeerAsync(Inbox[peer], peer,
                                   Cfull[r] + (size_t)peer * shard * N, r,
                                   s_elems * sizeof(bf16), heap.stream(r)));
        }
        heap.sync_all();
        for (int r = 0; r < WORLD; r++) {
            CHK(hipSetDevice(r));
            // The baseline's own shard is still sitting inside Cfull, so this
            // reduces in place there rather than into Cout.
            hipLaunchKernelGGL(combine, dim3(1024), dim3(256), 0, heap.stream(r),
                               Cfull[r] + (size_t)r * shard * N, Inbox[r], s_elems);
        }
        heap.sync_all();
    };

    // ------------------------------------------------------------------ verify
    // got[r] points at rank r's finished shard, wherever that path left it.
    auto verify = [&](std::vector<bf16 *> got, double *out_rel) {
        double worst = 0.0;
        for (int r = 0; r < WORLD; r++) {
            CHK(hipSetDevice(r));
            unsigned *d; CHK(hipMalloc(&d, 8)); CHK(hipMemset(d, 0, 8));
            // The reference lives on GPU 0; rank r reads its own slice of it
            // across the link.
            bf16 *staged = nullptr;
            const bf16 *ref = rC + (size_t)r * shard * N;
            if (r != 0) {
                CHK(hipMalloc(&staged, s_elems * sizeof(bf16)));
                CHK(hipMemcpyPeer(staged, r, rC + (size_t)r * shard * N, 0,
                                  s_elems * sizeof(bf16)));
                ref = staged;
            }
            hipLaunchKernelGGL(err_scale, dim3(1024), dim3(256), 0, 0,
                               got[r], ref, s_elems, d);
            CHK(hipDeviceSynchronize());
            unsigned h[2] = {0, 0};
            CHK(hipMemcpy(h, d, 8, hipMemcpyDeviceToHost));
            CHK(hipFree(d));
            if (staged) CHK(hipFree(staged));
            float fh[2];
            memcpy(fh, h, 8);
            // A reference that is zero, or near enough, makes the ratio
            // meaningless and would read as a perfect score. The terms being
            // summed are order 0.1 and there are thousands of them, so a
            // reference whose largest element is below 1 means the generator is
            // cancelling rather than the GEMM working -- which is exactly what
            // happened with the separable generator this file used first.
            if (!(fh[1] > 1.f)) { *out_rel = -1.0; return false; }
            if ((double)fh[0] / fh[1] > worst) worst = (double)fh[0] / fh[1];
        }
        *out_rel = worst;
        // Where the bound comes from. bf16 has 8 significand bits, so a single
        // rounding is up to 2^-8 = 3.9e-3 relative. The reference rounds once;
        // the distributed path rounds each of the two partials and then their
        // sum, and where the K-halves cancel the partials are larger than the
        // answer. 5e-2 is ~4x headroom over that while staying 20x below what
        // any routing bug costs: a misrouted tile delivers some other rank's
        // partial, wrong by order 100% of scale, not by parts per hundred.
        return worst < 5e-2;
    };

    double rel_f = 0, rel_s = 0;
    run_fused();
    bool ok = verify(Cout, &rel_f);
    run_sequential();
    std::vector<bf16 *> seq_out(WORLD);
    for (int r = 0; r < WORLD; r++) seq_out[r] = Cfull[r] + (size_t)r * shard * N;
    ok = verify(seq_out, &rel_s) && ok;

    // -------------------------------------------------------------------- time
    // Enough iterations to put ~4 TFLOP through each shape, so a 512-token
    // attn_out and a 16384-token mlp_down are timed over comparable windows
    // instead of one of them being three launches long.
    const double flops = 2.0 * M * N * K;
    int iters = (int)(4e12 / flops);
    if (iters < 5) iters = 5;
    if (iters > 100) iters = 100;

    // Host wall clock, not hip events: the work spans two devices and several
    // streams and every iteration ends in a host sync anyway, so an event on
    // one device's stream would be timing a subset of it.
    auto time_it = [&](auto &&fn) {
        for (int w = 0; w < 3; w++) fn();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) fn();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    const double ms_gem = time_it(run_gemm_only);
    const double ms_seq = time_it(run_sequential);
    const double ms_fus = time_it(run_fused);

    // ms_fus/ms_gem is the number that matters: 1.00 would mean the transfer
    // has disappeared entirely behind the math.
    printf("%-8s %4d %-9s %6d %7d %6d %5d %8.3f %8.3f %8.3f %7.2fx %6.2f %7.1f  %s\n",
           phase, conc, op, M, K, K_local, N, ms_gem, ms_seq, ms_fus,
           ms_seq / ms_fus, ms_fus / ms_gem, flops / (ms_fus * 1e-3) / 1e12,
           ok ? "ok" : (rel_f < 0 || rel_s < 0 ? "VACUOUS" : "MISMATCH"));

    CHK(hipSetDevice(0));
    (void)hipFree(rA); (void)hipFree(rB); (void)hipFree(rC);
    return ok;
}

// Push into the peer's heap with ordinary stores, the way a fused epilogue
// does. This is the same traffic hipMemcpyPeerAsync moves, with none of the
// copy-engine setup.
__global__ void push_bytes(sym_view v, float4 *dst, const float4 *src, int n4,
                           int peer) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 *remote = v.translate(dst, peer);
    for (; i < (size_t)n4; i += stride) remote[i] = src[i];
}

// ---------------------------------------------------------------------------
// What the transfer alone costs at a given volume, along two axes.
//
// The decode shapes cannot run through the GEMM at all, so this is the only
// part of them that is measurable here -- and it is easy to misread, because
// two quite different costs are superimposed and the first version of this
// function reported their sum as though it were one number.
//
// Axis 1, memcpy vs in-kernel push: a fused epilogue writes to the peer with
// ordinary stores, so if the copy engine's setup were the expensive part,
// fusing would remove it and the push column would be faster.
//
// Axis 2, blocking vs pipelined, and this is the one that was missing: every
// iteration used to end in a host sync on both devices. That made a 10 KB
// transfer and a 160 KB transfer cost the same ~40 us -- and the natural
// reading, "the fabric has a 40 us latency floor", was wrong. Most of it was
// the host round trip. A decode loop does not pay that: 64 layers are
// submitted ahead of the device and the host is not between them. Enqueuing
// the transfers back to back and synchronising once amortises the round trip
// away and leaves the part a real pipeline actually pays.
// ---------------------------------------------------------------------------
static void comm_floor(size_t bytes) {
    bytes = (bytes + 15) & ~size_t(15);   // whole float4s for the push kernel
    // Two buffers of `bytes` each, so the heap has to hold both. Sizing it for
    // one was survivable at 10-160 KB purely because the 1 MB of slack happened
    // to exceed the second allocation, and aborted the moment a prefill-scale
    // volume was asked for.
    sym_heap heap(2 * bytes + (1u << 20), WORLD);
    auto src = heap.allocate_all<float4>(bytes / 16);
    auto dst = heap.allocate_all<float4>(bytes / 16);
    const int n4 = (int)(bytes / 16);
    // Enough blocks to cover the payload, capped: at 10 KB this is 3 blocks,
    // and launching 1024 of them would be measuring block dispatch.
    const int blocks = (n4 + 255) / 256 < 1024 ? (n4 + 255) / 256 : 1024;

    // Enqueue only -- the sync is the caller's, because whether there is one
    // per transfer is the entire question here.
    auto memcpy_enq = [&]() {
        for (int r = 0; r < WORLD; r++) {
            CHK(hipSetDevice(r));
            CHK(hipMemcpyPeerAsync(dst[1 - r], 1 - r, src[r], r, bytes,
                                   heap.stream(r)));
        }
    };
    auto push_enq = [&]() {
        for (int r = 0; r < WORLD; r++) {
            CHK(hipSetDevice(r));
            hipLaunchKernelGGL(push_bytes, dim3(blocks), dim3(256), 0,
                               heap.stream(r), heap.view(r), dst[r], src[r],
                               n4, 1 - r);
        }
    };

    const int iters = 200;
    // One transfer, then a host sync on both devices, per iteration. This is
    // what a layer pays if it waits for its collective before going on.
    auto time_blocking = [&](auto &&enq) {
        for (int w = 0; w < 10; w++) { enq(); heap.sync_all(); }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) { enq(); heap.sync_all(); }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };
    // All the transfers enqueued back to back, one sync at the end. The host
    // round trip is amortised to nothing, which is the situation a decode loop
    // is actually in: 64 layers are submitted ahead and the host is not in the
    // loop. The difference between the two columns is the cost of synchronising,
    // not the cost of moving the bytes.
    auto time_pipelined = [&](auto &&enq) {
        for (int w = 0; w < 10; w++) enq();
        heap.sync_all();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) enq();
        heap.sync_all();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };

    const double blk_cpy  = time_blocking(memcpy_enq);
    const double blk_push = time_blocking(push_enq);
    const double pip_cpy  = time_pipelined(memcpy_enq);
    const double pip_push = time_pipelined(push_enq);
    printf("  %9.1f KB %8.3f %8.3f   %8.3f %8.3f %8.2f\n", bytes / 1024.0,
           blk_cpy, blk_push, pip_cpy, pip_push,
           bytes / (pip_push * 1e-3) / 1e9);
}

int main(int argc, char **argv) {
    // Unbuffered: this is meant to be run under an external timeout, and a full
    // stdio buffer dies with the process.
    setvbuf(stdout, nullptr, _IONBF, 0);

    int avail = 0; CHK(hipGetDeviceCount(&avail));
    if (avail < WORLD) { fprintf(stderr, "need %d GPUs, see %d\n", WORLD, avail); return 1; }
    hipDeviceProp_t prop; CHK(hipGetDeviceProperties(&prop, 0));
    printf("=== fused GEMM->ReduceScatter, %d x %s (%s) ===\n",
           WORLD, prop.name, prop.gcnArchName);

    // One explicit shape, for poking at something the sweep does not cover.
    if (argc > 3) {
        printf("%-8s %4s %-9s %6s %7s %6s %5s %8s %8s %8s %7s\n",
               "phase", "conc", "op", "M", "K", "K_loc", "N", "seq_ms",
               "fus_ms", "speedup", "TFLOPs");
        return run_shape("manual", 0, "manual", atoi(argv[1]), atoi(argv[2]),
                         atoi(argv[3])) ? 0 : 2;
    }

    // Qwen3.8-27B under TP=2. H=5120, I=17408, attention output width 6144 for
    // both the full and the linear layers.
    const int H = 5120, I = 17408, ATTN = 6144;

    printf("\nQwen3.8-27B, TP=2. N = hidden = %d; attn_out K = %d, mlp_down K = %d.\n",
           H, ATTN, I);
    printf("Prefill: C requests of 2048 tokens, batched to min(C*2048, 8192) --\n"
           "the chunk cap most engines default to, which is why the shape stops\n"
           "growing past C=4 and concurrency turns into more forwards instead.\n"
           "Decode: exactly C tokens per forward.\n\n");

    printf("gemm_ms is the two GEMMs with no communication and no reduce at "
           "all -- not a\ncorrect implementation, just the floor overlap is "
           "aiming at. fus/gemm of 1.00\nwould mean the transfer vanished "
           "completely behind the math.\n\n");
    printf("%-8s %4s %-9s %6s %7s %6s %5s %8s %8s %8s %8s %6s %7s  %s\n",
           "phase", "conc", "op", "M", "K", "K_loc", "N", "gemm_ms", "seq_ms",
           "fus_ms", "speedup", "f/g", "TFLOPs", "check");

    bool ok = true;
    const int CONC[] = {1, 2, 4, 8, 16, 32};
    // `gemm_rs comm` re-runs just the transfer measurements, which take seconds,
    // without the several minutes of GEMM sweep in front of them.
    const bool comm_only = (argc == 2 && !strcmp(argv[1], "comm"));

    for (int c : CONC) {
        if (comm_only) break;
        long long tok = (long long)c * 2048;
        const int M = (int)(tok < 8192 ? tok : 8192);
        ok &= run_shape("prefill", c, "attn_out", M, H, ATTN);
        ok &= run_shape("prefill", c, "mlp_down", M, H, I);
    }
    for (int c : CONC) {
        if (comm_only) break;
        ok &= run_shape("decode", c, "attn_out", c, H, ATTN);
        ok &= run_shape("decode", c, "mlp_down", c, H, I);
    }

    // The decode rows above are all limitation notices, so measure the part of
    // decode that does exist: the reduce-scatter volume, C*H*2 bytes, halved
    // because each rank only sends the other rank's shard.
    printf("\nThe transfer alone at decode volumes (C/2 x %d x 2 bytes -- each "
           "rank sends\nonly the other's shard). 'blocking' syncs the host after "
           "every transfer;\n'pipelined' enqueues them back to back and syncs "
           "once, which is the situation\n64 decode layers are in. The gap "
           "between the two is host round trip, not wire.\n", H);
    printf("  %9s %17s   %17s %8s\n", "", "blocking", "pipelined", "");
    printf("  %9s %8s %8s   %8s %8s %8s\n", "size", "cpy_ms", "push_ms",
           "cpy_ms", "push_ms", "GB/s");
    for (int c : CONC) comm_floor((size_t)(c / 2 ? c / 2 : 1) * H * sizeof(bf16));
    printf("\nAnd at a prefill volume, for scale:\n");
    comm_floor((size_t)4096 * H * sizeof(bf16));

    return ok ? 0 : 2;
}
