// Fused GEMM -> ReduceScatter on RDNA3, two ranks.
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
// number is exactly K_local. A K_local of 4096 is an ordinary shape; 17000 is
// not. At 8 ranks this node is fabric-bound however the kernel is written.
//
// What is measured:
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
// Running this: two idle GPUs, pinned. HIP_VISIBLE_DEVICES=0,3 ./gemm_rs, under
// an external timeout. The node is shared.
#define HK_GEMM_NO_PYBIND 1
#include "../gemm/bf16fp32/gemm.cpp"

#include "symmem.cuh"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace hk_dist;

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

int main(int argc, char **argv) {
    // Unbuffered: this is meant to be run under an external timeout, and a full
    // stdio buffer dies with the process.
    setvbuf(stdout, nullptr, _IONBF, 0);

    const int M = argc > 1 ? atoi(argv[1]) : 8192;
    const int N = argc > 2 ? atoi(argv[2]) : 8192;
    const int K = argc > 3 ? atoi(argv[3]) : 8192;
    const int ITERS = argc > 4 ? atoi(argv[4]) : 10;
    const int world = 2;

    const int K_local = K / world;
    const int shard   = M / world;

    // The epilogue decides ownership one warp tile at a time, so a shard
    // boundary falling inside a tile would silently split one. Requiring a
    // multiple of BLOCK_M is stricter than the REG_BLOCK_M the epilogue
    // actually needs, and unlike it does not depend on which config runs.
    if (M % (world * big_config::BLOCK_M) || N % big_config::BLOCK_N ||
        K_local % big_config::K_STEP) {
        fprintf(stderr, "need M %% %d == 0, N %% %d == 0, (K/%d) %% %d == 0\n",
                world * big_config::BLOCK_M, big_config::BLOCK_N, world,
                big_config::K_STEP);
        return 1;
    }

    int avail = 0; CHK(hipGetDeviceCount(&avail));
    if (avail < world) { fprintf(stderr, "need %d GPUs, see %d\n", world, avail); return 1; }
    hipDeviceProp_t prop; CHK(hipGetDeviceProperties(&prop, 0));
    printf("=== fused GEMM->ReduceScatter, %d x %s (%s) ===\n",
           world, prop.name, prop.gcnArchName);
    printf("M=%d N=%d K=%d   K_local=%d   shard=%d rows\n", M, N, K, K_local, shard);

    const size_t a_elems = (size_t)M * K_local;
    const size_t b_elems = (size_t)N * K_local;
    const size_t c_elems = (size_t)M * N;          // full local output, baseline
    const size_t s_elems = (size_t)shard * N;      // one shard

    // One symmetric heap holds everything a peer has to be able to name. Only
    // the inbox strictly has to be symmetric, but allocating in the same order
    // on every rank is what makes translate() a subtraction, so it all goes in.
    const size_t per_rank =
        (a_elems + b_elems + c_elems + 2 * s_elems) * sizeof(bf16) + (16u << 20);
    printf("symmetric heap: %.2f GB per rank\n", per_rank / 1073741824.0);
    sym_heap heap(per_rank, world);

    auto A     = heap.allocate_all<bf16>(a_elems);
    auto B     = heap.allocate_all<bf16>(b_elems);
    auto Cfull = heap.allocate_all<bf16>(c_elems);   // the baseline's local M x N
    auto Cout  = heap.allocate_all<bf16>(s_elems);   // my shard of the answer
    auto Inbox = heap.allocate_all<bf16>(s_elems);   // the peer's partial of it

    for (int r = 0; r < world; r++) {
        CHK(hipSetDevice(r));
        hipLaunchKernelGGL(fill_a, dim3(1024), dim3(256), 0, heap.stream(r),
                           A[r], M, K_local, r * K_local);
        hipLaunchKernelGGL(fill_b, dim3(1024), dim3(256), 0, heap.stream(r),
                           B[r], N, K_local, r * K_local);
    }
    heap.sync_all();
    printf("operands built\n");

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
    printf("reference built\n");

    // ------------------------------------------------------------------- fused
    // c_peer is built once, on the host: the peer's Inbox address translated
    // into this rank's address space. Nothing on the device does any address
    // arithmetic -- the epilogue's peer store is an ordinary vectorized global
    // store that happens to land on another GPU.
    std::vector<micro_globals_rs> gr;
    for (int r = 0; r < world; r++) {
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
        for (int r = 0; r < world; r++) { CHK(hipSetDevice(r)); launch<big_config>(gr[r]); }
        heap.sync_all();
        for (int r = 0; r < world; r++) {
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
    for (int r = 0; r < world; r++)
        gs.push_back(micro_globals{_gl_A(A[r], 1, 1, M, K_local),
                                   _gl_B(B[r], 1, 1, N, K_local),
                                   _gl_C(Cfull[r], 1, 1, M, N), heap.stream(r)});

    auto run_sequential = [&]() {
        for (int r = 0; r < world; r++) { CHK(hipSetDevice(r)); launch<big_config>(gs[r]); }
        for (int r = 0; r < world; r++) {
            const int peer = 1 - r;
            CHK(hipSetDevice(r));
            // My copy of the peer's shard starts at row peer*shard.
            CHK(hipMemcpyPeerAsync(Inbox[peer], peer,
                                   Cfull[r] + (size_t)peer * shard * N, r,
                                   s_elems * sizeof(bf16), heap.stream(r)));
        }
        heap.sync_all();
        for (int r = 0; r < world; r++) {
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
    auto verify = [&](const char *what, std::vector<bf16 *> got) {
        double worst = 0.0, worst_abs = 0.0, worst_scale = 0.0;
        bool degenerate = false;
        for (int r = 0; r < world; r++) {
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
            if (!(fh[1] > 1.f)) { degenerate = true; continue; }
            if (fh[0] > worst_abs)   worst_abs = fh[0];
            if (fh[1] > worst_scale) worst_scale = fh[1];
            if ((double)fh[0] / fh[1] > worst) worst = (double)fh[0] / fh[1];
        }
        // Where the bound comes from. bf16 has 8 significand bits, so a single
        // rounding is up to 2^-8 = 3.9e-3 relative. The reference rounds once;
        // the distributed path rounds each of the two partials and then their
        // sum, and where the K-halves cancel the partials are larger than the
        // answer. Three roundings puts the expectation near 1.2e-2 of full
        // scale, and both paths measure 1.08e-2 at 2048x2048x4096 -- the same
        // number for fused and sequential, which is itself the evidence that
        // this is arithmetic and not routing.
        //
        // 5e-2 is therefore ~4x headroom over the arithmetic while staying 20x
        // below what any routing bug costs: a misrouted tile delivers some
        // other rank's partial, which is wrong by order 100% of scale, not by
        // parts per hundred.
        if (degenerate) {
            printf("%-11s reference has no scale -- the comparison is vacuous   FAIL\n",
                   what);
            return false;
        }
        const bool ok = worst < 5e-2;
        // The scale is printed, not just the ratio: a ratio on its own cannot
        // be checked against the bf16 argument above, and it was a bad ratio
        // that hid the overflow in the first version of this reduction.
        printf("%-11s max|err| %.4g / max|ref| %.4g = %.2e   %s\n", what,
               worst_abs, worst_scale, worst, ok ? "OK" : "FAIL");
        return ok;
    };

    run_fused();
    bool ok = verify("fused", Cout);

    run_sequential();
    std::vector<bf16 *> seq_out(world);
    for (int r = 0; r < world; r++) seq_out[r] = Cfull[r] + (size_t)r * shard * N;
    ok = verify("sequential", seq_out) && ok;

    if (!ok) { fprintf(stderr, "numerics are wrong; not reporting timings\n"); return 2; }

    // -------------------------------------------------------------------- time
    // Host wall clock, not hip events: the work spans two devices and several
    // streams and every iteration ends in a host sync anyway, so an event on
    // one device's stream would be timing a subset of it.
    auto time_it = [&](const char *what, auto &&fn) {
        for (int w = 0; w < 3; w++) fn();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; i++) fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        // FLOPs for the whole problem: the two ranks together do one M x N x K
        // GEMM, so this is comparable to the single-GPU numbers in
        // ../gemm/bf16fp32/README.md doubled.
        printf("%-11s %8.3f ms   %6.1f TFLOPs (both ranks)\n", what, ms,
               2.0 * M * N * K / (ms * 1e-3) / 1e12);
        return ms;
    };

    const double ms_seq = time_it("sequential", run_sequential);
    const double ms_fus = time_it("fused", run_fused);
    printf("fused speedup %.2fx\n", ms_seq / ms_fus);

    // How much there was to win: what the baseline's bulk copy costs on its own.
    printf("(peer copy alone: %.1f MB per rank, %.2f ms at the ~17 GB/s this "
           "fabric measures)\n",
           s_elems * sizeof(bf16) / 1048576.0,
           s_elems * sizeof(bf16) / 17e9 * 1e3);

    CHK(hipSetDevice(0));
    (void)hipFree(rA); (void)hipFree(rB); (void)hipFree(rC);
    return 0;
}
