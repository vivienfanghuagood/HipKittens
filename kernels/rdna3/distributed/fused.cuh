#pragma once
//
// The device-side core of the fused GEMM->collective path: the globals structs
// the epilogue in gemm.cpp selects on, the peer access helpers, the combine
// kernels, and the cross-rank barrier.
//
// This exists so the benchmarks and the torch extension run the *same* code.
// Before it, gemm_rs_mp.hip and gemm_ar_mp.hip each carried their own copy, and
// a third copy in torch_ext/ would have meant the thing shipped to a framework
// was not the thing the sweeps verified. The benchmark-only pieces -- the
// deterministic generators, the error reductions, the diagnostics -- stay in
// their files; they are not part of what runs in production.
//
// Include *after* gemm.cpp (for _gl_A/_gl_B/_gl_C and bf16) and after
// ipc_heap.cuh (for sym_view).

#include <hip/hip_runtime.h>

namespace hk_dist {

// ---------------------------------------------------------------------------
// Globals
//
// Neither struct mentions a peer count. v.translate(p, owner) is correct for
// any world, which is the whole reason the epilogue takes a sym_view rather
// than a gl: kittens::gl has no default constructor and a __host__-only
// constructor, so an array of peer gls was never possible.
//
// Both flags exist on both structs so the shared epilogue can select on them
// with `if constexpr` and no trait.
// ---------------------------------------------------------------------------

// Reduce-scatter: sharded along M, because the framework shards tokens.
// c is my shard of the output; v and c_sym say where everyone else's shard of
// my partial goes.
struct micro_globals_rs {
    _gl_A a;
    _gl_B b;
    _gl_C c;        // my shard (shard_rows, N), local
    sym_view v;
    bf16 *c_sym;    // my slot of the inbox; translate() re-points it at an owner
    int shard_rows;
    int my_rank;
    hipStream_t stream;

    static constexpr bool FUSED_RS = true;
    static constexpr bool FUSED_AR = false;
};

// All-reduce: sharded along N.
//
// Which axis to shard is ours to choose here -- unlike RS, nothing downstream
// sees the intermediate -- and N is the better one. Ownership then falls out of
// col_tile, the only constraint is on N, and M is free, so decode (M=1) works,
// which row sharding cannot do at all.
//
// c is the final output, full M x N, and it is symmetric because the gather
// writes into peers' copies of it. c_own is my column shard of my own partial,
// which the combine turns into the finished shard. c_sym is my slot of the
// inbox; translated into an owner's address space it becomes that owner's slot
// for me, so no sender needs to know which slot it is writing.
//
// ws is the split-K scratch. Leave it null: launch() fills it from its own
// process-static allocator, which is the right owner because whether there is a
// split at all is the dispatcher's decision, not the caller's.
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

// ---------------------------------------------------------------------------
// Peer access
// ---------------------------------------------------------------------------

// The inbox loads are nontemporal, and that is a correctness requirement, not a
// hint. The inbox lives in my HBM but is written by a peer over PCIe, and on
// gfx1100 that write does not invalidate my L2 -- so the zeros I memset into
// the inbox myself, still sitting in my L2, are what an ordinary load returns,
// and the peer's partial contributes nothing. It cost rel 0.6 on rank 1 while
// every after-the-fact comparison of the same bytes read rel 0, because by then
// the lines had aged out. __builtin_nontemporal_load emits the glc/slc/dlc
// bypass bits; nothing else available here does. No locality is lost -- each
// inbox element is read exactly once.
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

// ---------------------------------------------------------------------------
// Combines
//
// Both accumulate in fp32. The partials are bf16 because that is what the GEMM
// epilogue emits, and summing world-1 of them in bf16 would throw away most of
// what little mantissa there is; both passes are entirely memory-bound, so the
// wider accumulator is free. Slot `me` is skipped rather than zeroed -- my own
// tiles never went through the inbox.
// ---------------------------------------------------------------------------

// Reduce-scatter's: add the peers' slots into my shard, in place. out is
// (shard_rows, N) = n elements; inbox is [world][n].
__global__ void combine_n(bf16 *out, const bf16 *inbox, size_t n,
                          int world, int me) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float acc = __bfloat162float(out[i]);
        for (int r = 0; r < world; r++)
            if (r != me) acc += inbox_load(inbox + (size_t)r * n + i);
        out[i] = __float2bfloat16(acc);
    }
}

// All-reduce's: finish my column shard and hand it to everyone.
//
// own is (M, ns): my partial of my own columns. inbox is [world][M][ns]. out is
// (M, N), symmetric, and v translates it into any rank's address space.
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

// ---------------------------------------------------------------------------
// An N-way device-side barrier, on the stream, with a deadline.
//
// A host barrier would be correct and useless: the file bootstrap costs
// milliseconds, the same order as the GEMMs being overlapped, so a barrier
// between them has to be a device one.
//
// The flags are in *host* memory. Device flags are unreliable on gfx1100 -- a
// peer's write into my HBM does not invalidate my L2 and no shader instruction
// can flush it -- and the deadline uses wall_clock64() rather than clock64(),
// because clock64() there is 20 bits wide and wraps in about 0.5 ms, so every
// timeout built on it fires at random.
// ---------------------------------------------------------------------------

#ifndef SPIN_TICK
#define SPIN_TICK __asm__ __volatile__("" ::: "memory")
#endif

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

}  // namespace hk_dist
