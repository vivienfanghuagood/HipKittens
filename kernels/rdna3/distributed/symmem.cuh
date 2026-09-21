// Symmetric device memory across the GPUs of one node, without MPI.
//
// Upstream HipKittens' distributed-kernels/ reaches peer memory through
// ROCm/iris: a fine-grained heap per rank, IPC handles exchanged over MPI, and
// a device-side view that turns a local pointer into a remote one by pointer
// arithmetic against a table of heap bases. Everything in that design that
// touches the GPU is architecture-neutral, and tools/rdna-probes/iris_p2p.hip
// confirms all of it works on gfx1100: fine-grained allocation, IPC across
// processes, remote atomics, and -- the one that actually gates a fused kernel
// -- a remote release store observed in order by a local acquire poll.
//
// This header keeps that device-side shape exactly and replaces only the
// bootstrap, for two reasons, one forced and one earned:
//
//   * Forced: the node this runs on is air-gapped and has no mpirun, no
//     pybind11 and no cmake. Iris's MPI use is five calls, all bootstrap --
//     init, finalize, an allgather of handles, a barrier, a comm split -- and
//     none of them are on the device. Replacing them costs less than making
//     MPI appear.
//   * Earned: iris_p2p.hip measured the IPC-mapped path at 14.5 GB/s against
//     27.0 GB/s for the same link, same size and same kernel reached through
//     hipDeviceEnablePeerAccess in a single process. Heap kind, store width,
//     transfer size, a spinning consumer and eager peer-enable were all ruled
//     out; the mapping is what differs. One process driving eight devices is
//     worth 1.9x here, and for kernel work there is no reason to pay it.
//
// The cost of the single-process model is that it is not a drop-in for a
// torch.distributed or MPI program -- it is a benchmark harness, not a runtime.
// Since the device-side names match Iris's, a kernel written against this
// compiles against iris::iris_device_view with a typedef change.
//
// Two places where this header deliberately does not match Iris, both found
// while checking that iris.hpp compiles for gfx1100 (it does):
//
//   * Iris declares every device-side atomic as
//         template <typename T, memory_scope scope = memory_scope_thread>
//     and memory_scope_thread is __HIP_MEMORY_SCOPE_SINGLETHREAD. A release
//     store written against those defaults orders nothing beyond the issuing
//     thread, so a cross-GPU handshake built on them is not synchronized at
//     all -- and, being a race, it will pass every small test. Overriding it
//     means spelling the value type too, since scope is the second template
//     parameter. Here the scope is not a parameter: it is SYSTEM, always.
//   * Iris keeps translate() private and exposes only get_heap_base(rank), so
//     a kernel cannot form a remote pointer and must go through the per-element
//     store()/load() wrappers -- which is why upstream's epilogue writes peer
//     memory one element at a time instead of with 16B vector stores. translate()
//     is public here so a fused epilogue can push wide.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace hk_dist {

static constexpr int MAX_RANKS = 8;

#define HKD_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            hipGetErrorString(_e)); abort(); } } while (0)

// ---------------------------------------------------------------------------
// Device view. One per rank; passed by value into kernels.
// ---------------------------------------------------------------------------
struct sym_view {
    int cur_rank_;
    int world_size_;
    uintptr_t heap_bases_[MAX_RANKS];

    __host__ __device__ int cur_rank()   const { return cur_rank_; }
    __host__ __device__ int world_size() const { return world_size_; }

    // The whole trick: every rank's heap is the same size and allocations are
    // made in the same order on every rank, so an offset into one heap is the
    // same offset into all of them. A local pointer therefore names a remote
    // object without any lookup.
    template <typename T>
    __host__ __device__ __forceinline__ T *translate(const T *p, int rank) const {
        return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(p)
                                     - heap_bases_[cur_rank_] + heap_bases_[rank]);
    }

    template <typename T>
    __device__ __forceinline__ void store(T *p, T v, int rank) {
        *translate(p, rank) = v;
    }
    template <typename T>
    __device__ __forceinline__ T load(const T *p, int rank) const {
        return *translate(p, rank);
    }

    // Scope matters and the default is deliberately the expensive one. A store
    // that has to be seen by another *device* is AGENT-scope at best from the
    // compiler's point of view; crossing to another GPU needs SYSTEM. Getting
    // this wrong does not fail loudly -- it produces a kernel that passes on
    // small inputs and corrupts on large ones -- so these wrappers exist mostly
    // to keep the scope from being forgotten at a call site.
    template <typename T>
    __device__ __forceinline__ T atomic_load(const T *p, int rank,
                                             int order = __ATOMIC_RELAXED) const {
        return __hip_atomic_load(translate(p, rank), order, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    template <typename T>
    __device__ __forceinline__ void atomic_store(T *p, T v, int rank,
                                                 int order = __ATOMIC_RELAXED) {
        __hip_atomic_store(translate(p, rank), v, order, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    template <typename T>
    __device__ __forceinline__ T atomic_fetch_add(T *p, T v, int rank,
                                                  int order = __ATOMIC_RELAXED) {
        return __hip_atomic_fetch_add(translate(p, rank), v, order,
                                      __HIP_MEMORY_SCOPE_SYSTEM);
    }

    // Producer half of a tile handoff: publish everything written so far, then
    // raise the flag. The release is what makes the payload visible first, and
    // iris_p2p.hip verified that this holds across PCIe on gfx1100 -- 0
    // mismatches out of 16.7M words with no host synchronization in between.
    __device__ __forceinline__ void signal(uint32_t *flag, uint32_t v, int rank) {
        __hip_atomic_store(translate(flag, rank), v, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
    // Consumer half. Spins on local memory that a peer is writing; the acquire
    // pairs with the producer's release. Measured to cost the peer nothing:
    // pushing 64 MB while the owner spins on the flag runs at the same 14.5
    // GB/s as pushing into an idle GPU.
    __device__ __forceinline__ void wait(const uint32_t *flag, uint32_t v) const {
        while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < v) { }
    }
};

// ---------------------------------------------------------------------------
// Host side: one process, N devices, a symmetric bump heap on each.
// ---------------------------------------------------------------------------
class sym_heap {
public:
    // Fine-grained is not optional. Coarse-grained device memory is only
    // coherent at kernel boundaries, so a peer store landing mid-kernel may
    // never become visible to a consumer spinning in another kernel -- which is
    // exactly the pattern every overlap kernel is built on. iris_p2p.hip
    // measured fine- and coarse-grained at the same bandwidth, so there is
    // nothing to trade away by always asking for fine.
    sym_heap(size_t bytes_per_rank, int world_size)
        : bytes_(bytes_per_rank), world_(world_size) {
        if (world_ > MAX_RANKS) { fprintf(stderr, "world > %d\n", MAX_RANKS); abort(); }
        bases_.resize(world_);
        used_ = 0;

        for (int r = 0; r < world_; r++) {
            HKD_CHECK(hipSetDevice(r));
            void *p = nullptr;
            HKD_CHECK(hipExtMallocWithFlags(&p, bytes_, hipDeviceMallocFinegrained));
            HKD_CHECK(hipMemset(p, 0, bytes_));
            bases_[r] = reinterpret_cast<uintptr_t>(p);
        }
        // Peer access is per (device, peer) and one-directional, so this is
        // N*(N-1) enables, not N*(N-1)/2.
        for (int r = 0; r < world_; r++) {
            HKD_CHECK(hipSetDevice(r));
            for (int q = 0; q < world_; q++) {
                if (q == r) continue;
                hipError_t e = hipDeviceEnablePeerAccess(q, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "peer %d->%d: %s\n", r, q, hipGetErrorString(e));
                    abort();
                }
            }
        }
        streams_.resize(world_);
        for (int r = 0; r < world_; r++) {
            HKD_CHECK(hipSetDevice(r));
            HKD_CHECK(hipStreamCreate(&streams_[r]));
        }
    }

    ~sym_heap() {
        for (int r = 0; r < world_; r++) {
            (void)hipSetDevice(r);
            (void)hipStreamDestroy(streams_[r]);
            (void)hipFree(reinterpret_cast<void *>(bases_[r]));
        }
    }

    // Allocate the same object on every rank and hand back the per-rank
    // pointers. This is the only allocation entry point on purpose: the bump
    // pointer must advance exactly once per logical object, and a caller that
    // allocated on a subset of ranks, or out of rank order, would desync the
    // heaps silently -- translate() would then still produce a valid-looking
    // pointer into the wrong object.
    template <typename T>
    std::vector<T *> allocate_all(size_t n) {
        std::vector<T *> out(world_);
        for (int r = 0; r < world_; r++) out[r] = allocate_one<T>(n, r);
        return out;
    }

    sym_view view(int rank) const {
        sym_view v{};
        v.cur_rank_ = rank;
        v.world_size_ = world_;
        for (int r = 0; r < world_; r++) v.heap_bases_[r] = bases_[r];
        return v;
    }

    hipStream_t stream(int rank) const { return streams_[rank]; }
    int world_size() const { return world_; }

    void sync_all() {
        for (int r = 0; r < world_; r++) {
            HKD_CHECK(hipSetDevice(r));
            HKD_CHECK(hipStreamSynchronize(streams_[r]));
        }
    }

private:
    // Bump allocation, identical on every rank by construction -- that
    // identical ordering is what makes translate() a subtraction instead of a
    // lookup.
    template <typename T>
    T *allocate_one(size_t n, int rank) {
        const size_t want = (n * sizeof(T) + 255) & ~size_t(255);
        if (used_ + want > bytes_) {
            fprintf(stderr, "symmetric heap exhausted: %zu + %zu > %zu\n",
                    used_, want, bytes_);
            abort();
        }
        T *p = reinterpret_cast<T *>(bases_[rank] + used_);
        if (rank == world_ - 1) used_ += want;   // advance once per round
        return p;
    }

    size_t bytes_, used_;
    int world_;
    std::vector<uintptr_t> bases_;
    std::vector<hipStream_t> streams_;
};

}  // namespace hk_dist
