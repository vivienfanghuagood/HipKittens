// symmem_test's three checks, re-run one-process-per-rank over IPC.
//
// The checks are the same and in the same order, because a failure in one makes
// the next meaningless: does translate() name the right memory, does a peer's
// release store get ordered ahead of its payload, and what does the ring cost.
// What is different is the process model, and that is the entire point --
// symmem_test measures the single-process path at 27.0 GB/s and no framework
// can use it. [3] here is the number that decides whether the fused kernel is
// worth integrating, so it is measured before any GEMM is built on top of it.
//
// Run:  ./run_mp.sh 2 ./ipc_heap_test [MB]
#include "ipc_heap.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace hk_dist;

typedef __attribute__((__vector_size__(4 * sizeof(int)))) int int4_t;

__global__ void fill(int *buf, int n, int rank) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < (size_t)n; i += stride) buf[i] = (int)(i * 2654435761u) ^ rank;
}

// Pull from the successor's heap and confirm it holds *their* pattern. A broken
// base table shows up here as reading your own data, which would otherwise look
// like success.
__global__ void check_peer(sym_view v, const int *buf, int n, int peer, int *bad) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const int *remote = v.translate(buf, peer);
    int errs = 0;
    for (; i < (size_t)n; i += stride)
        if (remote[i] != ((int)(i * 2654435761u) ^ peer)) errs++;
    if (errs) atomicAdd(bad, errs);
}

__global__ void push(sym_view v, int4_t *dst, const int4_t *src, int n4, int peer) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    int4_t *remote = v.translate(dst, peer);
    for (; i < (size_t)n4; i += stride) remote[i] = src[i];
}

// Separate launch from `push` so the payload has a grid-wide completion point
// behind it: one thread cannot establish that its whole grid has retired, and a
// release from one thread does not order other threads' stores.
__global__ void raise_flag(sym_view v, uint32_t *flag, uint32_t val, int peer) {
    if (threadIdx.x == 0 && blockIdx.x == 0) v.signal(flag, val, peer);
}

// One block, and a bounded spin. Both are deliberate, for the reasons
// symmem_test.hip records: a device-filling waiter starves the push that has to
// run on the same GPU, and a test that spins forever leaves a wedged kernel on
// a shared node whose only cure is a device reset.
//
// The deadline is in wall_clock64() ticks. clock64() cannot be used: on gfx1100
// it is a 20-bit counter with the high word zeroed, so `clock64() + 20e9` is a
// deadline no reading ever reaches and the spin never exits. See the longer
// note in symmem_test.hip.
//
// EXPECT INTERMITTENT TIMEOUTS HERE -- roughly 1 rank-instance in 6, either
// rank. This is not a regression from the deadline fix, it is what the dead
// deadline was hiding: [2] exercises a *device-resident* flag, and on gfx1100 a
// peer's release store to one is a plain global_store_b32 that can sit in the
// producer's L2 until something evicts it. Delivery is therefore unbounded, so
// no budget makes this test reliable. The shipping path does not use device
// flags (see symmem.cuh's signal/wait warning and host_flags in ipc_heap.cuh);
// this case is kept because it is the cheapest reproducer of the property.
__global__ void wait_flag(const uint32_t *flag, uint32_t val, int *timed_out,
                          long long budget) {
    if (threadIdx.x != 0) return;
    const long long t0 = wall_clock64();
    for (;;) {
        if (__hip_atomic_load(flag, __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_SYSTEM) >= val)
            return;
        __asm__ __volatile__("" ::: "memory");
        if (wall_clock64() - t0 > budget) break;
    }
    *timed_out = 1;
}

__global__ void verify_inbox(const int *inbox, int n, int from, int *bad,
                             const int *timed_out) {
    if (*timed_out) return;   // nothing arrived; an inbox of zeros is not news
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    int errs = 0;
    for (; i < (size_t)n; i += stride)
        if (inbox[i] != ((int)(i * 2654435761u) ^ from)) errs++;
    if (errs) atomicAdd(bad, errs);
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const char *er = getenv("HK_RANK"), *ew = getenv("HK_WORLD");
    if (!er || !ew) { fprintf(stderr, "HK_RANK/HK_WORLD unset; use run_mp.sh\n"); return 1; }
    const int rank = atoi(er), world = atoi(ew);
    const size_t MB = argc > 1 ? atoi(argv[1]) : 64;
    const int n  = (int)(MB << 20) / 4;
    const int n4 = n / 4;

    // Every rank sees every device and differs only in which it drives, which
    // is what torchrun does and what hipIpcOpenMemHandle needs.
    int avail = 0;
    HKD_CHECK(hipGetDeviceCount(&avail));
    if (world > avail) {
        fprintf(stderr, "rank %d: world %d but only %d visible devices\n",
                rank, world, avail);
        return 1;
    }

    file_bootstrap bs(rank, world);
    if (rank == 0) {
        hipDeviceProp_t p; HKD_CHECK(hipGetDeviceProperties(&p, 0));
        printf("=== ipc_heap, %d processes x %s (%s), %zu MB buffers ===\n",
               world, p.name, p.gcnArchName, MB);
    }

    ipc_heap heap((size_t)(MB << 20) * 2 + (1 << 20), rank, bs);
    int *mine       = heap.allocate<int>(n);
    int *inbox      = heap.allocate<int>(n);
    uint32_t *flags = heap.allocate<uint32_t>(64);
    heap.check_symmetry();
    if (rank == 0) printf("  heap ready, allocation sequences agree\n");

    // The peer-mapped base, and how well aligned it is.
    //
    // This is here because [3] below is bimodal run to run on a fixed pair of
    // GPUs -- ~15.5 GB/s or ~25 GB/s, nothing in between -- with identical code
    // and a fixed direction, so whatever differs has to be established at
    // process start. The mapping is the only thing that is. If the fast runs
    // are the ones whose peer base is 2 MB aligned, the transport is being
    // handed large pages sometimes and 4 KB pages others.
    {
        auto align_of = [](uintptr_t p) {
            int k = 0;
            while (k < 31 && !(p & (1u << k))) k++;
            return k;
        };
        const auto &v = heap.view();
        for (int r = 0; r < world; r++)
            printf("  rank %d: heap_base[%d] = 0x%012llx (2^%d aligned)%s\n",
                   rank, r, (unsigned long long)v.heap_bases_[r],
                   align_of(v.heap_bases_[r]), r == rank ? "  <- own" : "");
    }

    const int succ = (rank + 1) % world;
    const int pred = (rank + world - 1) % world;

    // [0] mismatches, [1] the handshake timed out. Different things: a mismatch
    // is a wrong answer, a timeout is no answer.
    int *bad; HKD_CHECK(hipMalloc(&bad, 8)); HKD_CHECK(hipMemset(bad, 0, 8));

    hipLaunchKernelGGL(fill, dim3(512), dim3(256), 0, heap.stream(), mine, n, rank);
    heap.barrier();

    // 1. translate()
    hipLaunchKernelGGL(check_peer, dim3(512), dim3(256), 0, heap.stream(),
                       heap.view(), mine, n, succ, bad);
    heap.barrier();
    int h = 0;
    HKD_CHECK(hipMemcpy(&h, bad, 4, hipMemcpyDeviceToHost));
    HKD_CHECK(hipMemset(bad, 0, 4));
    printf("[1] rank %d: translate to rank %d's heap: %s (%d mismatches)\n",
           rank, succ, h ? "FAIL" : "OK", h);

    // 2. release/acquire across the IPC mapping. The waiter has to be resident
    // before the pushes land, so it goes on its own stream and everyone
    // rendezvouses before any push is enqueued.
    hipStream_t wstream;
    HKD_CHECK(hipStreamCreate(&wstream));
    int wall_khz = 0;
    HKD_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate,
                                    heap.device()));
    hipLaunchKernelGGL(wait_flag, dim3(1), dim3(64), 0, wstream, flags, 1u,
                       bad + 1, (long long)wall_khz * 10000);  // 10 s
    hipLaunchKernelGGL(verify_inbox, dim3(512), dim3(256), 0, wstream,
                       inbox, n, pred, bad, bad + 1);
    heap.barrier();

    hipLaunchKernelGGL(push, dim3(2048), dim3(256), 0, heap.stream(),
                       heap.view(), (int4_t *)inbox, (const int4_t *)mine, n4, succ);
    hipLaunchKernelGGL(raise_flag, dim3(1), dim3(64), 0, heap.stream(),
                       heap.view(), flags, 1u, succ);
    HKD_CHECK(hipStreamSynchronize(heap.stream()));
    HKD_CHECK(hipStreamSynchronize(wstream));
    int hb[2] = {0, 0};
    HKD_CHECK(hipMemcpy(hb, bad, 8, hipMemcpyDeviceToHost));
    printf("[2] rank %d: peer push + release/acquire handoff: %s "
           "(%d mismatches, %d timeouts)\n",
           rank, (hb[0] || hb[1]) ? "FAIL" : "OK", hb[0], hb[1]);
    heap.barrier();
    if (hb[1]) {
        // The flag never arrived. Pushing more at a peer that is not listening
        // measures nothing, so stop rather than report a bandwidth number taken
        // on a fabric that does not deliver.
        fprintf(stderr, "rank %d: handshake timed out; skipping [3]\n", rank);
        HKD_CHECK(hipStreamDestroy(wstream));
        (void)hipFree(bad);
        return 2;
    }

    // 3. What the transport costs, with no waiters, so the number is the wire
    // and not the handshake.
    //
    // Two modes, because conflating them is how the number this whole plan
    // hangs on got measured wrong once already. iris_p2p.hip read 14.5 GB/s for
    // IPC against p2p_bw.hip's 27.0 for hipDeviceEnablePeerAccess and concluded
    // the mapping costs 1.9x -- but that comparison was one rank pushing while
    // the peer sat idle, and symmem_test's 16.8 GB/s per rank is two ranks
    // pushing at once. Those are different experiments. `uni` runs exactly the
    // probe's experiment in this process model so the two are finally
    // comparable; `ring` is the traffic a reduce-scatter actually generates.
    //
    // The host window is printed alongside the event time because the per-rank
    // figures in `ring` only mean anything if the ranks really were pushing
    // simultaneously. The barrier polls files at 2 ms, so a few ms of skew is
    // expected; a window that does not overlap means the "aggregate" below is
    // two unidirectional measurements added together, which is not a thing.
    const bool uni = (argc > 2 && !strcmp(argv[2], "uni"));
    hipEvent_t t0, t1;
    HKD_CHECK(hipEventCreate(&t0)); HKD_CHECK(hipEventCreate(&t1));
    auto round = [&]() {
        hipLaunchKernelGGL(push, dim3(2048), dim3(256), 0, heap.stream(),
                           heap.view(), (int4_t *)inbox, (const int4_t *)mine,
                           n4, succ);
    };
    const bool active = !uni || rank == 0;
    const int reps = argc > 3 ? atoi(argv[3]) : 50;
    if (active) for (int w = 0; w < 3; w++) round();
    heap.barrier();
    // CLOCK_REALTIME and not steady_clock: these are separate processes and the
    // only clock they are guaranteed to share is the wall clock.
    auto now_ms = [] {
        timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        return (double)(ts.tv_sec % 100) * 1e3 + ts.tv_nsec / 1e6;
    };
    const double w0 = now_ms();
    if (active) {
        HKD_CHECK(hipEventRecord(t0, heap.stream()));
        for (int i = 0; i < reps; i++) round();
        HKD_CHECK(hipEventRecord(t1, heap.stream()));
    }
    HKD_CHECK(hipStreamSynchronize(heap.stream()));
    const double w1 = now_ms();
    const double gb = (double)(MB << 20) / 1e9;
    if (active) {
        float ms = 0; HKD_CHECK(hipEventElapsedTime(&ms, t0, t1)); ms /= reps;
        printf("[3] rank %d -> %d: %s push %zu MB x%d: %.3f ms  %.1f GB/s "
               " [host window %.1f .. %.1f ms]\n",
               rank, succ, uni ? "uni " : "ring", MB, reps, ms,
               gb / (ms * 1e-3), w0, w1);
    } else {
        printf("[3] rank %d: idle (uni mode)                         "
               "          [host window %.1f .. %.1f ms]\n", rank, w0, w1);
    }
    heap.barrier();

    HKD_CHECK(hipEventDestroy(t0)); HKD_CHECK(hipEventDestroy(t1));
    HKD_CHECK(hipStreamDestroy(wstream));
    (void)hipFree(bad);
    return (h || hb[0] || hb[1]) ? 1 : 0;
}
