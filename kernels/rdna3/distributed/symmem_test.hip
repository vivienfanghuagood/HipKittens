// Smoke test and bandwidth floor for symmem.cuh.
//
// Three things get checked, in the order that a failure in one would make the
// next meaningless:
//
//   1. translate() names the right memory. Every rank writes a pattern derived
//      from its own id into its own heap, then reads its neighbour's heap
//      through translate() and checks it sees the neighbour's pattern, not its
//      own. A broken heap-base table shows up here as reading your own data,
//      which is the failure mode that would otherwise look like success.
//   2. signal()/wait() order a peer's writes. Rank r pushes a payload into
//      rank r+1's heap and then raises a flag there; rank r+1 spins on the flag
//      and verifies the payload the instant it appears, with no host
//      synchronization anywhere in between. iris_p2p.hip established that this
//      holds across PCIe on gfx1100 for one pair of GPUs in two processes; this
//      re-establishes it for eight GPUs in one process, which is the
//      configuration the kernels will actually use.
//   3. What the ring costs. All N ranks push to their successor at once, which
//      is the traffic shape of a ring reduce-scatter or all-gather, and is the
//      number a fused kernel has to beat by overlapping rather than by being
//      faster on the wire.
#include "symmem.cuh"
#include <cstdio>
#include <vector>

using namespace hk_dist;

typedef __attribute__((__vector_size__(4 * sizeof(int)))) int int4_t;

__global__ void fill(int *buf, int n, int rank) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < (size_t)n; i += stride) buf[i] = (int)(i * 2654435761u) ^ rank;
}

// Read the neighbour's heap through the view and confirm it holds *their*
// pattern. Pure pull traffic, no synchronization: the fill kernels have all
// completed on the host before this runs.
__global__ void check_peer(sym_view v, const int *buf, int n, int peer, int *bad) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const int *remote = v.translate(buf, peer);
    int errs = 0;
    for (; i < (size_t)n; i += stride)
        if (remote[i] != ((int)(i * 2654435761u) ^ peer)) errs++;
    if (errs) atomicAdd(bad, errs);
}

// Push into the successor's heap, then raise its flag. The flag is a separate
// launch on the same stream so the payload has a real grid-wide completion
// point behind it -- a single thread cannot establish that its whole grid has
// retired, and a release from one thread does not order other threads' stores.
__global__ void push(sym_view v, int4_t *dst, const int4_t *src, int n4, int peer) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    int4_t *remote = v.translate(dst, peer);
    for (; i < (size_t)n4; i += stride) remote[i] = src[i];
}

__global__ void raise_flag(sym_view v, uint32_t *flag, uint32_t val, int peer) {
    if (threadIdx.x == 0 && blockIdx.x == 0) v.signal(flag, val, peer);
}

// Wait on my own flag. One block, deliberately.
//
// The obvious way to write this test is one kernel that spins and then
// verifies, sized to cover the buffer. That deadlocks, and the reason is worth
// keeping: the waiter and the push to the *successor* run on the same GPU, so
// a waiter wide enough to fill the device leaves no room for the push to be
// scheduled. Rank 0 then waits on a push from rank 1 that rank 1's own waiter
// is blocking, all the way round the ring. Splitting wait from verify keeps
// the property under test -- a consumer resident and spinning while the
// producer writes, with no host synchronization between them -- while leaving
// the other 95 CUs free.
//
// The spin is bounded, unlike sym_view::wait. A real overlap kernel spins
// indefinitely because the alternative is worse, but a *test* that spins
// forever on a handshake that never completes leaves a kernel wedged on a GPU
// other people are sharing, and the way out of that is a device reset. One
// block also means one deadline: the first version put a deadline in each of
// 512 blocks, which do not run concurrently, so the timeouts ran in series and
// the "10 second" bound was closer to a minute.
//
// clock64() was verified to advance on gfx1100 before being trusted here --
// s_memtime and s_memrealtime are both gone in gfx11, so a counter reading as
// a constant would have made this a watchdog that silently wasn't one.
__global__ void wait_flag(const uint32_t *flag, uint32_t val, int *timed_out) {
    if (threadIdx.x != 0) return;
    const long long deadline = clock64() + 20000000000LL;  // ~10s at 2GHz
    while (clock64() < deadline) {
        if (__hip_atomic_load(flag, __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_SYSTEM) >= val)
            return;
    }
    *timed_out = 1;
}

// Verify what the predecessor left in my inbox. Stream-ordered behind
// wait_flag, so the acquire that made the payload visible has already happened.
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
    // Unbuffered, because this program can be killed by an external timeout and
    // a full stdio buffer dies with it. Losing the trace of where it got to is
    // exactly the information a timeout is being used to obtain.
    setvbuf(stdout, nullptr, _IONBF, 0);

    int avail = 0;
    HKD_CHECK(hipGetDeviceCount(&avail));
    const int N = argc > 1 ? atoi(argv[1]) : avail;
    const size_t MB = argc > 2 ? atoi(argv[2]) : 64;
    const int n  = (int)(MB << 20) / 4;        // ints per buffer
    const int n4 = n / 4;

    hipDeviceProp_t p; HKD_CHECK(hipGetDeviceProperties(&p, 0));
    printf("=== symmem on %d x %s (%s) ===\n", N, p.name, p.gcnArchName);

    // Three buffers plus a flag page per rank, allocated in the same order on
    // every rank so the offsets stay symmetric.
    printf("  allocating symmetric heap ...\n");
    sym_heap heap((size_t)(MB << 20) * 2 + (1 << 20), N);
    auto mine  = heap.allocate_all<int>(n);        // my pattern
    auto inbox = heap.allocate_all<int>(n);        // what my predecessor sends
    auto flags = heap.allocate_all<uint32_t>(64);
    printf("  heap ready\n");

    // Two counters per rank: [0] mismatches, [1] the handshake timed out. They
    // mean different things -- a mismatch is a wrong answer, a timeout is no
    // answer -- and conflating them would report a hang as a data error.
    std::vector<int *> bad(N);
    for (int r = 0; r < N; r++) {
        HKD_CHECK(hipSetDevice(r));
        HKD_CHECK(hipMalloc(&bad[r], 8));
        HKD_CHECK(hipMemset(bad[r], 0, 8));
        hipLaunchKernelGGL(fill, dim3(512), dim3(256), 0, heap.stream(r),
                           mine[r], n, r);
    }
    heap.sync_all();
    printf("  fill done\n");

    // 1. translate()
    for (int r = 0; r < N; r++) {
        HKD_CHECK(hipSetDevice(r));
        hipLaunchKernelGGL(check_peer, dim3(512), dim3(256), 0, heap.stream(r),
                           heap.view(r), mine[r], n, (r + 1) % N, bad[r]);
    }
    heap.sync_all();
    int total = 0;
    for (int r = 0; r < N; r++) {
        int h = 0; HKD_CHECK(hipSetDevice(r));
        HKD_CHECK(hipMemcpy(&h, bad[r], 4, hipMemcpyDeviceToHost));
        total += h;
        HKD_CHECK(hipMemset(bad[r], 0, 4));
    }
    printf("[1] translate to neighbour heap: %s (%d mismatches)\n",
           total ? "FAIL" : "OK", total);

    // 2 + 3. Everyone pushes to their successor and flags it, everyone waits on
    // their predecessor. Timed, because this is also the ring measurement.
    hipEvent_t t0, t1;
    HKD_CHECK(hipSetDevice(0));
    HKD_CHECK(hipEventCreate(&t0)); HKD_CHECK(hipEventCreate(&t1));

    auto round = [&](uint32_t tag) {
        for (int r = 0; r < N; r++) {
            HKD_CHECK(hipSetDevice(r));
            const int succ = (r + 1) % N;
            hipLaunchKernelGGL(push, dim3(2048), dim3(256), 0, heap.stream(r),
                               heap.view(r), (int4_t *)inbox[r],
                               (const int4_t *)mine[r], n4, succ);
            hipLaunchKernelGGL(raise_flag, dim3(1), dim3(64), 0, heap.stream(r),
                               heap.view(r), flags[r], tag, succ);
        }
    };

    // Correctness round: the waiters must be resident before the pushes land,
    // so they are launched first on their own streams.
    printf("  launching waiters ...\n");
    std::vector<hipStream_t> wstream(N);
    for (int r = 0; r < N; r++) {
        HKD_CHECK(hipSetDevice(r));
        HKD_CHECK(hipStreamCreate(&wstream[r]));
        const int pred = (r + N - 1) % N;
        hipLaunchKernelGGL(wait_flag, dim3(1), dim3(64), 0, wstream[r],
                           flags[r], 1u, bad[r] + 1);
        hipLaunchKernelGGL(verify_inbox, dim3(512), dim3(256), 0, wstream[r],
                           inbox[r], n, pred, bad[r], bad[r] + 1);
    }
    printf("  pushing ...\n");
    round(1u);
    heap.sync_all();
    for (int r = 0; r < N; r++) {
        HKD_CHECK(hipSetDevice(r)); HKD_CHECK(hipStreamSynchronize(wstream[r]));
    }
    total = 0;
    int hangs = 0;
    for (int r = 0; r < N; r++) {
        int h[2] = {0, 0}; HKD_CHECK(hipSetDevice(r));
        HKD_CHECK(hipMemcpy(h, bad[r], 8, hipMemcpyDeviceToHost));
        total += h[0];
        hangs += h[1];
    }
    printf("[2] peer push + release/acquire handoff: %s (%d mismatches, %d timeouts)\n",
           (total || hangs) ? "FAIL" : "OK", total, hangs);
    if (hangs) {
        // The flag never arrived. Pushing more traffic at a peer that is not
        // listening tells us nothing, so stop rather than run the bandwidth
        // round on a broken fabric.
        fprintf(stderr, "handshake timed out on %d rank(s); skipping [3]\n", hangs);
        for (int r = 0; r < N; r++) {
            HKD_CHECK(hipSetDevice(r));
            (void)hipFree(bad[r]); (void)hipStreamDestroy(wstream[r]);
        }
        return 2;
    }

    // Bandwidth round: pushes only, no waiters, so the number is the transport
    // and not the handshake.
    const int reps = 10;
    for (int w = 0; w < 2; w++) round(2u);
    heap.sync_all();
    HKD_CHECK(hipSetDevice(0));
    HKD_CHECK(hipEventRecord(t0, heap.stream(0)));
    for (int i = 0; i < reps; i++) round(3u);
    HKD_CHECK(hipSetDevice(0));
    HKD_CHECK(hipEventRecord(t1, heap.stream(0)));
    heap.sync_all();
    float ms = 0; HKD_CHECK(hipEventElapsedTime(&ms, t0, t1)); ms /= reps;
    const double gb = (double)(MB << 20) / 1e9;
    printf("[3] ring push %zu MB x %d: %.3f ms  %.1f GB/s per GPU  %.1f GB/s aggregate\n",
           MB, N, ms, gb / (ms * 1e-3), N * gb / (ms * 1e-3));

    for (int r = 0; r < N; r++) {
        HKD_CHECK(hipSetDevice(r)); (void)hipFree(bad[r]); (void)hipStreamDestroy(wstream[r]);
    }
    return total != 0;
}
