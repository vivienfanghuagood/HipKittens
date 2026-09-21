// Does the Iris substrate work on a PCIe-attached RDNA3 box?
//
// Iris (ROCm/iris, branch muhaawd/irisx) is what upstream HipKittens'
// distributed-kernels/ is built on. The header is 391 lines and contains no
// architecture check of any kind -- no gfx test, no CDNA test, no XGMI
// assumption -- so it should compile for gfx1100 unchanged. Compiling is not
// the question. The question is whether the three runtime facts it silently
// depends on actually hold on a W7900D:
//
//   1. hipExtMallocWithFlags(hipDeviceMallocFinegrained) succeeds. Iris's whole
//      symmetric heap is one fine-grained allocation. Coarse-grained memory is
//      only coherent at kernel boundaries, so if the fine-grained flag is
//      silently ignored or refused, every device-side peer load/store Iris
//      offers is unsound and the library is unusable even though it builds.
//   2. hipIpcGetMemHandle / hipIpcOpenMemHandle work across processes on this
//      GPU. Iris exchanges handles over MPI; there is one process per rank.
//      IPC is not guaranteed on workstation parts the way it is on MI300X.
//   3. A release store from a *remote* rank is observed in order by a local
//      reader. Iris's atomic_store(..., release) and atomic_load(..., acquire)
//      are plain __hip_atomic_* with a caller-chosen scope; whether that
//      actually fences a PCIe write stream is a hardware property, not a
//      library one. Every producer/consumer overlap kernel -- write my tile to
//      the owner, raise a flag, let the owner consume it -- rests on this. If
//      it does not hold, the design has to fall back to kernel-boundary
//      synchronization and fine-grained overlap is off the table.
//
// The shape of the test: fork before HIP is initialized (HIP does not survive
// fork the other way round), child takes GPU 1, parent takes GPU 0. Parent owns
// the buffer and hands its IPC handle down a pipe. Child opens it, fills the
// payload with a checkable pattern, then raises the flag with a release store
// at system scope. Parent polls the flag with an acquire load and, the instant
// it sees it, verifies the payload -- without any host synchronization in
// between, which is the only way the ordering claim is actually under test.
//
// A payload mismatch here means the release did not order the payload writes.
// A timeout means the flag write is not visible at all.
//
// Results on the 8x W7900D node, GPU0<->GPU1 and GPU0<->GPU4, 64 MB:
//
//   fine-grained alloc            OK
//   hipIpcGet/OpenMemHandle       OK, both inside and across the PCIe groups
//   payload mismatches            0 of 16777216
//   remote fetch_add              exact
//
// So all three assumptions hold: Iris's substrate is sound on RDNA3, and a
// remote release store really does order the write stream ahead of it. A
// producer/consumer overlap kernel is buildable.
//
// The bandwidth is the surprise. Every cell of the matrix lands on 14.5 GB/s:
//
//   coarse, 16B stores   14.5      fine,   16B stores   14.5
//   coarse,  4B stores   14.6      fine,    4B stores   14.6
//   local->peer copy     15.8      fine, 16B, peer spinning on the flag  14.5
//
// p2p_bw.hip pushes the same link at 27.0 GB/s at the same 64 MB. Everything
// that could explain the 1.8x has been ruled out here: heap kind, store width,
// buffer size (27.0 at both 64 and 256 MB), whether the consumer is spinning,
// whether peer access is enabled eagerly, and whether the kernel reads local
// memory as well as writing peer. The one remaining difference is the mapping
// itself -- p2p_bw has both GPUs in one process and reaches them through
// hipDeviceEnablePeerAccess, while this has one process per rank and reaches
// them through hipIpcOpenMemHandle, which is what Iris and RCCL both do.
//
// That reprices every overlap decision: against 14.5 GB/s rather than 25, the
// break-even is ~5300 FLOP per byte, not 3080. It is a deployment-model cost
// rather than a hardware one -- a single process driving all eight devices
// would keep the 27 -- but it is the cost any MPI-shaped or torch.distributed-
// shaped program pays, including the RCCL baseline being measured against.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("[%s] %s failed: %s\n", who, #x, hipGetErrorString(e)); \
    return 1; } } while (0)

static const int   N_PAY  = 16 * 1024 * 1024;             // ints of payload (64 MB)
static const size_t BYTES = (size_t)(N_PAY + 64) * 4;     // + a cache line for the flag

// The flag lives past the payload so the two are in different cache lines and
// the ordering being tested is real memory ordering, not coincidence.
__device__ __forceinline__ int *flag_of(int *buf) { return buf + N_PAY; }

// Writer: runs on the child's GPU, stores into the parent's memory across PCIe.
// The pattern is index-derived so a partial or reordered write is detectable
// rather than just "wrong".
__global__ void write_then_signal(int *peer, int n, int tag) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < (size_t)n; i += stride) peer[i] = (int)(i * 2654435761u) ^ tag;
}

// Separate launch for the flag: within one kernel the compiler and the memory
// system are both free to reorder across the whole grid, and a grid-wide
// "everyone finished" is not something a single thread can establish. Two
// launches on the same stream give the payload a real completion point, which
// is exactly what a producer warp in a real overlap kernel would have to
// arrange for itself with a barrier.
__global__ void signal_only(int *peer, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        __hip_atomic_store(flag_of(peer), value, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}

// Reader: runs on the parent's GPU against its own memory, which the peer is
// writing. Thread 0 spins on the flag; everyone else waits at the barrier and
// then checks. No host sync anywhere in here.
__global__ void poll_then_verify(int *buf, int n, int tag, int *out) {
    __shared__ int seen;
    if (threadIdx.x == 0) {
        seen = 0;
        long long deadline = clock64() + 20000000000LL;   // ~10 s at 2 GHz
        while (clock64() < deadline) {
            if (__hip_atomic_load(flag_of(buf), __ATOMIC_ACQUIRE,
                                  __HIP_MEMORY_SCOPE_SYSTEM) == 0xABCD) {
                seen = 1; break;
            }
        }
    }
    __syncthreads();
    if (!seen) { if (threadIdx.x == 0) out[1] = 1; return; }   // timed out

    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    int bad = 0;
    for (; i < (size_t)n; i += stride)
        if (buf[i] != ((int)(i * 2654435761u) ^ tag)) bad++;
    if (bad) atomicAdd(&out[0], bad);
}

// The bandwidth matrix. p2p_bw.hip measured 24-25 GB/s pushing coarse-grained
// memory with 16B stores; the first run of this probe got 9.2 GB/s pushing
// Iris's fine-grained heap with 4B stores. Two variables moved at once, and
// which one matters decides whether a fused kernel is free:
//
//   - If fine-grained is the cost, every Iris peer store pays it, because the
//     symmetric heap has to be fine-grained for device-side coherence to mean
//     anything. Fusion would then cost 2.7x bandwidth against SDMA on
//     coarse-grained memory, and the break-even shifts from 3080 FLOP/byte to
//     about 8400 -- which would put every collective on this box firmly in the
//     comm-bound regime and make the fused kernel the wrong design.
//   - If the store width is the cost, it is a code fix and nothing else.
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int int4_t;

// Matching p2p_bw.hip's PUSH exactly: read local, write peer. Kept as a control
// because that kernel reached 27 GB/s on the same link and the only remaining
// differences were the read and the mapping.
__global__ void push_copy(int4_t *__restrict__ peer, const int4_t *__restrict__ local, int n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < (size_t)n4; i += stride) peer[i] = local[i];
}

template <int WIDTH>
__global__ void push_bw(int *peer, int n) {
    if constexpr (WIDTH == 16) {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = (size_t)gridDim.x * blockDim.x;
        int4_t v = {1, 2, 3, 4};
        for (; i < (size_t)n / 4; i += stride) ((int4_t *)peer)[i] = v;
    } else {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = (size_t)gridDim.x * blockDim.x;
        for (; i < (size_t)n; i += stride) peer[i] = 1;
    }
}

// Round-trip a remote atomic. Iris exposes fetch_add and CAS on remote ranks
// and a barrier or a work queue would be built out of them; a device atomic
// that silently operates on a stale local copy would be worse than one that
// fails outright, so count the result rather than trusting the call.
__global__ void remote_fetch_add(int *peer, int reps) {
    for (int r = 0; r < reps; r++)
        __hip_atomic_fetch_add(flag_of(peer) + 1, 1, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_SYSTEM);
}

static int parent_side(int rfd, int wfd, int dev) {
    const char *who = "parent";
    CHK(hipSetDevice(dev));

    void *buf = nullptr;
    hipError_t fg = hipExtMallocWithFlags(&buf, BYTES, hipDeviceMallocFinegrained);
    printf("[parent] fine-grained alloc (%zu MB): %s\n", BYTES >> 20,
           fg == hipSuccess ? "OK" : hipGetErrorString(fg));
    if (fg != hipSuccess) {
        printf("[parent] falling back to hipMalloc -- coarse-grained, peer stores\n"
               "         would only be coherent at kernel boundaries\n");
        CHK(hipMalloc(&buf, BYTES));
    }
    CHK(hipMemset(buf, 0, BYTES));

    // A coarse-grained buffer alongside it, purely as the bandwidth control.
    void *coarse = nullptr;
    CHK(hipMalloc(&coarse, BYTES));
    CHK(hipMemset(coarse, 0, BYTES));

    hipIpcMemHandle_t h, hc;
    hipError_t ipc  = hipIpcGetMemHandle(&h, buf);
    hipError_t ipc2 = hipIpcGetMemHandle(&hc, coarse);
    printf("[parent] hipIpcGetMemHandle: %s\n",
           ipc == hipSuccess ? "OK" : hipGetErrorString(ipc));
    int ok = (ipc == hipSuccess && ipc2 == hipSuccess);
    if (write(wfd, &ok, sizeof ok) != sizeof ok) return 1;
    if (!ok) return 1;
    if (write(wfd, &h, sizeof h) != (ssize_t)sizeof h) return 1;
    if (write(wfd, &hc, sizeof hc) != (ssize_t)sizeof hc) return 1;

    int *out = nullptr;
    CHK(hipMalloc(&out, 8));
    CHK(hipMemset(out, 0, 8));

    // Stay off the link until the child's clean bandwidth matrix is done.
    int ack0 = 0;
    if (read(rfd, &ack0, sizeof ack0) != sizeof ack0) return 1;

    // Launch the poll before telling the child to go, so the reader is already
    // spinning when the writes land.
    hipLaunchKernelGGL(poll_then_verify, dim3(256), dim3(256), 0, 0,
                       (int *)buf, N_PAY, 0x5A5A, out);
    int go = 1;
    if (write(wfd, &go, sizeof go) != sizeof go) return 1;

    CHK(hipDeviceSynchronize());
    int hout[2] = {0, 0};
    CHK(hipMemcpy(hout, out, 8, hipMemcpyDeviceToHost));
    if (hout[1]) {
        printf("[parent] TIMEOUT: the peer's flag never became visible\n");
    } else {
        printf("[parent] flag observed; payload mismatches: %d of %d  -> %s\n",
               hout[0], N_PAY,
               hout[0] == 0 ? "release/acquire ORDERS peer writes across PCIe"
                            : "ORDERING BROKEN -- flag arrived before payload");
    }

    // Let the child run its remote atomics, then read the counter.
    int ack = 0;
    if (read(rfd, &ack, sizeof ack) != sizeof ack) return 1;
    int counter = 0;
    CHK(hipMemcpy(&counter, (int *)buf + N_PAY + 1, 4, hipMemcpyDeviceToHost));
    printf("[parent] remote fetch_add counter: %d (expected %d) -> %s\n",
           counter, 1024 * 64, counter == 1024 * 64 ? "OK" : "WRONG");
    return hout[1] || hout[0];
}

static int child_side(int rfd, int wfd, int dev) {
    const char *who = "child";
    CHK(hipSetDevice(dev));

    int ok = 0;
    if (read(rfd, &ok, sizeof ok) != sizeof ok || !ok) return 1;
    hipIpcMemHandle_t h, hc;
    if (read(rfd, &h, sizeof h) != (ssize_t)sizeof h) return 1;
    if (read(rfd, &hc, sizeof hc) != (ssize_t)sizeof hc) return 1;

    // Explicitly enable peer access before opening the handle. IPC open already
    // asks for it lazily; this checks whether an eager enable maps the pages
    // any differently, which is the one cheap lever on the 1.9x IPC penalty.
    hipError_t pe = hipDeviceEnablePeerAccess(0, 0);
    printf("[child ] explicit hipDeviceEnablePeerAccess(GPU0): %s\n",
           (pe == hipSuccess || pe == hipErrorPeerAccessAlreadyEnabled)
               ? "OK" : hipGetErrorString(pe));

    void *peer = nullptr, *peer_coarse = nullptr;
    hipError_t e  = hipIpcOpenMemHandle(&peer, h, hipIpcMemLazyEnablePeerAccess);
    hipError_t e2 = hipIpcOpenMemHandle(&peer_coarse, hc, hipIpcMemLazyEnablePeerAccess);
    printf("[child ] hipIpcOpenMemHandle on GPU%d: %s\n", dev,
           e == hipSuccess ? "OK" : hipGetErrorString(e));
    if (e != hipSuccess || e2 != hipSuccess) return 1;

    // The matrix runs *before* the go handshake, so nothing is running on the
    // other GPU while it is timed. An earlier version measured it with the
    // parent's poll kernel already spinning and read 14.5 GB/s against
    // p2p_bw.hip's 24-25 for the same link; a consumer spinning on a
    // system-scope atomic load is not free, and conflating that with the
    // transport's own cost would have been a wrong conclusion about Iris.
    {
        const double mb = (double)N_PAY * 4 / (1 << 20);
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        auto bench = [&](const char *label, void *dst, int width) {
            // hipLaunchKernelGGL expands to a do{}while, so it cannot sit in a
            // ternary; the launch has to be a statement.
            auto go1 = [&]() {
                if (width == 16)
                    hipLaunchKernelGGL((push_bw<16>), dim3(2048), dim3(256), 0, 0, (int *)dst, N_PAY);
                else
                    hipLaunchKernelGGL((push_bw<4>),  dim3(2048), dim3(256), 0, 0, (int *)dst, N_PAY);
            };
            for (int warm = 0; warm < 2; warm++) go1();
            (void)hipDeviceSynchronize();
            (void)hipEventRecord(a);
            const int reps = 5;
            for (int r = 0; r < reps; r++) go1();
            (void)hipEventRecord(b); (void)hipEventSynchronize(b);
            float ms; (void)hipEventElapsedTime(&ms, a, b); ms /= reps;
            printf("[child ]   %-22s %7.3f ms  %6.1f GB/s\n", label, ms,
                   mb * (1 << 20) / (ms * 1e6));
        };
        printf("[child ] push %.0f MB, store width x heap kind:\n", mb);
        bench("coarse, 16B stores", peer_coarse, 16);
        bench("coarse,  4B stores", peer_coarse, 4);
        bench("fine,   16B stores", peer, 16);
        bench("fine,    4B stores", peer, 4);

        // local -> peer copy, the exact shape p2p_bw.hip measures.
        void *local = nullptr;
        (void)hipMalloc(&local, (size_t)N_PAY * 4);
        for (int warm = 0; warm < 2; warm++)
            hipLaunchKernelGGL(push_copy, dim3(2048), dim3(256), 0, 0,
                               (int4_t *)peer, (const int4_t *)local, N_PAY / 4);
        (void)hipDeviceSynchronize();
        (void)hipEventRecord(a);
        for (int r = 0; r < 5; r++)
            hipLaunchKernelGGL(push_copy, dim3(2048), dim3(256), 0, 0,
                               (int4_t *)peer, (const int4_t *)local, N_PAY / 4);
        (void)hipEventRecord(b); (void)hipEventSynchronize(b);
        { float ms; (void)hipEventElapsedTime(&ms, a, b); ms /= 5;
          printf("[child ]   %-22s %7.3f ms  %6.1f GB/s\n", "local->peer copy", ms,
                 (double)N_PAY * 4 / (ms * 1e6)); }
        (void)hipFree(local);

        int ack0 = 1;
        if (write(wfd, &ack0, sizeof ack0) != sizeof ack0) return 1;
        int go = 0;
        if (read(rfd, &go, sizeof go) != sizeof go) return 1;
        // Same push, but now the owner is spinning on the flag. This is the
        // number a real overlap kernel lives with, because the consumer has to
        // be waiting for the tile it is about to consume.
        bench("fine,16B, peer spins", peer, 16);
    }

    hipEvent_t s, t; hipEventCreate(&s); hipEventCreate(&t);
    hipEventRecord(s);
    hipLaunchKernelGGL(write_then_signal, dim3(2048), dim3(256), 0, 0,
                       (int *)peer, N_PAY, 0x5A5A);
    hipLaunchKernelGGL(signal_only, dim3(1), dim3(64), 0, 0, (int *)peer, 0xABCD);
    hipEventRecord(t); CHK(hipEventSynchronize(t));
    float ms; hipEventElapsedTime(&ms, s, t);
    printf("[child ] pushed %zu MB + flag in %.3f ms (%.1f GB/s)\n",
           (size_t)N_PAY * 4 >> 20, ms, (double)N_PAY * 4 / (ms * 1e6));

    hipLaunchKernelGGL(remote_fetch_add, dim3(64), dim3(256), 0, 0, (int *)peer, 4);
    CHK(hipDeviceSynchronize());
    int ack = 1;
    if (write(wfd, &ack, sizeof ack) != sizeof ack) return 1;

    hipIpcCloseMemHandle(peer); hipIpcCloseMemHandle(peer_coarse);
    return 0;
}

int main(int argc, char **argv) {
    const int dev_a = argc > 1 ? atoi(argv[1]) : 0;
    const int dev_b = argc > 2 ? atoi(argv[2]) : 1;
    printf("=== Iris substrate probe: GPU%d owns memory, GPU%d writes it ===\n",
           dev_a, dev_b);

    // Unbuffered: the child leaves via _exit, which does not flush stdio, and
    // interleaving two processes' line-buffered output is unreadable anyway.
    setvbuf(stdout, nullptr, _IONBF, 0);

    int p2c[2], c2p[2];
    if (pipe(p2c) || pipe(c2p)) { perror("pipe"); return 1; }

    // fork first: a process that has already touched HIP cannot be forked into
    // a usable second HIP context.
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        close(p2c[1]); close(c2p[0]);
        int r = child_side(p2c[0], c2p[1], dev_b);
        _exit(r);
    }
    close(p2c[0]); close(c2p[1]);
    int r = parent_side(c2p[0], p2c[1], dev_a);
    int st = 0; waitpid(pid, &st, 0);
    return r || WEXITSTATUS(st);
}
