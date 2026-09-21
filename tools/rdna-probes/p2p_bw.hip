// What does GPU-to-GPU actually cost on a PCIe-attached RDNA3 box?
//
// This is the foundation number for any compute/communication overlap kernel.
// On MI300X the answer is XGMI and everyone quotes it; on a W7900 node there is
// no XGMI at all -- rocm-smi reports PCIE for all 28 pairs -- so the design
// space has to be re-derived from measurement.
//
// Three transports, because they are not the same number and a fused kernel can
// only use the last one:
//
//   SDMA    hipMemcpyPeerAsync. The DMA engine moves it, the compute units are
//           free. This is what RCCL uses. It cannot be interleaved *inside* a
//           kernel, only overlapped with one on another stream.
//   PUSH    a kernel on the source GPU storing into the peer's memory. Requires
//           large-BAR (checked below). This is the transport a fused
//           GEMM+ReduceScatter needs: the epilogue writes its tile straight to
//           the owner. Writes are posted, so this should behave well.
//   PULL    a kernel on the destination GPU loading from the peer's memory.
//           Same wire, but each load is a round trip, so the expectation going
//           in was that this would be much worse than PUSH.
//
// Pairs are chosen from the topology: 0->1 sits inside one PCIe/NUMA group
// (rocm-smi link weight 40), 0->4 crosses to the other (weight 72).
//
// Measured on the 8x W7900D node, 256 MB:
//
//   0<->1   SDMA 24.0   PUSH 24.0   PULL 23.8   LOCAL 337.8 GB/s
//   0<->4   SDMA 26.7   PUSH 25.7   PULL 25.5   LOCAL 345.0 GB/s
//   ring    all 8 concurrent: 25.7 GB/s per GPU, 205.8 GB/s aggregate
//
// Three things to take from that, and one of them was a wrong guess:
//
// 1. All three transports are the same number. PULL does *not* collapse, so
//    the expectation above was wrong -- the link is bandwidth-bound, not
//    round-trip-bound, and a fused kernel may read peer memory as freely as it
//    writes it. That matters: an AllGather->GEMM can have its tile loader read
//    straight off the peer rather than staging locally first.
// 2. A kernel push costs the same as the DMA engine, so moving the transfer
//    inside a kernel costs nothing in bandwidth. What it buys is granularity:
//    SDMA can only start once a whole buffer is ready.
// 3. The ring does not degrade. Per-GPU bandwidth is the same with 8 peers
//    running as with one pair, so the fabric is genuinely 8-way and there is
//    no switch uplink to design around.
//
// The ratio that sizes every overlap decision: 77 TFLOP/s of bf16 against
// 25 GB/s off-chip is **3080 FLOP per byte communicated**. A GEMM moves 2 bytes
// per output element and spends 2K FLOP on it, so it breaks even at K ~ 3080
// for one pass over the output -- about 5400 for an all-reduce at n=8, about
// 2700 for a one-way reduce-scatter or all-gather. Below that the fused kernel
// is comm-bound no matter how good it is.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("  %s failed: %s\n", #x, hipGetErrorString(e)); return; } } while (0)

typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;

// Grid-stride 16B copy. Whichever side is peer memory is set by the caller.
__global__ void copy16(float4_t *__restrict__ dst, const float4_t *__restrict__ src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

static double time_ms(void (*body)(void *), void *arg, int reps) {
    hipEvent_t s, e;
    hipEventCreate(&s); hipEventCreate(&e);
    body(arg); hipDeviceSynchronize();          // warm
    hipEventRecord(s);
    for (int r = 0; r < reps; r++) body(arg);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms / reps;
}

struct Args { void *dst, *src; size_t bytes; int srcDev, dstDev; };
static void do_sdma(void *a) {
    Args *x = (Args *)a;
    (void)hipMemcpyPeerAsync(x->dst, x->dstDev, x->src, x->srcDev, x->bytes, 0);
}
static void do_kernel(void *a) {
    Args *x = (Args *)a;
    const size_t n = x->bytes / sizeof(float4_t);
    hipLaunchKernelGGL(copy16, dim3(2048), dim3(256), 0, 0,
                       (float4_t *)x->dst, (const float4_t *)x->src, n);
}

static void pair(int A, int B, size_t bytes) {
    printf("\n--- GPU%d <-> GPU%d, %zu MB ---\n", A, B, bytes >> 20);
    void *bufA = nullptr, *bufB = nullptr;
    CHK(hipSetDevice(A)); CHK(hipMalloc(&bufA, bytes));
    CHK(hipSetDevice(B)); CHK(hipMalloc(&bufB, bytes));

    // Peer access is per (device, peer) and one-directional; enable both ways.
    CHK(hipSetDevice(A));
    hipError_t ea = hipDeviceEnablePeerAccess(B, 0);
    CHK(hipSetDevice(B));
    hipError_t eb = hipDeviceEnablePeerAccess(A, 0);
    if ((ea != hipSuccess && ea != hipErrorPeerAccessAlreadyEnabled) ||
        (eb != hipSuccess && eb != hipErrorPeerAccessAlreadyEnabled)) {
        printf("  peer access refused: %s / %s\n",
               hipGetErrorString(ea), hipGetErrorString(eb));
        return;
    }

    const int reps = 20;
    const double gb = (double)bytes / 1e9;
    Args a;

    // SDMA: engine moves A -> B.
    CHK(hipSetDevice(A));
    a = {bufB, bufA, bytes, A, B};
    double ms = time_ms(do_sdma, &a, reps);
    printf("  SDMA  memcpyPeer   %8.3f ms  %7.1f GB/s\n", ms, gb / (ms * 1e-3));

    // PUSH: kernel on A reads local, writes peer.
    CHK(hipSetDevice(A));
    a = {bufB, bufA, bytes, A, B};
    ms = time_ms(do_kernel, &a, reps);
    printf("  PUSH  kernel store %8.3f ms  %7.1f GB/s\n", ms, gb / (ms * 1e-3));

    // PULL: kernel on A reads peer, writes local.
    CHK(hipSetDevice(A));
    a = {bufA, bufB, bytes, B, A};
    ms = time_ms(do_kernel, &a, reps);
    printf("  PULL  kernel load  %8.3f ms  %7.1f GB/s\n", ms, gb / (ms * 1e-3));

    // Local HBM copy on A, as the denominator: what the same kernel does when
    // nothing crosses the bus.
    void *bufA2 = nullptr;
    CHK(hipSetDevice(A)); CHK(hipMalloc(&bufA2, bytes));
    a = {bufA2, bufA, bytes, A, A};
    ms = time_ms(do_kernel, &a, reps);
    printf("  LOCAL kernel copy  %8.3f ms  %7.1f GB/s   (read+write, so x2 on the wire)\n",
           ms, gb / (ms * 1e-3));

    hipFree(bufA); hipFree(bufA2);
    CHK(hipSetDevice(B)); hipFree(bufB);
}

// Every GPU pushes to its ring successor at the same time. A pairwise number
// says nothing about whether eight of them fit: if the PCIe switch uplink is
// the real limit, per-GPU bandwidth collapses here and a ring algorithm is the
// wrong shape. If it holds, the fabric is genuinely 8-way and the bisection is
// n * pairwise.
static void ring(int n, size_t bytes) {
    printf("\n--- all %d GPUs push to (i+1)%%%d concurrently, %zu MB each ---\n",
           n, n, bytes >> 20);
    std::vector<void *> buf(n, nullptr), dst(n, nullptr);
    std::vector<hipStream_t> st(n);
    for (int i = 0; i < n; i++) {
        CHK(hipSetDevice(i));
        CHK(hipMalloc(&buf[i], bytes));
        CHK(hipStreamCreate(&st[i]));
        for (int j = 0; j < n; j++)
            if (j != i) { hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                          if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                              printf("  peer %d->%d refused\n", i, j); return; } }
    }
    for (int i = 0; i < n; i++) dst[i] = buf[(i + 1) % n];

    const size_t elems = bytes / sizeof(float4_t);
    const int reps = 20;
    auto launch_all = [&]() {
        for (int i = 0; i < n; i++) {
            hipSetDevice(i);
            hipLaunchKernelGGL(copy16, dim3(2048), dim3(256), 0, st[i],
                               (float4_t *)dst[i], (const float4_t *)buf[i], elems);
        }
    };
    launch_all();
    for (int i = 0; i < n; i++) { hipSetDevice(i); hipStreamSynchronize(st[i]); }

    hipEvent_t s, e;
    hipSetDevice(0); hipEventCreate(&s); hipEventCreate(&e);
    hipDeviceSynchronize();
    hipEventRecord(s, st[0]);
    for (int r = 0; r < reps; r++) launch_all();
    hipSetDevice(0); hipEventRecord(e, st[0]);
    for (int i = 0; i < n; i++) { hipSetDevice(i); hipStreamSynchronize(st[i]); }
    float ms; hipSetDevice(0); hipEventElapsedTime(&ms, s, e); ms /= reps;
    const double gb = (double)bytes / 1e9;
    printf("  %8.3f ms   %7.1f GB/s per GPU   %7.1f GB/s aggregate\n",
           ms, gb / (ms * 1e-3), n * gb / (ms * 1e-3));

    for (int i = 0; i < n; i++) { hipSetDevice(i); hipFree(buf[i]); hipStreamDestroy(st[i]); }
}

int main(int argc, char **argv) {
    int n = 0; hipGetDeviceCount(&n);
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("%d x %s (%s), isLargeBar=%d\n", n, p.name, p.gcnArchName, p.isLargeBar);
    const size_t bytes = (argc > 1 ? atoi(argv[1]) : 256) * (size_t)(1 << 20);
    pair(0, 1, bytes);                 // same PCIe/NUMA group (link weight 40)
    if (n > 4) pair(0, 4, bytes);      // across groups (link weight 72)
    if (n > 1) ring(n, bytes);
    return 0;
}
