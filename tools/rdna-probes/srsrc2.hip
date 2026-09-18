// Narrow it down: LLVM's own make_buffer_rsrc path vs a hand-built V#,
// against a plain pointer load as control.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buffer_resource { uint64_t ptr; uint32_t range; uint32_t config; };
__device__ uint32_t rbl_legacy(i32x4 r, uint32_t v, uint32_t s, uint32_t c)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__global__ void k(const float* in, uint32_t nbytes, uint32_t cfg, float* out) {
    uint32_t off = threadIdx.x * 4;
    out[threadIdx.x + 0]  = in[threadIdx.x];                       // control
    auto rs = __builtin_amdgcn_make_buffer_rsrc((void*)in, 0, nbytes, 0);
    out[threadIdx.x + 32] = __builtin_amdgcn_raw_buffer_load_b32(rs, off, 0, 0);
    buffer_resource br{ (uint64_t)(uintptr_t)in, nbytes, cfg };
    out[threadIdx.x + 64] = __builtin_bit_cast(float,
        rbl_legacy(*reinterpret_cast<const i32x4*>(&br), off, 0, 0));
}

int main() {
    const int N = 8, T = 16;
    float h[N]; for (int i = 0; i < N; i++) h[i] = 100.0f + i;
    float *din, *dout;
    if (hipMalloc(&din, 4096) || hipMalloc(&dout, 96*4)) { printf("malloc\n"); return 1; }
    hipMemset(din, 0, 4096); hipMemset(dout, 0xFF, 96*4);
    hipMemcpy(din, h, N*4, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k, dim3(1), dim3(T), 0, 0, din, (uint32_t)(N*4), 0u, dout);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("sync: %s\n", hipGetErrorString(e)); return 1; }
    float o[96]; hipMemcpy(o, dout, 96*4, hipMemcpyDeviceToHost);
    const char* names[3] = {"plain ptr   ", "make_rsrc   ", "hand V# cfg0"};
    for (int b = 0; b < 3; b++) {
        printf("%s: ", names[b]);
        for (int i = 0; i < T; i++) printf("%.0f ", o[b*32 + i]);
        printf("\n");
    }
    return 0;
}
