// gfx10+ V# word3 sweep. Suspicion: with OOB_SELECT (bits 29:28) = 0, NumRecords
// is counted in units of STRIDE, and STRIDE is 0 for a raw buffer -- so every
// access is out of bounds and reads 0. OOB_SELECT = 3 makes NumRecords bytes.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buffer_resource { uint64_t ptr; uint32_t range; uint32_t config; };
__device__ uint32_t rbl(i32x4 r, uint32_t v, uint32_t s, uint32_t c)
    __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ void rbs(uint32_t d, i32x4 r, uint32_t v, uint32_t s, uint32_t c)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__global__ void k(const float* in, float* scratch, uint32_t nbytes,
                  uint32_t cfg, float* out) {
    uint32_t off = threadIdx.x * 4;
    buffer_resource bi{ (uint64_t)(uintptr_t)in,      nbytes, cfg };
    buffer_resource bo{ (uint64_t)(uintptr_t)scratch, nbytes, cfg };
    out[threadIdx.x] = __builtin_bit_cast(float,
        rbl(*reinterpret_cast<const i32x4*>(&bi), off, 0, 0));
    rbs(0x42280000u, *reinterpret_cast<const i32x4*>(&bo), off, 0, 0);
}

int main() {
    const int N = 8, T = 16;
    float h[N]; for (int i = 0; i < N; i++) h[i] = 100.0f + i;
    float *din, *dscr, *dout;
    hipMalloc(&din, 4096); hipMalloc(&dscr, 4096); hipMalloc(&dout, T*4);
    uint32_t cfgs[] = {0x31014000u, 0x30014000u, 0x31004000u, 0x31010000u,
                       0x30004000u, 0x30010000u, 0x30000000u, 0x00014000u};
    for (uint32_t cfg : cfgs) {
        hipMemset(din, 0, 4096); hipMemset(dscr, 0, 4096); hipMemset(dout, 0, T*4);
        hipMemcpy(din, h, N*4, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(k, dim3(1), dim3(T), 0, 0, din, dscr, (uint32_t)(N*4), cfg, dout);
        if (hipDeviceSynchronize() != hipSuccess) { printf("%08x: fault\n", cfg); continue; }
        float o[T], s[T];
        hipMemcpy(o, dout, T*4, hipMemcpyDeviceToHost);
        hipMemcpy(s, dscr, T*4, hipMemcpyDeviceToHost);
        bool rd = true, wr = true;
        for (int i = 0; i < N; i++) { rd &= (o[i] == 100.0f + i); wr &= (s[i] == 42.0f); }
        for (int i = N; i < T; i++) { rd &= (o[i] == 0.0f);       wr &= (s[i] == 0.0f);  }
        printf("cfg %08x  oob_sel=%u  read %-4s write %-4s | loads: ",
               cfg, (cfg >> 28) & 3, rd ? "ok" : "BAD", wr ? "ok" : "BAD");
        for (int i = 0; i < T; i++) printf("%.0f ", o[i]);
        printf("\n");
    }
    return 0;
}
