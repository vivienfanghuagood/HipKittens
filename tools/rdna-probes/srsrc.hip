// Does a hand-built V# with config word 0 behave on gfx11: correct in bounds,
// clamped (reads 0, writes dropped) past num_records?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
using i32x4 = int32_t __attribute__((ext_vector_type(4)));

struct buffer_resource { uint64_t ptr; uint32_t range; uint32_t config; };

__device__ uint32_t rbl_b32(i32x4 r, uint32_t v, uint32_t s, uint32_t c)
    __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ void rbs_b32(uint32_t d, i32x4 r, uint32_t v, uint32_t s, uint32_t c)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ inline i32x4 srsrc(const void* p, uint32_t range, uint32_t cfg) {
    buffer_resource r{ (uint64_t)(uintptr_t)p, range, cfg };
    return *reinterpret_cast<const i32x4*>(&r);
}

__global__ void k(const float* in, float* outbuf, uint32_t nbytes,
                  uint32_t cfg, float* got) {
    i32x4 ri = srsrc(in, nbytes, cfg);
    i32x4 ro = srsrc(outbuf, nbytes, cfg);
    uint32_t off = threadIdx.x * 4;
    got[threadIdx.x] = -7.0f;          // control: did the kernel run at all?
    __builtin_amdgcn_s_waitcnt(0);
    uint32_t v = rbl_b32(ri, off, 0, 0);
    got[threadIdx.x] = __builtin_bit_cast(float, v);
    rbs_b32(0x42280000u /* 42.0f */, ro, off, 0, 0);   // must be dropped past nbytes
}

static bool trial(uint32_t cfg) {
    const int N = 8, OVER = 16;
    float h[N]; for (int i = 0; i < N; i++) h[i] = 100.0f + i;
    float *din=nullptr, *dout=nullptr, *dgot=nullptr;
    #define C(x) do{ hipError_t _e=(x); if(_e!=hipSuccess){printf("%s: %s\n", #x, hipGetErrorString(_e)); return false;} }while(0)
    C(hipMalloc(&din, 4096)); C(hipMalloc(&dout, 4096)); C(hipMalloc(&dgot, OVER*4));
    printf("  din=%p dout=%p dgot=%p\n", (void*)din, (void*)dout, (void*)dgot);
    hipMemset(din, 0, 4096); hipMemset(dout, 0, 4096);
    hipMemcpy(din, h, N*4, hipMemcpyHostToDevice);
    for (int i = 0; i < N; i++) h[i] = 100.0f + i;
    hipLaunchKernelGGL(k, dim3(1), dim3(OVER), 0, 0, din, dout, (uint32_t)(N*4), cfg, dgot);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("cfg %08x: launch %s\n", cfg, hipGetErrorString(e)); return false; }
    float g[OVER], o[OVER];
    hipMemcpy(g, dgot, OVER*4, hipMemcpyDeviceToHost);
    hipMemcpy(o, dout, OVER*4, hipMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++)    { ok &= (g[i] == 100.0f + i); ok &= (o[i] == 42.0f); }
    for (int i = N; i < OVER; i++) { ok &= (g[i] == 0.0f);       ok &= (o[i] == 0.0f);  }
    printf("config %08x: %s  loads[", cfg, ok ? "PASS" : "FAIL");
    for (int i = 0; i < OVER; i++) printf("%.0f%s", g[i], i+1<OVER?" ":"");
    printf("]  stores[");
    for (int i = 0; i < OVER; i++) printf("%.0f%s", o[i], i+1<OVER?" ":"");
    printf("]\n");
    hipFree(din); hipFree(dout); hipFree(dgot);
    return ok;
}

int main() {
    bool a = trial(0x00000000u);   // what LLVM emits on gfx11
    bool b = trial(0x00020000u);   // what the CDNA3 global_to_register path uses
    bool c = trial(0x00110000u);   // what the CDNA3 make_srsrc uses
    printf("%s\n", a ? "config 0 is usable on gfx11" : "config 0 FAILED");
    (void)b; (void)c;
    return a ? 0 : 1;
}
