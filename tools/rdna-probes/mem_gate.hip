// Phase 4 gate: every memory path, round-tripped.
//
// Phase 3 proved the rt_base layout against WMMA itself. This proves that the
// *movers* agree with that layout, by loading a known pattern through each path
// and writing it back out. If a lane mapping is wrong the round trip either
// drops elements (they stay at the sentinel) or duplicates them, and either way
// the comparison catches it.
//
// Covered:
//   T1  global <-> register tile, all four (dtype, layout) shapes
//   T2  the same with a row stride that is not 16-byte aligned, which forces
//       global_to_register's elementwise fallback
//   T3  global -> shared -> register -> shared -> global
//   T4  global <-> register vector, all three rv layouts x {bf16, float}
//   T5  shared <-> register vector
//   T6  the 4-warp group shared <-> register tile path
//
// Everything is an identity round trip, so the reference is the input.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "kittens.cuh"

using namespace kittens;

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e)); exit(1); } } while(0)

static int failures = 0;

// bf16 can hold every integer up to 256 exactly, so an identity round trip of
// small integers is exact and any tolerance would only hide a bug.
template<typename T> static double to_d(T x);
template<> double to_d<float>(float x) { return (double)x; }
template<> double to_d<bf16>(bf16 x)   { return (double)__bfloat162float(x); }

template<typename T>
static void check(const char* name, const std::vector<T>& got, const std::vector<double>& want) {
    double worst = 0; int worst_i = -1;
    for (size_t i = 0; i < want.size(); i++) {
        double d = fabs(to_d(got[i]) - want[i]);
        if (d > worst) { worst = d; worst_i = (int)i; }
    }
    if (worst > 1e-6) {
        printf("%-22s FAIL  at %d: got %.6f want %.6f\n",
               name, worst_i, to_d(got[worst_i]), want[worst_i]);
        failures++;
    } else {
        printf("%-22s PASS\n", name);
    }
}

// ------------------------------------------------------- T1/T2 global <-> rt

constexpr int R = 32, C = 32;

template<typename T, typename L>
__global__ void k_g2r(const gl<T,1,1,-1,-1> in, const gl<T,1,1,-1,-1> out) {
    rt<T, R, C, L> t;
    load(t, in, {0,0,0,0});
    store(out, t, {0,0,0,0});
}

template<typename T, typename L>
static void t_g2r(const char* name, int cols) {
    // `cols` may exceed C: the extra columns make the row stride odd so that
    // rows after the first are not 16-byte aligned.
    std::vector<double> want(R*C);
    std::vector<T> host(R*cols);
    for (int r = 0; r < R; r++) for (int c = 0; c < cols; c++) {
        double v = (r*7 + c*3) % 200 - 100;
        host[r*cols+c] = base_types::convertor<T,float>::convert((float)v);
        if (c < C) want[r*C+c] = v;
    }
    T *din, *dout;
    CHK(hipMalloc(&din,  R*cols*sizeof(T)));
    CHK(hipMalloc(&dout, R*cols*sizeof(T)));
    CHK(hipMemcpy(din, host.data(), R*cols*sizeof(T), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, R*cols*sizeof(T)));

    gl<T,1,1,-1,-1> gin (din,  nullptr, nullptr, R, cols);
    gl<T,1,1,-1,-1> gout(dout, nullptr, nullptr, R, cols);
    hipLaunchKernelGGL((k_g2r<T,L>), dim3(1), dim3(32), 0, 0, gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<T> res(R*cols);
    CHK(hipMemcpy(res.data(), dout, R*cols*sizeof(T), hipMemcpyDeviceToHost));
    std::vector<T> tile(R*C);
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) tile[r*C+c] = res[r*cols+c];
    check(name, tile, want);
    CHK(hipFree(din)); CHK(hipFree(dout));
}

// ------------------------------------------- T3 global -> shared -> rt -> ...

__global__ void k_g2s2r(const gl<bf16,1,1,-1,-1> in, const gl<bf16,1,1,-1,-1> out) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<R,C> (&s) = al.allocate<st_bf<R,C>>();

    load(s, in, {0,0,0,0});
    __syncthreads();
    rt_bf<R,C,ducks::rt_layout::row> t;
    load(t, s);
    zero(s);            // scrub, so a dropped element cannot pass by accident
    __syncthreads();
    store(s, t);
    __syncthreads();
    store(out, s, {0,0,0,0});
}

static void t_g2s2r() {
    std::vector<double> want(R*C);
    std::vector<bf16> host(R*C);
    for (int i = 0; i < R*C; i++) {
        double v = (i*11) % 200 - 100;
        host[i] = base_types::convertor<bf16,float>::convert((float)v);
        want[i] = v;
    }
    bf16 *din, *dout;
    CHK(hipMalloc(&din,  R*C*sizeof(bf16)));
    CHK(hipMalloc(&dout, R*C*sizeof(bf16)));
    CHK(hipMemcpy(din, host.data(), R*C*sizeof(bf16), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, R*C*sizeof(bf16)));

    gl<bf16,1,1,-1,-1> gin (din,  nullptr, nullptr, R, C);
    gl<bf16,1,1,-1,-1> gout(dout, nullptr, nullptr, R, C);
    hipLaunchKernelGGL(k_g2s2r, dim3(1), dim3(32), 65536, 0, gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<bf16> res(R*C);
    CHK(hipMemcpy(res.data(), dout, R*C*sizeof(bf16), hipMemcpyDeviceToHost));
    check("g2s -> s2r -> g", res, want);
    CHK(hipFree(din)); CHK(hipFree(dout));
}

// ---------------------------------------------------------- T4/T5 vectors

constexpr int VL = 64;

template<typename T, typename RVL>
__global__ void k_g2rv(const gl<T,1,1,1,-1> in, const gl<T,1,1,1,-1> out) {
    rv<T, VL, RVL> v;
    load(v, in, {0,0,0,0});
    store(out, v, {0,0,0,0});
}

// Through shared as well, which is the path reductions actually use.
template<typename T, typename RVL>
__global__ void k_s2rv(const gl<T,1,1,1,-1> in, const gl<T,1,1,1,-1> out) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    sv<T,VL> (&s) = al.allocate<sv<T,VL>>();

    load(s, in, {0,0,0,0});
    __syncthreads();
    rv<T, VL, RVL> v;
    load(v, s);
    zero(s);
    __syncthreads();
    store(s, v);
    __syncthreads();
    store(out, s, {0,0,0,0});
}

template<typename T, typename RVL, bool via_shared>
static void t_vec(const char* name) {
    std::vector<double> want(VL);
    std::vector<T> host(VL);
    for (int i = 0; i < VL; i++) {
        double v = (i*5) % 200 - 100;
        host[i] = base_types::convertor<T,float>::convert((float)v);
        want[i] = v;
    }
    T *din, *dout;
    CHK(hipMalloc(&din,  VL*sizeof(T)));
    CHK(hipMalloc(&dout, VL*sizeof(T)));
    CHK(hipMemcpy(din, host.data(), VL*sizeof(T), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, VL*sizeof(T)));

    gl<T,1,1,1,-1> gin (din,  nullptr, nullptr, nullptr, VL);
    gl<T,1,1,1,-1> gout(dout, nullptr, nullptr, nullptr, VL);
    if constexpr (via_shared)
        hipLaunchKernelGGL((k_s2rv<T,RVL>), dim3(1), dim3(32), 65536, 0, gin, gout);
    else
        hipLaunchKernelGGL((k_g2rv<T,RVL>), dim3(1), dim3(32), 0, 0, gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<T> res(VL);
    CHK(hipMemcpy(res.data(), dout, VL*sizeof(T), hipMemcpyDeviceToHost));
    check(name, res, want);
    CHK(hipFree(din)); CHK(hipFree(dout));
}

// ------------------------------------------------------------- T6 group

constexpr int GR = 64, GC = 32;   // 4 subtile rows per warp across 4 warps
using G = kittens::group<4>;

__global__ void k_group(const gl<bf16,1,1,-1,-1> in, const gl<bf16,1,1,-1,-1> out) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<GR,GC> (&s) = al.allocate<st_bf<GR,GC>>();

    G::load(s, in, {0,0,0,0});
    __syncthreads();
    rt_bf<GR/4, GC, ducks::rt_layout::row> t;   // each warp owns a quarter
    G::load(t, s);
    zero(s);
    __syncthreads();
    G::store(s, t);
    __syncthreads();
    G::store(out, s, {0,0,0,0});
}

static void t_group() {
    std::vector<double> want(GR*GC);
    std::vector<bf16> host(GR*GC);
    for (int i = 0; i < GR*GC; i++) {
        double v = (i*13) % 200 - 100;
        host[i] = base_types::convertor<bf16,float>::convert((float)v);
        want[i] = v;
    }
    bf16 *din, *dout;
    CHK(hipMalloc(&din,  GR*GC*sizeof(bf16)));
    CHK(hipMalloc(&dout, GR*GC*sizeof(bf16)));
    CHK(hipMemcpy(din, host.data(), GR*GC*sizeof(bf16), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, GR*GC*sizeof(bf16)));

    gl<bf16,1,1,-1,-1> gin (din,  nullptr, nullptr, GR, GC);
    gl<bf16,1,1,-1,-1> gout(dout, nullptr, nullptr, GR, GC);
    hipLaunchKernelGGL(k_group, dim3(1), dim3(128), 65536, 0, gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<bf16> res(GR*GC);
    CHK(hipMemcpy(res.data(), dout, GR*GC*sizeof(bf16), hipMemcpyDeviceToHost));
    check("group s2r roundtrip", res, want);
    CHK(hipFree(din)); CHK(hipFree(dout));
}

// ------------------------------------------- T7 register staging (buffer ops)

// The only path that uses buffer loads, hence the only test that exercises the
// gfx11 V# descriptor. If the config word regresses to a gfx9 one this reads
// all zeros rather than faulting, so the round trip is the detector.
__global__ void k_stage(const gl<bf16,1,1,-1,-1> in, const gl<bf16,1,1,-1,-1> out) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<R,C> (&s) = al.allocate<st_bf<R,C>>();

    constexpr int BUF = stage_calls<st_bf<R,C>>;
    float4 buf[BUF];
    load_global_to_register_buffer<2, false, WARP_THREADS>(buf, BUF, in, {0,0,0,0}, s);
    store_register_buffer_to_shared<WARP_THREADS>(s, buf, BUF);
    __syncthreads();
    store(out, s, {0,0,0,0});
}

static void t_stage() {
    std::vector<double> want(R*C);
    std::vector<bf16> host(R*C);
    for (int i = 0; i < R*C; i++) {
        double v = (i*17) % 200 - 100;
        host[i] = base_types::convertor<bf16,float>::convert((float)v);
        want[i] = v;
    }
    bf16 *din, *dout;
    CHK(hipMalloc(&din,  R*C*sizeof(bf16)));
    CHK(hipMalloc(&dout, R*C*sizeof(bf16)));
    CHK(hipMemcpy(din, host.data(), R*C*sizeof(bf16), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, R*C*sizeof(bf16)));

    gl<bf16,1,1,-1,-1> gin (din,  nullptr, nullptr, R, C);
    gl<bf16,1,1,-1,-1> gout(dout, nullptr, nullptr, R, C);
    hipLaunchKernelGGL(k_stage, dim3(1), dim3(32), 65536, 0, gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<bf16> res(R*C);
    CHK(hipMemcpy(res.data(), dout, R*C*sizeof(bf16), hipMemcpyDeviceToHost));
    check("staged g2s (buffer)", res, want);
    CHK(hipFree(din)); CHK(hipFree(dout));
}

// ------------------------------------------------------------------ main

int main() {
    using row = ducks::rt_layout::row;
    using col = ducks::rt_layout::col;

    t_g2r<bf16,  row>("g2r bf16 row",      C);
    t_g2r<bf16,  col>("g2r bf16 col",      C);
    t_g2r<float, row>("g2r fl   row",      C);
    t_g2r<float, col>("g2r fl   col",      C);
    t_g2r<bf16,  row>("g2r bf16 row (odd)", C + 8);  // stride 40 elems = 80 B

    t_g2s2r();

    t_vec<float, naive_l, false>("g2rv fl   naive");
    t_vec<float, ortho_l, false>("g2rv fl   ortho");
    t_vec<float, align_l, false>("g2rv fl   align");
    t_vec<bf16,  naive_l, false>("g2rv bf16 naive");
    t_vec<bf16,  ortho_l, false>("g2rv bf16 ortho");
    t_vec<bf16,  align_l, false>("g2rv bf16 align");

    t_vec<float, naive_l, true>("s2rv fl   naive");
    t_vec<float, ortho_l, true>("s2rv fl   ortho");
    t_vec<float, align_l, true>("s2rv fl   align");
    t_vec<bf16,  align_l, true>("s2rv bf16 align");

    t_stage();
    t_group();

    printf(failures ? "\n%d FAILED\n" : "\nALL PASS\n", failures);
    return failures != 0;
}
