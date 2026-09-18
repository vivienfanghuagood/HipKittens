// Phase 3 gate: does the RDNA3 rt_base layout derivation actually agree with
// what gfx11 WMMA does, once shared_to_register puts real data in the registers?
//
// Two kernels, both single-wave (32 threads):
//   k_single : one 16x16x16 tile, the minimal statement of the layout claim.
//   k_block  : 32x32x64 blocked GEMM, which exercises multi-tile indexing, the
//              vectorized A path, the elementwise B (col-layout) path and the
//              accumulator store path all at once.
// Both compare elementwise against a CPU reference in double.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "kittens.cuh"

using namespace kittens;

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e)); exit(1); } } while(0)

// ---------------------------------------------------------------- 16x16x16

__global__ void k_single(const bf16* __restrict__ Ag,
                         const bf16* __restrict__ Bg,
                         float* __restrict__ Cg) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<16,16> (&As) = al.allocate<st_bf<16,16>>();
    st_bf<16,16> (&Bs) = al.allocate<st_bf<16,16>>();
    st_fl<16,16> (&Cs) = al.allocate<st_fl<16,16>>();

    // Plain scalar staging: global->shared is Phase 4's job, not this test's.
    for (int t = threadIdx.x; t < 256; t += 32) {
        As[{t/16, t%16}] = Ag[t];
        Bs[{t/16, t%16}] = Bg[t];
    }
    __syncthreads();

    rt_bf<16,16,ducks::rt_layout::row> a;
    rt_bf<16,16,ducks::rt_layout::col> b;
    rt_fl<16,16,ducks::rt_layout::col> c;
    load(a, As);
    load(b, Bs);
    zero(c);
    mma_AB(c, a, b, c);
    store(Cs, c);
    __syncthreads();

    for (int t = threadIdx.x; t < 256; t += 32) Cg[t] = Cs[{t/16, t%16}];
}

// ---------------------------------------------------------------- 32x32x64

constexpr int M = 32, N = 32, K = 64;

__global__ void k_block(const bf16* __restrict__ Ag,
                        const bf16* __restrict__ Bg,
                        float* __restrict__ Cg) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<M,K> (&As) = al.allocate<st_bf<M,K>>();
    st_bf<K,N> (&Bs) = al.allocate<st_bf<K,N>>();
    st_fl<M,N> (&Cs) = al.allocate<st_fl<M,N>>();

    for (int t = threadIdx.x; t < M*K; t += 32) As[{t/K, t%K}] = Ag[t];
    for (int t = threadIdx.x; t < K*N; t += 32) Bs[{t/N, t%N}] = Bg[t];
    __syncthreads();

    rt_bf<M,K,ducks::rt_layout::row> a;
    rt_bf<K,N,ducks::rt_layout::col> b;
    rt_fl<M,N,ducks::rt_layout::col> c;
    load(a, As);
    load(b, Bs);
    zero(c);
    mma_AB(c, a, b, c);
    store(Cs, c);
    __syncthreads();

    for (int t = threadIdx.x; t < M*N; t += 32) Cg[t] = Cs[{t/N, t%N}];
}

// ------------------------------------------- the other three mma variants
//
// Same math as k_block, but reaching it through mma_ABt / mma_AtB / mma_AtBt.
// The host feeds each of these the *same* row-major A and B it feeds k_block,
// so each kernel has to read its operand out of shared memory transposed.

__global__ void k_abt(const bf16* __restrict__ Ag, const bf16* __restrict__ Bg,
                      float* __restrict__ Cg) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<M,K> (&As) = al.allocate<st_bf<M,K>>();
    st_bf<N,K> (&Bs) = al.allocate<st_bf<N,K>>();   // B transposed: (N,K)
    st_fl<M,N> (&Cs) = al.allocate<st_fl<M,N>>();

    for (int t = threadIdx.x; t < M*K; t += 32) As[{t/K, t%K}] = Ag[t];
    for (int t = threadIdx.x; t < K*N; t += 32) Bs[{t%N, t/N}] = Bg[t];
    __syncthreads();

    rt_bf<M,K,ducks::rt_layout::row> a;
    rt_bf<N,K,ducks::rt_layout::row> b;
    rt_fl<M,N,ducks::rt_layout::col> c;
    load(a, As); load(b, Bs); zero(c);
    mma_ABt(c, a, b, c);
    store(Cs, c);
    __syncthreads();
    for (int t = threadIdx.x; t < M*N; t += 32) Cg[t] = Cs[{t/N, t%N}];
}

__global__ void k_atb(const bf16* __restrict__ Ag, const bf16* __restrict__ Bg,
                      float* __restrict__ Cg) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<K,M> (&As) = al.allocate<st_bf<K,M>>();   // A transposed: (K,M)
    st_bf<K,N> (&Bs) = al.allocate<st_bf<K,N>>();
    st_fl<M,N> (&Cs) = al.allocate<st_fl<M,N>>();

    for (int t = threadIdx.x; t < M*K; t += 32) As[{t%K, t/K}] = Ag[t];
    for (int t = threadIdx.x; t < K*N; t += 32) Bs[{t/N, t%N}] = Bg[t];
    __syncthreads();

    rt_bf<K,M,ducks::rt_layout::col> a;
    rt_bf<K,N,ducks::rt_layout::col> b;
    rt_fl<M,N,ducks::rt_layout::col> c;
    load(a, As); load(b, Bs); zero(c);
    mma_AtB(c, a, b, c);
    store(Cs, c);
    __syncthreads();
    for (int t = threadIdx.x; t < M*N; t += 32) Cg[t] = Cs[{t/N, t%N}];
}

__global__ void k_atbt(const bf16* __restrict__ Ag, const bf16* __restrict__ Bg,
                       float* __restrict__ Cg) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<K,M> (&As) = al.allocate<st_bf<K,M>>();
    st_bf<N,K> (&Bs) = al.allocate<st_bf<N,K>>();
    st_fl<M,N> (&Cs) = al.allocate<st_fl<M,N>>();

    for (int t = threadIdx.x; t < M*K; t += 32) As[{t%K, t/K}] = Ag[t];
    for (int t = threadIdx.x; t < K*N; t += 32) Bs[{t%N, t/N}] = Bg[t];
    __syncthreads();

    rt_bf<K,M,ducks::rt_layout::col> a;
    rt_bf<N,K,ducks::rt_layout::row> b;
    rt_fl<M,N,ducks::rt_layout::col> c;
    load(a, As); load(b, Bs); zero(c);
    mma_AtBt(c, a, b, c);
    store(Cs, c);
    __syncthreads();
    for (int t = threadIdx.x; t < M*N; t += 32) Cg[t] = Cs[{t/N, t%N}];
}

// -------------------------------------------- accumulator load round trip
//
// k_block only proved the accumulator *store* path. This one stores C, reloads
// it into a register tile of the given layout, and stores it again -- so a bug
// in either direction of the f32 (strided, wave-half-interleaved) mapping shows
// up as a permuted result rather than cancelling out.

template<ducks::rt_layout::all L>
__global__ void k_roundtrip(const bf16* __restrict__ Ag, const bf16* __restrict__ Bg,
                            float* __restrict__ Cg) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<M,K> (&As) = al.allocate<st_bf<M,K>>();
    st_bf<K,N> (&Bs) = al.allocate<st_bf<K,N>>();
    st_fl<M,N> (&Cs) = al.allocate<st_fl<M,N>>();

    for (int t = threadIdx.x; t < M*K; t += 32) As[{t/K, t%K}] = Ag[t];
    for (int t = threadIdx.x; t < K*N; t += 32) Bs[{t/N, t%N}] = Bg[t];
    __syncthreads();

    rt_bf<M,K,ducks::rt_layout::row> a;
    rt_bf<K,N,ducks::rt_layout::col> b;
    rt_fl<M,N,ducks::rt_layout::col> c;
    load(a, As); load(b, Bs); zero(c);
    mma_AB(c, a, b, c);
    store(Cs, c);
    __syncthreads();

    rt_fl<M,N,L> reloaded;
    load(reloaded, Cs);
    __syncthreads();
    zero(c);
    store(Cs, c);          // scrub, so a no-op reload cannot pass by accident
    __syncthreads();
    store(Cs, reloaded);
    __syncthreads();
    for (int t = threadIdx.x; t < M*N; t += 32) Cg[t] = Cs[{t/N, t%N}];
}

// ---------------------------------------------------------------- driver

static float rnd() { return (float)((rand() % 2001) - 1000) / 1000.0f; }

// bf16 has 8 mantissa bits; a K-long dot product of O(1) terms lands around
// K * 2^-8 of relative slop, so scale tolerance with K rather than fixing it.
static bool check(const char* name, const std::vector<float>& got,
                  const std::vector<double>& want, int k) {
    double tol = 0.02 * std::sqrt((double)k);
    double worst = 0; int worst_i = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::fabs(got[i] - want[i]);
        if (d > worst) { worst = d; worst_i = (int)i; }
    }
    bool ok = worst <= tol;
    printf("%-10s %s  max abs err %.5f (tol %.5f) at %d: got %.5f want %.5f\n",
           name, ok ? "PASS" : "FAIL", worst, tol, worst_i,
           got[worst_i], want[worst_i]);
    return ok;
}

template<int m, int n, int k, typename Kern>
static bool run(const char* name, Kern kern, size_t smem) {
    std::vector<float> ha(m*k), hb(k*n);
    for (auto& x : ha) x = rnd();
    for (auto& x : hb) x = rnd();

    std::vector<bf16> ba(m*k), bb(k*n);
    for (int i = 0; i < m*k; i++) ba[i] = __float2bfloat16(ha[i]);
    for (int i = 0; i < k*n; i++) bb[i] = __float2bfloat16(hb[i]);
    // Reference from the *rounded* inputs, so we only measure the accumulation.
    for (int i = 0; i < m*k; i++) ha[i] = __bfloat162float(ba[i]);
    for (int i = 0; i < k*n; i++) hb[i] = __bfloat162float(bb[i]);

    std::vector<double> ref(m*n, 0.0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int p = 0; p < k; p++) s += (double)ha[i*k+p] * (double)hb[p*n+j];
            ref[i*n+j] = s;
        }

    bf16 *da, *db; float *dc;
    CHK(hipMalloc(&da, m*k*sizeof(bf16)));
    CHK(hipMalloc(&db, k*n*sizeof(bf16)));
    CHK(hipMalloc(&dc, m*n*sizeof(float)));
    CHK(hipMemcpy(da, ba.data(), m*k*sizeof(bf16), hipMemcpyHostToDevice));
    CHK(hipMemcpy(db, bb.data(), k*n*sizeof(bf16), hipMemcpyHostToDevice));
    CHK(hipMemset(dc, 0, m*n*sizeof(float)));

    hipLaunchKernelGGL(kern, dim3(1), dim3(32), smem, 0, da, db, dc);
    CHK(hipGetLastError());
    CHK(hipDeviceSynchronize());

    std::vector<float> hc(m*n);
    CHK(hipMemcpy(hc.data(), dc, m*n*sizeof(float), hipMemcpyDeviceToHost));
    CHK(hipFree(da)); CHK(hipFree(db)); CHK(hipFree(dc));
    return check(name, hc, ref, k);
}

int main() {
    srand(1234);
    bool ok = true;
    ok &= run<16,16,16>("16x16x16", k_single, 16*1024);
    ok &= run<M,N,K>   ("mma_AB",   k_block,  32*1024);
    ok &= run<M,N,K>   ("mma_ABt",  k_abt,    32*1024);
    ok &= run<M,N,K>   ("mma_AtB",  k_atb,    32*1024);
    ok &= run<M,N,K>   ("mma_AtBt", k_atbt,   32*1024);
    ok &= run<M,N,K>   ("rt_fl col", k_roundtrip<ducks::rt_layout::col>, 32*1024);
    printf("%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
