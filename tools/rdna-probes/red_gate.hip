// Phase 5 gate, part 1: tile reductions.
//
// Covers both reduction shapes against both replication classes:
//
//              row_reduce          col_reduce
//   rt_fl row  element + halfswap  lane butterfly
//   rt_fl col  lane butterfly      element + halfswap
//   rt_bf row  element (mirrored)  lane butterfly
//   rt_bf col  lane butterfly      element (mirrored)
//
// The f32 cases are the ones that need the cross-half exchange, since an
// accumulator's two wave halves hold alternating positions along the element
// axis rather than mirroring. If that exchange is missing, a row_reduce over a
// 32-wide tile returns the reduction of half the columns -- which a max test
// would often still pass by luck, so sums are tested too.

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

template<typename T> static double to_d(T x);
template<> double to_d<float>(float x) { return (double)x; }
template<> double to_d<bf16>(bf16 x)   { return (double)__bfloat162float(x); }

constexpr int R = 32, C = 32;

enum Which { ROW_SUM, ROW_MAX, COL_SUM, COL_MAX };

template<typename T, typename L, Which W>
__global__ void k_red(const gl<T,1,1,-1,-1> in, const gl<T,1,1,1,-1> out) {
    rt<T, R, C, L> t;
    load(t, in, {0,0,0,0});
    if constexpr (W == ROW_SUM || W == ROW_MAX) {
        typename rt<T,R,C,L>::col_vec v;
        if constexpr (W == ROW_SUM) row_sum(v, t); else row_max(v, t);
        store(out, v, {0,0,0,0});
    } else {
        typename rt<T,R,C,L>::row_vec v;
        if constexpr (W == COL_SUM) col_sum(v, t); else col_max(v, t);
        store(out, v, {0,0,0,0});
    }
}

template<typename T, typename L, Which W>
static void t_red(const char* name) {
    const bool by_row = (W == ROW_SUM || W == ROW_MAX);
    const bool is_sum = (W == ROW_SUM || W == COL_SUM);
    const int  OUT    = by_row ? R : C;

    std::vector<T> host(R*C);
    std::vector<double> a(R*C);
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) {
        // Small integers: exact in bf16, and a 32-term sum still is.
        double v = ((r*5 + c*3) % 7) - 3;
        a[r*C+c] = v;
        host[r*C+c] = base_types::convertor<T,float>::convert((float)v);
    }
    std::vector<double> want(OUT);
    for (int o = 0; o < OUT; o++) {
        double acc = is_sum ? 0.0 : -1e30;
        for (int k = 0; k < (by_row ? C : R); k++) {
            double v = by_row ? a[o*C + k] : a[k*C + o];
            acc = is_sum ? acc + v : fmax(acc, v);
        }
        want[o] = acc;
    }

    T *din, *dout;
    CHK(hipMalloc(&din,  R*C*sizeof(T)));
    CHK(hipMalloc(&dout, OUT*sizeof(T)));
    CHK(hipMemcpy(din, host.data(), R*C*sizeof(T), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, OUT*sizeof(T)));

    gl<T,1,1,-1,-1> gin (din,  nullptr, nullptr, R, C);
    gl<T,1,1,1,-1>  gout(dout, nullptr, nullptr, nullptr, OUT);
    hipLaunchKernelGGL((k_red<T,L,W>), dim3(1), dim3(32), 0, 0, gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<T> res(OUT);
    CHK(hipMemcpy(res.data(), dout, OUT*sizeof(T), hipMemcpyDeviceToHost));

    double worst = 0; int worst_i = -1;
    for (int i = 0; i < OUT; i++) {
        double d = fabs(to_d(res[i]) - want[i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    if (worst > 1e-6) {
        printf("%-20s FAIL  at %d: got %.4f want %.4f\n", name, worst_i,
               to_d(res[worst_i]), want[worst_i]);
        failures++;
    } else printf("%-20s PASS\n", name);

    CHK(hipFree(din)); CHK(hipFree(dout));
}

int main() {
    using row = ducks::rt_layout::row;
    using col = ducks::rt_layout::col;

    t_red<float, row, ROW_SUM>("fl  row row_sum");
    t_red<float, row, ROW_MAX>("fl  row row_max");
    t_red<float, row, COL_SUM>("fl  row col_sum");
    t_red<float, row, COL_MAX>("fl  row col_max");

    t_red<float, col, ROW_SUM>("fl  col row_sum");
    t_red<float, col, ROW_MAX>("fl  col row_max");
    t_red<float, col, COL_SUM>("fl  col col_sum");
    t_red<float, col, COL_MAX>("fl  col col_max");

    t_red<bf16, row, ROW_SUM>("bf  row row_sum");
    t_red<bf16, row, COL_SUM>("bf  row col_sum");
    t_red<bf16, col, ROW_SUM>("bf  col row_sum");
    t_red<bf16, col, COL_SUM>("bf  col col_sum");

    printf(failures ? "\n%d FAILED\n" : "\nALL PASS\n", failures);
    return failures != 0;
}
