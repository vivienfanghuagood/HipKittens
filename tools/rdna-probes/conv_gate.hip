// Phase 5 gate, part 2: register-tile conversions.
//
// Three families, and the interesting one is the middle:
//
//   swap_layout / transpose -- the 16x16 bit-swap network across lanes. Checked
//     both ways round: a swap_layout round trip must be the identity on the
//     matrix, and transpose_sep must be a genuine transpose.
//
//   copy across replication classes -- rt_fl (8 elements per lane, halves
//     interleaved) <-> rt_bf (16 elements per lane, halves mirrored). The
//     widening direction is the permlanex16 merge that attention needs; the
//     narrowing direction is a discard. If the merge picks the wrong parity the
//     result is a matrix with every other row duplicated, which is why the test
//     data varies along both axes.
//
//   masks and fills -- coordinate predicates, checked against the same
//     predicates on the host.
//
// Every value is a small integer so bf16 is exact and a mismatch is a layout bug
// rather than rounding.

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
constexpr int FILL_IDX = 11;   // deliberately not a multiple of 16
static const float FILLV = -9.0f;

/* ----------  KERNELS  ---------- */

template<typename T, typename L>
__global__ void k_swap(const gl<T,1,1,-1,-1> in, const gl<T,1,1,-1,-1> out) {
    rt<T,R,C,L> t;
    load(t, in, {0,0,0,0});
    store(out, swap_layout_inplace(t), {0,0,0,0});
}

template<typename T, typename L>
__global__ void k_trans(const gl<T,1,1,-1,-1> in, const gl<T,1,1,-1,-1> out) {
    rt<T,R,C,L> t, d;
    load(t, in, {0,0,0,0});
    transpose_sep(d, t);
    store(out, d, {0,0,0,0});
}

template<typename L>
__global__ void k_widen(const gl<float,1,1,-1,-1> in, const gl<bf16,1,1,-1,-1> out) {
    rt<float,R,C,L> a; rt<bf16,R,C,L> b;
    load(a, in, {0,0,0,0});
    copy(b, a);
    store(out, b, {0,0,0,0});
}

template<typename L>
__global__ void k_narrow(const gl<bf16,1,1,-1,-1> in, const gl<float,1,1,-1,-1> out) {
    rt<bf16,R,C,L> b; rt<float,R,C,L> a;
    load(b, in, {0,0,0,0});
    copy(a, b);
    store(out, a, {0,0,0,0});
}

enum MaskOp { CAUSAL, CAUSAL_T, TRIL, TRIU, RFILL, LFILL, UFILL, DFILL };

template<typename T, typename L, MaskOp OP>
__global__ void k_mask(const gl<T,1,1,-1,-1> in, const gl<T,1,1,-1,-1> out) {
    rt<T,R,C,L> t, d;
    load(t, in, {0,0,0,0});
    const T v = base_types::convertor<T,float>::convert(FILLV);
    if      constexpr (OP == CAUSAL)   make_causal  (d, t, v);
    else if constexpr (OP == CAUSAL_T) make_causal_t(d, t, v);
    else if constexpr (OP == TRIL)     tril (d, t, FILL_IDX, v);
    else if constexpr (OP == TRIU)     triu (d, t, FILL_IDX, v);
    else if constexpr (OP == RFILL)    right_fill(d, t, FILL_IDX, v);
    else if constexpr (OP == LFILL)    left_fill (d, t, FILL_IDX, v);
    else if constexpr (OP == UFILL)    upper_fill(d, t, FILL_IDX, v);
    else                               lower_fill(d, t, FILL_IDX, v);
    store(out, d, {0,0,0,0});
}

/* ----------  HARNESS  ---------- */

// Distinct per (row, col) and exact in bf16: 16*row + col is at most 527, and
// bf16 has 8 mantissa bits, so make the spread multiplicative instead.
static double ref_in(int r, int c) { return (double)(((r * 3 + c * 5) % 13) - 6) + (r % 2 ? 0.5 : 0.0); }

// The launch goes through a lambda because <<<>>> needs the kernel symbol
// itself; a function pointer parameter will not do.
template<typename TI, typename TO, typename Launch, typename Ref>
static void run(const char* name, Launch&& launch, Ref&& ref) {
    std::vector<TI> host(R*C);
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++)
        host[r*C+c] = base_types::convertor<TI,float>::convert((float)ref_in(r,c));

    TI *din; TO *dout;
    CHK(hipMalloc(&din,  R*C*sizeof(TI)));
    CHK(hipMalloc(&dout, R*C*sizeof(TO)));
    CHK(hipMemcpy(din, host.data(), R*C*sizeof(TI), hipMemcpyHostToDevice));
    CHK(hipMemset(dout, 0, R*C*sizeof(TO)));

    gl<TI,1,1,-1,-1> gin (din,  nullptr, nullptr, R, C);
    gl<TO,1,1,-1,-1> gout(dout, nullptr, nullptr, R, C);
    launch(gin, gout);
    CHK(hipGetLastError()); CHK(hipDeviceSynchronize());

    std::vector<TO> res(R*C);
    CHK(hipMemcpy(res.data(), dout, R*C*sizeof(TO), hipMemcpyDeviceToHost));

    double worst = 0; int wr = -1, wc = -1;
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) {
        double d = fabs(to_d(res[r*C+c]) - ref(r,c));
        if (d > worst) { worst = d; wr = r; wc = c; }
    }
    if (worst > 1e-6) {
        printf("%-24s FAIL  at (%d,%d): got %.3f want %.3f\n", name, wr, wc,
               to_d(res[wr*C+wc]), ref(wr,wc));
        failures++;
    } else printf("%-24s PASS\n", name);

    CHK(hipFree(din)); CHK(hipFree(dout));
}

#define RUN(name, TI, TO, KERNEL, REF)                                      \
    run<TI,TO>(name,                                                        \
        [](gl<TI,1,1,-1,-1> a, gl<TO,1,1,-1,-1> b) {                        \
            hipLaunchKernelGGL((KERNEL), dim3(1), dim3(32), 0, 0, a, b);    \
        }, REF)

/* ----------  HOST-SIDE PREDICATES  ---------- */

static double m_causal  (int r, int c) { return c <= r ? ref_in(r,c) : FILLV; }
static double m_causal_t(int r, int c) { return c >= r ? ref_in(r,c) : FILLV; }
static double m_tril    (int r, int c) { return c <= r - FILL_IDX ? ref_in(r,c) : FILLV; }
static double m_triu    (int r, int c) { return c <  r - FILL_IDX ? FILLV : ref_in(r,c); }
static double m_rfill   (int r, int c) { return c >= FILL_IDX ? FILLV : ref_in(r,c); }
static double m_lfill   (int r, int c) { return c <  FILL_IDX ? FILLV : ref_in(r,c); }
static double m_ufill   (int r, int c) { return r <  FILL_IDX ? FILLV : ref_in(r,c); }
static double m_dfill   (int r, int c) { return r >= FILL_IDX ? FILLV : ref_in(r,c); }

#define MASKS(T, L, tag)                                                     \
    RUN(tag " make_causal",   T, T, (k_mask<T,L,CAUSAL>),   m_causal);       \
    RUN(tag " make_causal_t", T, T, (k_mask<T,L,CAUSAL_T>), m_causal_t);     \
    RUN(tag " tril",          T, T, (k_mask<T,L,TRIL>),     m_tril);         \
    RUN(tag " triu",          T, T, (k_mask<T,L,TRIU>),     m_triu);         \
    RUN(tag " right_fill",    T, T, (k_mask<T,L,RFILL>),    m_rfill);        \
    RUN(tag " left_fill",     T, T, (k_mask<T,L,LFILL>),    m_lfill);        \
    RUN(tag " upper_fill",    T, T, (k_mask<T,L,UFILL>),    m_ufill);        \
    RUN(tag " lower_fill",    T, T, (k_mask<T,L,DFILL>),    m_dfill)

int main() {
    using row = ducks::rt_layout::row;
    using col = ducks::rt_layout::col;

    auto ident = [](int r, int c) { return ref_in(r, c); };
    auto trans = [](int r, int c) { return ref_in(c, r); };

    RUN("fl row swap_layout", float, float, (k_swap<float,row>), ident);
    RUN("fl col swap_layout", float, float, (k_swap<float,col>), ident);
    RUN("bf row swap_layout", bf16,  bf16,  (k_swap<bf16, row>), ident);
    RUN("bf col swap_layout", bf16,  bf16,  (k_swap<bf16, col>), ident);

    RUN("fl row transpose",   float, float, (k_trans<float,row>), trans);
    RUN("fl col transpose",   float, float, (k_trans<float,col>), trans);
    RUN("bf row transpose",   bf16,  bf16,  (k_trans<bf16, row>), trans);
    RUN("bf col transpose",   bf16,  bf16,  (k_trans<bf16, col>), trans);

    RUN("fl->bf row copy",    float, bf16,  (k_widen <row>), ident);
    RUN("fl->bf col copy",    float, bf16,  (k_widen <col>), ident);
    RUN("bf->fl row copy",    bf16,  float, (k_narrow<row>), ident);
    RUN("bf->fl col copy",    bf16,  float, (k_narrow<col>), ident);

    MASKS(float, row, "fl row");
    MASKS(float, col, "fl col");
    MASKS(bf16,  row, "bf row");

    printf(failures ? "\n%d FAILED\n" : "\nALL PASS\n", failures);
    return failures != 0;
}
