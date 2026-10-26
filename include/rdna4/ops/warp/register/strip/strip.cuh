/**
 * @file
 * @brief Warp operations on register strips (rs) and their per-row vectors
 * (rs_cv): maps, row and column reductions, broadcasts, fills, conversions.
 *
 * Everything here but the row reductions is per lane: a strip's lanes own
 * disjoint runs, so an elementwise op never crosses a lane and a column
 * reduction (over a strip's rows) never does either. A row reduction folds
 * the lane's own runs into one value per row, then a butterfly across the
 * wave -- DPP row_xmask within a half, v_permlanex16 across -- which leaves
 * the same bits in every lane: each stage combines a value with its
 * partner's, and the partner computes the same pair the other way round,
 * which IEEE addition, max and min give identically.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

namespace detail {

/// Fold N values with `op`, as a balanced tree: a lane's runs are
/// independent, and a chain of N-1 dependent adds is N-1 latencies.
template<typename op, typename T, int N>
__device__ inline T strip_tree(T (&a)[N]) {
    #pragma unroll
    for (int n = N; n > 1; n = (n + 1) / 2) {
        const int h = (n + 1) / 2;
        #pragma unroll
        for (int i = 0; i < n - h; i++) a[i] = op::template op<T>(a[i], a[i + h]);
    }
    return a[0];
}

/// Combine one value across all 32 lanes; every lane ends with the answer.
template<typename op, typename T>
__device__ inline T strip_butterfly(T v) {
    v = op::template op<T>(v, lane_xor<1>(v));
    v = op::template op<T>(v, lane_xor<2>(v));
    v = op::template op<T>(v, lane_xor<4>(v));
    v = op::template op<T>(v, lane_xor<8>(v));
    v = op::template op<T>(v, lane_xor<HALF_WAVE_SWAP>(v));
    return v;
}

template<typename RS, typename F>
__device__ inline void strip_each(F &&f) {
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) f(r, j, e);
        }
    }
}

template<typename RS, typename U>
__device__ inline typename RS::T strip_scalar(const U &u) {
    using T = typename RS::T;
    if constexpr (std::is_same_v<U, T>) return u;
    else return base_types::convertor<T, float>::convert(float(u));
}

} // namespace detail

/* ----------  MAPS  ---------- */

template<typename op, ducks::rs::all RS>
__device__ static inline void unary_map(RS &dst, const RS &src) {
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e]);
    });
}
template<typename op, ducks::rs::cv CV>
__device__ static inline void unary_map(CV &dst, const CV &src) {
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = op::template op<typename CV::T>(src.data[r][0]);
}

/// A binary map whose right operand is another strip of the same type or a
/// scalar converted to the element type.
template<typename op, ducks::rs::all RS, typename U>
__device__ static inline void bin_map(RS &dst, const RS &lhs, const U &rhs) {
    using T = typename RS::T;
    if constexpr (std::is_same_v<U, RS>) {
        detail::strip_each<RS>([&](int r, int j, int e) {
            dst.data[r][j][e] = op::template op<T>(lhs.data[r][j][e], rhs.data[r][j][e]);
        });
    } else {
        const T s = detail::strip_scalar<RS>(rhs);
        detail::strip_each<RS>([&](int r, int j, int e) {
            dst.data[r][j][e] = op::template op<T>(lhs.data[r][j][e], s);
        });
    }
}
template<typename op, ducks::rs::cv CV, typename U>
__device__ static inline void bin_map(CV &dst, const CV &lhs, const U &rhs) {
    using T = typename CV::T;
    #pragma unroll
    for (int r = 0; r < CV::length; r++) {
        if constexpr (std::is_same_v<U, CV>)
            dst.data[r][0] = op::template op<T>(lhs.data[r][0], rhs.data[r][0]);
        else
            dst.data[r][0] = op::template op<T>(lhs.data[r][0], detail::strip_scalar<CV>(rhs));
    }
}

#define KITTENS_STRIP_UNARY(name, opname)                                          \
    template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)          \
    __device__ static inline void name(S &dst, const S &src) {                     \
        unary_map<base_ops::opname, S>(dst, src);                                   \
    }
KITTENS_STRIP_UNARY(exp,   exp)
KITTENS_STRIP_UNARY(exp2,  exp2)
KITTENS_STRIP_UNARY(log,   log)
KITTENS_STRIP_UNARY(log2,  log2)
KITTENS_STRIP_UNARY(abs,   abs)
KITTENS_STRIP_UNARY(relu,  relu)
KITTENS_STRIP_UNARY(gelu,  gelu)
KITTENS_STRIP_UNARY(dgelu, dgelu)
KITTENS_STRIP_UNARY(sqrt,  sqrt)
KITTENS_STRIP_UNARY(rsqrt, rsqrt)
KITTENS_STRIP_UNARY(silu,  silu)
KITTENS_STRIP_UNARY(step,  step)
#undef KITTENS_STRIP_UNARY

#define KITTENS_STRIP_FILL(name, opname)                                           \
    template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)          \
    __device__ static inline void name(S &dst) {                                   \
        unary_map<base_ops::opname, S>(dst, dst);                                   \
    }
KITTENS_STRIP_FILL(zero,      zero)
KITTENS_STRIP_FILL(one,       one)
KITTENS_STRIP_FILL(pos_infty, pos_infty)
KITTENS_STRIP_FILL(neg_infty, neg_infty)
#undef KITTENS_STRIP_FILL

#define KITTENS_STRIP_BINARY(name, opname)                                         \
    template<typename S, typename U> requires (ducks::rs::all<S> || ducks::rs::cv<S>) \
    __device__ static inline void name(S &dst, const S &lhs, const U &rhs) {       \
        bin_map<base_ops::opname, S>(dst, lhs, rhs);                                \
    }
KITTENS_STRIP_BINARY(add, sum)
KITTENS_STRIP_BINARY(sub, sub)
KITTENS_STRIP_BINARY(mul, mul)
KITTENS_STRIP_BINARY(div, div)
KITTENS_STRIP_BINARY(max, max)
KITTENS_STRIP_BINARY(min, min)
#undef KITTENS_STRIP_BINARY

/**
 * @brief Copy a strip or a per-row vector, converting the element type.
 */
template<ducks::rs::all D, ducks::rs::all S>
__device__ static inline void copy(D &dst, const S &src) {
    static_assert(D::rows == S::rows && D::cols == S::cols && D::run == S::run,
                  "a strip copy keeps the shape and the run");
    detail::strip_each<D>([&](int r, int j, int e) {
        dst.data[r][j][e] =
            base_types::convertor<typename D::T, typename S::T>::convert(src.data[r][j][e]);
    });
}
template<ducks::rs::cv D, ducks::rs::cv S>
__device__ static inline void copy(D &dst, const S &src) {
    static_assert(D::length == S::length, "a vector copy keeps the length");
    #pragma unroll
    for (int r = 0; r < D::length; r++)
        dst.data[r][0] =
            base_types::convertor<typename D::T, typename S::T>::convert(src.data[r][0]);
}

/* ----------  REDUCTIONS  ---------- */

namespace detail {
template<typename op, bool reset, ducks::rs::cv CV, ducks::rs::all RS>
__device__ inline void strip_row_reduce(CV &dst, const RS &src, const CV &acc) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    using T = typename RS::T;
    constexpr int N = RS::chunks * RS::run;
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        T a[N];
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) a[j*RS::run + e] = src.data[r][j][e];
        }
        const T v = strip_butterfly<op>(strip_tree<op>(a));
        if constexpr (reset) dst.data[r][0] = v;
        else                 dst.data[r][0] = op::template op<T>(acc.data[r][0], v);
    }
}

template<typename op, bool reset, ducks::rs::all RV, ducks::rs::all RS>
__device__ inline void strip_col_reduce(RV &dst, const RS &src, const RV &acc) {
    static_assert(RV::rows == 1 && RV::cols == RS::cols && RV::run == RS::run,
                  "a column reduction gives the strip's row_vec");
    using T = typename RS::T;
    #pragma unroll
    for (int j = 0; j < RS::chunks; j++) {
        #pragma unroll
        for (int e = 0; e < RS::run; e++) {
            T v = src.data[0][j][e];
            #pragma unroll
            for (int r = 1; r < RS::rows; r++) v = op::template op<T>(v, src.data[r][j][e]);
            if constexpr (reset) dst.data[0][j][e] = v;
            else                 dst.data[0][j][e] = op::template op<T>(acc.data[0][j][e], v);
        }
    }
}
} // namespace detail

#define KITTENS_STRIP_REDUCE(name, opname, kind, VEC)                              \
    template<ducks::rs::VEC V, ducks::rs::all RS>                                   \
    __device__ static inline void name(V &dst, const RS &src) {                    \
        detail::strip_##kind##_reduce<base_ops::opname, true>(dst, src, dst);      \
    }                                                                              \
    template<ducks::rs::VEC V, ducks::rs::all RS>                                   \
    __device__ static inline void name(V &dst, const RS &src, const V &acc) {      \
        detail::strip_##kind##_reduce<base_ops::opname, false>(dst, src, acc);     \
    }
KITTENS_STRIP_REDUCE(row_sum,  sum, row, cv)
KITTENS_STRIP_REDUCE(row_max,  max, row, cv)
KITTENS_STRIP_REDUCE(row_min,  min, row, cv)
KITTENS_STRIP_REDUCE(row_prod, mul, row, cv)
KITTENS_STRIP_REDUCE(col_sum,  sum, col, all)
KITTENS_STRIP_REDUCE(col_max,  max, col, all)
KITTENS_STRIP_REDUCE(col_min,  min, col, all)
KITTENS_STRIP_REDUCE(col_prod, mul, col, all)
#undef KITTENS_STRIP_REDUCE

/* ----------  BROADCASTS  ---------- */

/// `op` row r's value into every element of row r.
template<typename op, ducks::rs::all RS, ducks::rs::cv CV>
__device__ static inline void row_map(RS &dst, const RS &src, const CV &v) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e], v.data[r][0]);
    });
}
/// `op` column c's value into every element of column c.
template<typename op, ducks::rs::all RS>
__device__ static inline void col_map(RS &dst, const RS &src, const typename RS::row_vec &v) {
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e], v.data[0][j][e]);
    });
}

#define KITTENS_STRIP_BROADCAST(name, opname)                                      \
    template<ducks::rs::all RS, ducks::rs::cv CV>                                   \
    __device__ static inline void name##_row(RS &dst, const RS &src, const CV &v) { \
        row_map<base_ops::opname>(dst, src, v);                                     \
    }                                                                              \
    template<ducks::rs::all RS>                                                     \
    __device__ static inline void name##_col(RS &dst, const RS &src,               \
                                             const typename RS::row_vec &v) {      \
        col_map<base_ops::opname, RS>(dst, src, v);                                 \
    }
KITTENS_STRIP_BROADCAST(add, sum)
KITTENS_STRIP_BROADCAST(sub, sub)
KITTENS_STRIP_BROADCAST(mul, mul)
KITTENS_STRIP_BROADCAST(div, div)
#undef KITTENS_STRIP_BROADCAST

/* ----------  FILLS  ---------- */

namespace detail {
template<bool left, ducks::rs::all RS>
__device__ inline void strip_axis_fill(RS &dst, const RS &src, int col_idx,
                                       const typename RS::T &val) {
    const int lane0 = laneid() * RS::run;
    detail::strip_each<RS>([&](int r, int j, int e) {
        const int c = j*RS::span + lane0 + e;
        const bool hit = left ? c < col_idx : c >= col_idx;
        dst.data[r][j][e] = hit ? val : src.data[r][j][e];
    });
}
} // namespace detail

/// Columns col_idx and past it become `val`; the rest are copied from `src`.
template<ducks::rs::all RS>
__device__ static inline void right_fill(RS &dst, const RS &src, const int col_idx,
                                         const typename RS::T &val = 0) {
    detail::strip_axis_fill<false>(dst, src, col_idx, val);
}
/// Columns before col_idx become `val`; the rest are copied from `src`.
template<ducks::rs::all RS>
__device__ static inline void left_fill(RS &dst, const RS &src, const int col_idx,
                                        const typename RS::T &val = 0) {
    detail::strip_axis_fill<true>(dst, src, col_idx, val);
}

} // namespace kittens
