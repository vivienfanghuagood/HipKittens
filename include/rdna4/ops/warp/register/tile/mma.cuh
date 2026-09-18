/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * gfx12 WMMA.  One shape only here: 16x16x16, wave32, f32 accumulate.
 *
 * The change from gfx11 is the operand fragment.  There a lane held the whole
 * K=16 vector and the two wave halves had to mirror each other, costing 8 VGPRs
 * for a bf16 operand; here the halves split k (lower half holds k 0..7, upper
 * half k 8..15), so an operand is v8f16 / v8bf16 and costs 4 (v2i and 2 for
 * fp8, see below).  That halving is
 * the main reason RDNA4 can carry a bigger register block than RDNA3 at the
 * same 256-VGPR budget.  rt_base<bf16|half, *> is sized for exactly that
 * (replication 1, 4 VGPRs), so the fragments stay a straight reinterpret of
 * `.data`.  The accumulator is unchanged: v8f32, held as float2[4].
 *
 * The builtins carry a `_gfx12` suffix and are mutually exclusive with the
 * gfx11 ones -- neither set compiles for the other target -- which is why this
 * is a separate header tree rather than an #ifdef inside the RDNA3 one.
 *
 * !! NOT HARDWARE-VERIFIED.  The signatures below were confirmed by compiling
 * for gfx1201, and the resulting v_wmma_* instructions were read back out of
 * the ISA, but no gfx12 part has executed this code.  The element-to-lane
 * mapping in particular is inferred from the fragment width; see the note on
 * WMMA_HALF_SHIFT in common/util.cuh. !!
 */

typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 wmma_half8_t;
typedef __attribute__((__vector_size__(8 * sizeof(short))))  short  wmma_short8_t;
typedef __attribute__((__vector_size__(8 * sizeof(float))))  float  wmma_float8_t;
typedef __attribute__((__vector_size__(2 * sizeof(int))))    int    wmma_int2_t;

__device__ static inline void wmma161616(float2 (&D)[4],
                                         const half_2 (&A)[4],
                                         const half_2 (&B)[4],
                                         const float2 (&C)[4]) {
    *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        *(wmma_half8_t*)A,
        *(wmma_half8_t*)B,
        *(wmma_float8_t*)C
    );
}

__device__ static inline void wmma161616(float2 (&D)[4],
                                         const bf16_2 (&A)[4],
                                         const bf16_2 (&B)[4],
                                         const float2 (&C)[4]) {
    *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
        *(wmma_short8_t*)A,
        *(wmma_short8_t*)B,
        *(wmma_float8_t*)C
    );
}

/*
 * The fp8 forms, which gfx11 does not have at all.
 *
 * Four instructions, one per (A encoding, B encoding) pair -- AMD's mnemonic
 * calls e4m3 "fp8" and e5m2 "bf8", and the two can be mixed freely, so A and B
 * are independent template parameters rather than one shared type.  The
 * encoding is baked into the opcode; there is no operand-select bit to set,
 * which is why this dispatches at compile time and costs nothing.
 *
 * The shape is the same 16x16x16 as bf16, *not* CDNA's K=32 fp8 MFMA.  So an
 * fp8 operand is the bf16 fragment with narrower elements: still 8 per lane,
 * still k = 8*(l/16) .. +7, but 8 bytes and 2 VGPRs instead of 16 and 4.  That
 * is why nothing in TILE_ROW_DIM / TILE_COL_DIM needs an fp8 specialization
 * the way the CDNA trees do, and why rt_base_coord() is shared verbatim.
 *
 * The consequence worth knowing before reaching for this: at a fixed 16x16x16
 * shape, fp8 buys registers and LDS traffic, not FLOPs.  One WMMA is one WMMA.
 * The win is that an operand tile costs half of bf16, so the register block can
 * grow -- and the accumulator, which is f32 either way, is then the binding
 * constraint.
 */
template<typename AT, typename BT>
    requires ((std::is_same_v<AT, fp8e4m3_4> || std::is_same_v<AT, fp8e5m2_4>) &&
              (std::is_same_v<BT, fp8e4m3_4> || std::is_same_v<BT, fp8e5m2_4>))
__device__ static inline void wmma161616(float2 (&D)[4],
                                         const AT (&A)[2],
                                         const BT (&B)[2],
                                         const float2 (&C)[4]) {
    const wmma_int2_t   a = *(const wmma_int2_t*)A;
    const wmma_int2_t   b = *(const wmma_int2_t*)B;
    const wmma_float8_t c = *(const wmma_float8_t*)C;
    constexpr bool a_e4m3 = std::is_same_v<AT, fp8e4m3_4>;
    constexpr bool b_e4m3 = std::is_same_v<BT, fp8e4m3_4>;

    if      constexpr ( a_e4m3 &&  b_e4m3)
        *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    else if constexpr ( a_e4m3 && !b_e4m3)
        *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(a, b, c);
    else if constexpr (!a_e4m3 &&  b_e4m3)
        *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(a, b, c);
    else
        *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c);
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * @param[out] d The output rt_base<float, col_layout> accumulator.
 * @param[in] a The first input rt_base<bf16, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float, col_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
/// fp8 A and B, in either encoding and independently of each other.
template<typename AT, typename BT>
    requires (ducks::base_types::fp8<AT> && ducks::base_types::fp8<BT>)
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<AT, ducks::rt_layout::row> &a,
                                    const rt_base<BT, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}

/**
 * @brief Base dot product operation for row layout.
 *
 * A row-layout operand and a col-layout operand have bit-identical register
 * contents on gfx12, exactly as on gfx11 -- lane l holds index l%16 of whichever
 * axis the layout names, and elements run along k either way -- so transposing
 * an operand is purely a relabelling and every variant below lowers to the same
 * instruction.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
template<typename AT, typename BT>
    requires (ducks::base_types::fp8<AT> && ducks::base_types::fp8<BT>)
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<AT, ducks::rt_layout::row> &a,
                                     const rt_base<BT, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::col> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
template<typename AT, typename BT>
    requires (ducks::base_types::fp8<AT> && ducks::base_types::fp8<BT>)
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<AT, ducks::rt_layout::col> &a,
                                     const rt_base<BT, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<half, ducks::rt_layout::col> &a,
                                      const rt_base<half, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
template<typename AT, typename BT>
    requires (ducks::base_types::fp8<AT> && ducks::base_types::fp8<BT>)
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<AT, ducks::rt_layout::col> &a,
                                      const rt_base<BT, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}

/// The type combinations gfx12 WMMA can actually take. The accumulator is
/// always f32 here -- the f16-accumulate form exists in hardware but is not
/// wired up. bf16 and half operands must match each other; the two fp8
/// encodings may be mixed, because there is a distinct opcode for each pair.
template<typename D, typename A, typename B, typename C>
constexpr bool mma_types_ok =
    std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float> &&
    ((std::is_same_v<typename A::T, typename B::T> &&
      (std::is_same_v<typename A::T, bf16> || std::is_same_v<typename A::T, half>))
     || (ducks::base_types::fp8<typename A::T> && ducks::base_types::fp8<typename B::T>));

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx12 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Dot product operation for row layout.
 *
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx12 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx12 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx12 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
}
