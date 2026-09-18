/**
 * @file
 * @brief The basic 16x16 register tile on which larger register tiles are built.
 */
 
#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"
#include "rv_layout.cuh"

namespace kittens {

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

namespace ducks {
/**
 * @namespace rt_base
 * 
 * @brief The namespace where concepts and abstract types for register base (16x16) tiles live.
 */
namespace rt_base {
/**
 * @brief A dummy type used to identify register base tiles.
 * 
 * For a type to quack like an rt_base, it should define its identifier as ducks::rt_base::identifier.
 * If a type quacks like ducks::rt_base::identifier, it will be treated as an rt_base by compiler checks.
 */
struct identifier {};
}
} // namespace ducks

/**
 * @brief Basic tile structure for computation in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _layout The layout of the base tile, either row-major or column-major.
 *
 * This type is a primarily utility for building larger inline templates
 * out of PTX primitives and managing layouts.
 *
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 *
 * The shapes that occur on gfx1200/gfx1201, with `l` the lane id and `e` the
 * element index within a lane (data[] is these packed two at a time for the
 * 2- and 4-byte types, four at a time for fp8):
 *
 *   rt_base<bf16|half, row>  A operand   e -> (row l%16, col 8*(l/16) + e), e in 0..7
 *   rt_base<bf16|half, col>  B operand   e -> (row 8*(l/16) + e, col l%16), e in 0..7
 *   rt_base<fp8e4m3|fp8e5m2, row|col>    same as the 2-byte operands
 *   rt_base<float, col>      accumulator e -> (row 2e + l/16, col l%16),    e in 0..7
 *   rt_base<float, row>      transposed  e -> (row l%16, col 2e + l/16),    e in 0..7
 *
 * fp8 shares the operand mapping exactly -- gfx12's fp8 WMMA is the same
 * 16x16x16 shape as its bf16 one, not CDNA's K=32 variant, so the only thing
 * that changes is that the lane's 8 elements are 8 bytes instead of 16.
 *
 * The convention matches CDNA's: `row` means the lane picks the row and the
 * elements run along the columns, `col` is the transpose.  Every shape here is
 * exactly 8 elements per lane -- gfx12 replicates nothing -- and the only thing
 * that varies is how the two wave halves divide the element axis: operands
 * split k into 0..7 and 8..15, the accumulator interleaves even and odd rows.
 *
 * !! The operand half-split is inferred, not measured.  See the note on
 * WMMA_HALF_SHIFT in common/util.cuh. !!
 */
template<typename _T, ducks::rt_layout::all _layout> struct rt_base {
    using identifier = ducks::rt_base::identifier; ///< Type identifier for the rt_base structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< Data type of the matrix elements

    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2>
        || std::is_same_v<dtype, fp8e4m3_4> || std::is_same_v<dtype, fp8e5m2_4>,
        "rt_base was provided an unsupported type."
    );

    static constexpr int tile_size_row        = kittens::TILE_ROW_DIM<T, layout>;
    static constexpr int tile_size_col        = kittens::TILE_COL_DIM<T, layout>;
    static constexpr int rows                 = tile_size_row; ///< Number of rows.
    static constexpr int cols                 = tile_size_col; ///< Number of cols.
    static constexpr int num_elements         = rows*cols;

    /// gfx12 duplicates nothing, so a wave tiles the 256 elements exactly.
    static constexpr int replication          = kittens::WMMA_REPLICATION<T>;
    static constexpr int elements_per_thread  = replication * num_elements / kittens::WARP_THREADS;

    static constexpr int packed_per_thread    = (elements_per_thread / base_types::packing<dtype>::num()) ;
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // registers are 32-bit words

    // 8 elements per lane whatever the type, so the VGPR cost follows the width:
    // 2 for an fp8 operand, 4 for a 2-byte one, 8 for the f32 accumulator. This
    // is the RDNA3 tree's static_assert(== 8) relaxed -- there, operand
    // mirroring made every type cost 8, and here it is the halved operand that
    // is the whole point. The three widths are exactly the three builtin
    // operand types: v2i, v8f16/v8bf16, v8f32.
    static_assert(elements_per_thread == 8, "unexpected gfx12 WMMA fragment size");
    static_assert(registers_per_thread == (std::is_same_v<T, float> ? 8
                                          : ducks::base_types::fp8<T> ? 2 : 4),
                  "unexpected gfx12 WMMA fragment width");

    /// Distance along the element axis between element `e` and element `e+1` of
    /// a single lane. 1 for a WMMA operand (a lane holds a contiguous run of k);
    /// 2 for the f32 accumulator (the two wave halves interleave along the axis).
    static constexpr int element_stride = kittens::WMMA_ELEMENT_STRIDE<T>;
    /// How far the upper wave half is shifted along that axis: 8 for an operand
    /// (the halves split k), 1 for the accumulator (they interleave rows).
    static constexpr int half_shift = kittens::WMMA_HALF_SHIFT<T>;
    /// Kept for symmetry with the RDNA3 tree, where it distinguished the mirrored
    /// operands from the interleaved accumulator. On gfx12 every shape is split.
    static constexpr bool halves_interleave = (half_shift != 0);

    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>; // for holding column reductions
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>; // for holding row reductions

    dtype data[packed_per_thread]; ///< The actual storage for the base tile
};

/**
 * @brief Where element `e` of lane `l` sits inside a 16x16 base tile, as {row, col}.
 *
 * `layout::row` means the lane selects the row and the elements run along the
 * columns; `col` is the transpose.  The position along that element axis is
 * `element_stride*e`, shifted by the wave half when the halves interleave.
 *
 * This is the single place the gfx12 fragment layouts are written down as
 * arithmetic.  Memory ops, reductions and layout conversions all derive their
 * indices from here rather than open-coding `laneid%16` the way the CDNA tree
 * does, because on RDNA the mapping differs between operands and accumulators.
 *
 * Note the one substantive difference from the RDNA3 tree: there `half_shift`
 * was 0 for an operand, because the halves were mirrors of each other and a
 * lane's element index did not depend on which half it was in.  On gfx12 there
 * is no mirroring, so every shape has a non-zero shift.
 */
template<typename T, ducks::rt_layout::all L>
__device__ static inline int2 rt_base_coord(int e, int lane) {
    using base = rt_base<T, L>;
    const int pos = base::element_stride * e + base::half_shift * (lane >> 4);
    const int l16 = lane & 15;
    if constexpr (std::is_same_v<L, ducks::rt_layout::row>) { return int2{l16, pos}; }
    else                                                    { return int2{pos, l16}; }
}

/**
 * @brief Where entry `p` of an rv_layout::align vector lives within a lane.
 *
 * An `align` vector is indexed along a tile's element axis, so it inherits that
 * axis' lane mapping: the vector is replicated across the 16 lanes of a wave
 * half, and each half holds only part of it -- every other entry for the f32
 * accumulator, a contiguous run of 8 for a WMMA operand.  (On RDNA3 the operand
 * case was the easy one, since both halves held everything; here it is not.)
 *
 * Returns the element index `e` in 0..elements_per_thread-1 that this lane uses
 * for entry `p` in 0..15, or -1 if this lane's half does not hold `p` at all.
 * It is the inverse of the `pos` computed by rt_base_coord().
 */
template<typename T>
__device__ static inline int rv_align_elem(int p, int lane) {
    using base = rt_base<T, ducks::rt_layout::row>; // stride/shift depend only on T
    // Invert pos = element_stride*e + half_shift*h for this lane's half, and
    // report -1 when no e in range produces p. For the accumulator that reads
    // "this half holds the even (or odd) entries"; for an operand, "this half
    // holds k in 0..7 (or 8..15)".
    const int q = p - base::half_shift * (lane >> 4);
    if (q < 0 || q >= base::element_stride * base::elements_per_thread) return -1;
    if constexpr (base::element_stride > 1) if (q % base::element_stride) return -1;
    return q / base::element_stride;
}

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt_base {
/**
* @brief Concept for all register base tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt_base::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
} // namespace rt
} // namespace ducks

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fl = rt_base<float, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_hf = rt_base<half, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e4m3 = rt_base<fp8e4m3, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e5m2 = rt_base<fp8e5m2, L>;
}
