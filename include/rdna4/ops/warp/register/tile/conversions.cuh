/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * Everything in this file is derived from one picture. Index a base tile by
 * (lane_low, pos), where lane_low = laneid & 15 and pos is the coordinate that
 * rt_base_coord() builds from the element index:
 *
 *   row layout: (row, col) = (lane_low, pos)
 *   col layout: (row, col) = (pos, lane_low)
 *
 * How pos is physically stored is the whole difference between the two kinds of
 * tile gfx12 has.  Both spread the 16 positions over the two wave halves, 8 per
 * lane, and differ only in which bit of pos the wave-half bit carries:
 *
 *   operands (bf16/half): pos = e + 8*(laneid >> 4). Bit 3 of pos lives in the
 *     wave-half bit; bits 2:0 are the element index.
 *
 *   accumulators (float): pos = 2*e + (laneid >> 4). Bit 0 of pos lives in the
 *     wave-half bit; bits 3:1 are the element index.
 *
 * So each shape is `pos = element_stride*e + half_shift*h`, and everything below
 * is parameterised on log2(half_shift) -- 3 for an operand, 0 for an
 * accumulator -- rather than on the RDNA3 tree's `halves_interleave` flag, which
 * there distinguished the mirrored operand (half bit carries nothing) from the
 * interleaved accumulator.  On gfx12 there is no mirroring and no such split.
 *
 * That is why the conversions here are not the CDNA ones with a different lane
 * formula -- on CDNA every tile spreads its element axis the same way, so copy()
 * between two dtypes is pointwise. Here a float tile and a bf16 tile of the same
 * shape hold their values in genuinely different places, and copy() has to move
 * data across the wave halves.
 *
 * !! NOT HARDWARE-VERIFIED.  The operand half-split these derivations rest on is
 * inferred from the builtin's fragment width; see the note on WMMA_HALF_SHIFT in
 * common/util.cuh. !!
 */

/* ----------  LAYOUT SWAPS  ---------- */

namespace detail {

/// log2 of a power of two, for turning `half_shift` / `element_stride` into the
/// bit positions transpose_base_data is parameterised on.
__device__ static constexpr int log2_exact(int n) { int r = 0; while(n > 1) { n >>= 1; r++; } return r; }

/**
 * @brief One stage of the transpose network: swap bit i of lane_low with bit i of pos.
 *
 * Every value whose two bits disagree moves to the lane with LANEMASK flipped and
 * the element with ELEMMASK flipped. That is a symmetric pair swap, so a single
 * butterfly carries both directions at once.
 *
 * The selects are written as ternaries rather than branches on purpose: `a` is
 * divergent, and a lane exchange has to be issued with the whole wave active.
 */
template<int LANEMASK, int ELEMMASK, typename T, int E>
__device__ static inline void xpose_stage(T (&v)[E], const int lane) {
    const bool a = (lane & LANEMASK) != 0;
    #pragma unroll
    for(int e = 0; e < E; e++) {
        if((e & ELEMMASK) != 0) continue;              // visit each pair once
        const T lo = v[e], hi = v[e | ELEMMASK];
        const T recv = lane_xor<LANEMASK>(a ? lo : hi);
        v[e]            = a ? recv : lo;
        v[e | ELEMMASK] = a ? hi   : recv;
    }
}

/**
 * @brief Transpose a base tile's 16x16 (lane_low, pos) grid.
 *
 * swap_layout and transpose are the same movement in that grid -- one relabels
 * the axes, the other relabels the matrix -- so they share this routine and
 * differ only in the type they hand back.
 *
 * Four stages, stage i swapping bit i of lane_low with bit i of pos.  Exactly
 * one bit of pos -- bit HALF_BIT = log2(half_shift) -- is carried by the wave
 * half rather than by the element index, and that stage is the odd one out:
 *
 *   the HALF_BIT stage has to swap lane bit HALF_BIT with lane bit 4.  That is
 *     not a power-of-two butterfly; it is lane ^ ((1<<HALF_BIT) | 16), applied
 *     only to the lanes where the two bits disagree (those lanes swap their
 *     entire register file, since all of a lane's values share that pos bit).
 *
 *   the other three stages are the ordinary butterfly, pairing lane bit
 *     j+ES_SHIFT with element-index bit j, where ES_SHIFT = log2(element_stride).
 *
 * For the accumulator (ES_SHIFT 1, HALF_BIT 0) that is lane^17 plus stages
 * <2,1> <4,2> <8,4>; for an operand (ES_SHIFT 0, HALF_BIT 3) it is lane^24 plus
 * <1,1> <2,2> <4,4>.  Either way: 8 permlanex16 and 12 DPP movs per base tile.
 * Compare a round trip through LDS, which is what the CDNA code's
 * dynamically-indexed __shfl lowers to.
 *
 * The RDNA3 version of this had a second, cheaper path for operands, because
 * there the halves were mirrors and the wave-half bit could simply be ignored --
 * 20 pure-DPP exchanges and no permlanex16.  gfx12 buys its halved operand
 * register cost partly here.
 */
template<int HALF_BIT, int ES_SHIFT, typename T2, int P>
__device__ static inline void transpose_base_data(T2 (&dst)[P], const T2 (&src)[P], const int lane) {
    using T = typename base_types::packing<T2>::unpacked_type;
    constexpr int E = 2 * P;
    static_assert(E == 8, "a gfx12 base tile holds 8 elements per lane");
    static_assert(HALF_BIT == 0 || HALF_BIT == 3, "only the two gfx12 fragment shapes are derived here");
    static_assert(ES_SHIFT == (HALF_BIT == 0 ? 1 : 0), "element_stride and half_shift must tile pos exactly");

    T v[E];
    #pragma unroll
    for(int k = 0; k < P; k++) { v[2*k] = src[k].x; v[2*k+1] = src[k].y; }

    // The stage that crosses the wave halves.
    const bool swap_halves = (((lane >> HALF_BIT) ^ (lane >> 4)) & 1) != 0;
    #pragma unroll
    for(int e = 0; e < E; e++) {
        const T other = lane_half_xor<HALF_BIT>(v[e]);
        v[e] = swap_halves ? other : v[e];
    }
    // The three in-half butterflies.
    xpose_stage<1 << (0 + ES_SHIFT), 1 << 0, T, E>(v, lane);
    xpose_stage<1 << (1 + ES_SHIFT), 1 << 1, T, E>(v, lane);
    xpose_stage<1 << (2 + ES_SHIFT), 1 << 2, T, E>(v, lane);

    #pragma unroll
    for(int k = 0; k < P; k++) { dst[k].x = v[2*k]; dst[k].y = v[2*k+1]; }
}

} // namespace detail

/**
 * @brief Swaps the layout of a register base tile.
 *
 * The matrix is unchanged; only the mapping from lanes and registers to its
 * entries is. See detail::transpose_base_data for why that is a transpose.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
template<typename T, ducks::rt_layout::all layout>
__device__ inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    using base = rt_base<T, layout>;
    detail::transpose_base_data<detail::log2_exact(base::half_shift),
                                detail::log2_exact(base::element_stride)>(
        dst.data, src.data, laneid());
}

/**
 * @brief Swaps the layout of a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}

/**
 * @brief Swaps the layout of a register tile in place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief Transposes a register base tile.
 *
 * Same movement as swap_layout, and safe when dst and src alias, because the
 * values are lifted into a local array first.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<typename T, ducks::rt_layout::all layout>
__device__ inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    using base = rt_base<T, layout>;
    detail::transpose_base_data<detail::log2_exact(base::half_shift),
                                detail::log2_exact(base::element_stride)>(
        dst.data, src.data, laneid());
}

/**
 * @brief Transposes a register tile.
 *
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam RT The type of the destination register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<ducks::rt::all RT>
__device__ static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}

/**
 * @brief Transposes a register tile by relabeling, without moving any data.
 *
 * The other transpose above moves values; this one does not move anything at all.
 * A base tile's storage depends only on its dtype -- the layout tag is purely how
 * (lane_low, pos) is read, (row, col) for row and (col, row) for col -- so reading
 * the same registers under the opposite tag already yields the transpose. Pair that
 * with sending tiles[i][j] to tiles[j][i] and the whole tile is transposed for free.
 *
 * dst and src must not alias; the grid permutation would overwrite as it goes.
 *
 * @param result[out] The transposed tile, of the mirrored shape and opposite layout.
 * @param tile[in] The tile to transpose.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline void transpose(rt<T2, _cols, _rows, typename ducks::rt_layout::transpose<layout>::type> &result, const rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for (int i = 0; i < tile.height; i++) {
        #pragma unroll
        for (int j = 0; j < tile.width; j++) {
            #pragma unroll
            for (int k = 0; k < tile.packed_per_tile; k++) {
                result.tiles[j][i].data[k] = tile.tiles[i][j].data[k];
            }
        }
    }
}

/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _rows The height of the register tile. (Must equal _cols.)
 * @tparam _cols The width of the register tile. (Must equal _rows.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * Between two tiles of the same element_stride this is pointwise, as on CDNA.
 * Across strides it is not, because the two kinds of tile spread `pos`
 * differently: an accumulator lane owns the 8 positions whose parity matches its
 * wave half, an operand lane owns the contiguous run 8h..8h+7.  Neither is a
 * subset of the other, so -- unlike RDNA3, where the operand held everything and
 * the operand->accumulator direction was a free discard -- *both* directions
 * have to cross the halves here.
 *
 * Both are the same 8 exchanges.  Step e hands a lane its own element e and its
 * mirror's element e, which between them are two known positions; the lane keeps
 * whichever of the two it needs and drops the other.  Half the exchanges are
 * therefore wasted in each half, and that is not avoidable with a symmetric
 * permlanex16: the element index a lane wants at a given step depends on its own
 * wave half, so the two partners can never both be asking for the register the
 * other one needs.
 *
 *   accumulator -> operand (float -> bf16/half).  This is the conversion
 *     attention needs between the score matrix and the second matmul's operand.
 *     Source element u is position 2u+h and source element u+4 is 2u+8+h; a lane
 *     in half 0 wants positions 2u and 2u+1 (from the first exchange), a lane in
 *     half 1 wants 2u+8 and 2u+9 (from the second).
 *
 *   operand -> accumulator.  Source element m is position m+8h.  Destination
 *     element t is position 2t+h and t+4 is 2t+8+h, which sit in element 2t+h of
 *     one half or the other depending on t.
 *
 * @tparam T The data type of the destination register elements.
 * @tparam U The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using D  = rt_base<T, layout>;
    using S  = rt_base<U, layout>;
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;

    // Read element e of a packed array, and write element e of another.
    auto get = [](const auto &a, int e) { return (e & 1) ? a[e>>1].y : a[e>>1].x; };

    if constexpr (D::element_stride == S::element_stride) {
        #pragma unroll
        for(int k = 0; k < D::packed_per_thread; k++) {
            dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
        }
    }
    else if constexpr (S::element_stride == 2) {
        // accumulator (pos 2e+h) -> operand (pos m+8h).
        const int h = laneid() >> 4;
        #pragma unroll
        for(int u = 0; u < D::packed_per_thread; u++) {   // 4
            // Step u: source element u carries position 2u+h, so after the
            // exchange this lane has both 2u and 2u+1 -- which half 0 wants.
            const U own_lo   = get(src.data, u);
            const U other_lo = lane_xor<HALF_WAVE_SWAP>(own_lo);
            // Step u+4: source element u+4 carries 2u+8+h, giving both 2u+8 and
            // 2u+9 -- which half 1 wants, as its operand elements 2u and 2u+1.
            const U own_hi   = get(src.data, u + S::elements_per_thread/2);
            const U other_hi = lane_xor<HALF_WAVE_SWAP>(own_hi);
            // Select on U = float here, then convert; see the note below on why
            // the other direction cannot be written that way round.
            dst.data[u].x = base_types::convertor<T, U>::convert(h ? other_hi : own_lo);
            dst.data[u].y = base_types::convertor<T, U>::convert(h ? own_hi   : other_lo);
        }
    }
    else {
        // operand (pos m+8h) -> accumulator (pos 2t+h).
        //
        // Convert before selecting, not after. U is bf16 or half here, which
        // are classes wrapping a short; a runtime select between two of them
        // materialises a temporary, and that is enough to make LLVM give up
        // on SROA and park the whole source tile in scratch. T is float in
        // this branch -- selecting on it stays in registers.
        const int h = laneid() >> 4;
        constexpr int Q = D::packed_per_thread;                // 4
        #pragma unroll
        for(int t = 0; t < Q; t++) {
            // Source element 2t carries position 2t+8h, so the exchange yields
            // positions 2t (from half 0) and 2t+8 (from half 1); element 2t+1
            // likewise yields 2t+1 and 2t+9.  Half 0 wants the even pair, half 1
            // the odd pair, as destination elements t and t+4.
            const T own_ev   = base_types::convertor<T, U>::convert(get(src.data, 2*t));
            const T other_ev = lane_xor<HALF_WAVE_SWAP>(own_ev);
            const T own_od   = base_types::convertor<T, U>::convert(get(src.data, 2*t + 1));
            const T other_od = lane_xor<HALF_WAVE_SWAP>(own_od);
            const T lo = h ? other_od : own_ev;    // position 2t + h
            const T hi = h ? own_od   : other_ev;  // position 2t + 8 + h
            if(t & 1) dst.data[t>>1].y = lo; else dst.data[t>>1].x = lo;
            const int th = t + Q;
            if(th & 1) dst.data[th>>1].y = hi; else dst.data[th>>1].x = hi;
        }
    }
}

/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/* ----------  MASKS AND FILLS  ---------- */

namespace detail {

/**
 * @brief Rewrite every element of a tile as a function of its (row, col).
 *
 * The CDNA versions of the masks below carry hand-derived lane formulas, and the
 * on-diagonal cases carry 64-bit magic masks indexed by laneid. None of that
 * survives a change of wave width or replication, so this asks rt_base_coord()
 * for the coordinate instead and writes the predicate out in full. These are not
 * hot -- they run once per tile of a masked matmul, not once per k-step -- and a
 * predicate the next person can read is worth more here than the mask trick.
 */
template<ducks::rt::all RT, typename F>
__device__ static inline void map_by_coord(RT &dst, const RT &src, F &&f) {
    using T    = typename RT::T;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;
    const int lane = laneid();
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for(int e = 0; e < base::elements_per_thread; e++) {
                const int2 c   = rt_base_coord<T, L>(e, lane);
                const int  row = i * base::tile_size_row + c.x;
                const int  col = j * base::tile_size_col + c.y;
                const T    val = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                         : src.tiles[i][j].data[e>>1].x;
                const T    out = f(row, col, val);
                if(e & 1) dst.tiles[i][j].data[e>>1].y = out;
                else      dst.tiles[i][j].data[e>>1].x = out;
            }
        }
    }
}

} // namespace detail

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param val[in] The value to fill the upper triangle with.
 */
template<ducks::rt::all RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return col <= row ? v : val; });
}

/**
 * @brief Makes a square register tile anti-causal by zeroing elements below the main diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param val[in] The value to fill the lower triangle with.
 */
template<ducks::rt::all RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return col >= row ? v : val; });
}

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by filling elements above the shifted diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return col <= row - row_idx ? v : val; });
}

/**
 * @brief Makes a register tile triangular by filling elements below the shifted diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return col < row - row_idx ? val : v; });
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief Fills a register tile with a value from a column index rightwards.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill from and onwards to the right.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx >= RT::cols) return;
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return col >= col_idx ? val : v; });
}

/**
 * @brief Fills a register tile with a value to the left of a column index.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill to the left of (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx <= 0) return;
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return col < col_idx ? val : v; });
}

/**
 * @brief Fills a register tile with a value above a row index.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill to, from the top (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx <= 0) return;
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return row < row_idx ? val : v; });
}

/**
 * @brief Fills a register tile with a value from a row index downwards.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill from and onwards to the bottom (inclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx >= RT::rows) return;
    using T = typename RT::T;
    detail::map_by_coord(dst, src, [&](int row, int col, T v) { return row >= row_idx ? val : v; });
}

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
__device__ inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}

}
