/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template<ducks::rv::all RV2, ducks::rv::all RV1>
__device__ static inline void copy(RV2 &dst, const RV1 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;

    using D1_1 = base_types::packing<D1>::unpacked_type;
    using D1_2 = base_types::packing<D1_1>::packed_type;

    using D2_1 = base_types::packing<D2>::unpacked_type;
    using D2_2 = base_types::packing<D2_1>::packed_type;

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) {
        /*
         * naive and ortho give a lane one entry per outer step whatever the dtype,
         * so for them a typecast really is pointwise.
         *
         * align does not. Its outer step is a tile's element axis, and gfx12
         * spreads that axis differently depending on element_stride -- see the
         * copy() for rt_base, which this mirrors. An f32 lane holds the 8
         * positions whose parity matches its wave half; a bf16/half lane holds
         * the contiguous run 8h..8h+7. Both are 4 packed registers, so unlike
         * RDNA3 the inner_dims agree and cannot be used to tell the cases apart;
         * the discriminator is the stride itself.
         */
        constexpr bool is_align  = std::is_same_v<typename RV1::layout, ducks::rv_layout::align>;
        constexpr int SRC_STRIDE = kittens::WMMA_ELEMENT_STRIDE<D1_1>;
        constexpr int DST_STRIDE = kittens::WMMA_ELEMENT_STRIDE<D2_1>;
        if constexpr (!is_align || SRC_STRIDE == DST_STRIDE) {
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for(int j = 0; j < RV1::inner_dim; j++) {
                    dst[i][j] = base_types::convertor<D2, D1>::convert(src[i][j]);
                }
            }
        }
        else if constexpr (SRC_STRIDE == 2) {
            // accumulator -> operand. Entry e of this lane is position 2e+h and
            // entry e+4 is 2e+8+h; the mirror lane holds the other parity of
            // each. Two exchanges per output register, and h picks which pair
            // this half keeps. See the rt_base copy for the full derivation.
            const int h = laneid() >> 4;
            constexpr int Q = RV2::inner_dim;
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for(int u = 0; u < Q; u++) {
                    const int m = u + Q;
                    const D1_1 own_lo   = (u & 1) ? src[i][u>>1].y : src[i][u>>1].x;
                    const D1_1 other_lo = lane_xor<HALF_WAVE_SWAP>(own_lo);
                    const D1_1 own_hi   = (m & 1) ? src[i][m>>1].y : src[i][m>>1].x;
                    const D1_1 other_hi = lane_xor<HALF_WAVE_SWAP>(own_hi);
                    dst[i][u].x = base_types::convertor<D2_1, D1_1>::convert(h ? other_hi : own_lo);
                    dst[i][u].y = base_types::convertor<D2_1, D1_1>::convert(h ? own_hi   : other_lo);
                }
            }
        }
        else {
            // operand -> accumulator. Not a discard on gfx12: this lane's half
            // holds only the run 8h..8h+7, so half the positions it needs come
            // from the mirror. Convert before selecting -- see the rt_base copy
            // for why a select on bf16/half is what pushes the vector to scratch.
            const int h = laneid() >> 4;
            constexpr int Q = RV2::inner_dim;
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for(int t = 0; t < Q; t++) {
                    const D2_1 own_ev   = base_types::convertor<D2_1, D1_1>::convert(src[i][t].x);
                    const D2_1 other_ev = lane_xor<HALF_WAVE_SWAP>(own_ev);
                    const D2_1 own_od   = base_types::convertor<D2_1, D1_1>::convert(src[i][t].y);
                    const D2_1 other_od = lane_xor<HALF_WAVE_SWAP>(own_od);
                    const D2_1 lo = h ? other_od : own_ev;   // position 2t + h
                    const D2_1 hi = h ? own_od   : other_ev; // position 2t + 8 + h
                    const int  th = t + Q;
                    if(t & 1)  dst[i][t>>1].y  = lo; else dst[i][t>>1].x  = lo;
                    if(th & 1) dst[i][th>>1].y = hi; else dst[i][th>>1].x = hi;
                }
            }
        }
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::naive> && std::is_same_v<typename RV2::layout, ducks::rv_layout::align>) { 
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::align> && std::is_same_v<typename RV2::layout, ducks::rv_layout::naive>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::align> && std::is_same_v<typename RV2::layout, ducks::rv_layout::ortho>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::ortho> && std::is_same_v<typename RV2::layout, ducks::rv_layout::align>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::ortho> && std::is_same_v<typename RV2::layout, ducks::rv_layout::naive>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::naive> && std::is_same_v<typename RV2::layout, ducks::rv_layout::ortho>) {
        static_assert(false, "Unsupported layout conversion");
    } else {
        static_assert(false, "Unsupported layout conversion");
    }
}

} // namespace kittens