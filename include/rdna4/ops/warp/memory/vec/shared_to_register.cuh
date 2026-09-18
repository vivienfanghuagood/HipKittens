/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/*
 * gfx12 register-vector layouts. All three follow from the tile layouts in
 * types/register/rt_base.cuh -- a register vector is just one axis of a tile.
 *
 *   naive : entry i lives in lane i%32, outer step i/32. No replication; the
 *           layout for plain vector work that never touches a tile.
 *   ortho : indexed along the *lane* axis (l%16), so entry 16*w + l%16 sits in
 *           data[w][0], replicated across the two wave halves.
 *   align : indexed along the *element* axis, so the vector is replicated across
 *           the 16 lanes of a half. Each lane holds 8 of the 16 entries, but
 *           which 8 depends on the element type -- a bf16/half operand's halves
 *           take the runs 0..7 and 8..15, an f32 accumulator's take the even and
 *           odd entries. That is what rv_align_elem() resolves. (On gfx11 the
 *           operand lane really did cover all 16; gfx12 halves it.)
 *
 * None of these are bandwidth-critical (a vector is 1/16 of a tile), so they are
 * all plain elementwise accesses. Lanes that map to the same entry issue the
 * same address and LDS broadcasts them.
 */

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = typename RV::dtype;
    using U  = typename SV::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);

    const int lane = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            #pragma unroll
            for(int p = 0; p < kittens::TILE_ROW_DIM<T>; p++) {
                const int e = rv_align_elem<T>(p, lane);
                if(e < 0) continue;
                const T val = base_types::convertor<T, U>::convert(src.data[w*kittens::TILE_ROW_DIM<T> + p]);
                if(e & 1) dst.data[w][e>>1].y = val;
                else      dst.data[w][e>>1].x = val;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            dst[w][0] = base_types::convertor<T, U>::convert(
                src.data[w*kittens::TILE_ROW_DIM<T> + (lane & 15)]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            const int idx = w*WARP_THREADS + lane;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src.data[idx]);
            }
        }
    }
}

/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = typename RV::dtype;
    using U  = typename SV::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);

    const int lane = ::kittens::laneid();

    // In the replicated layouts several lanes hold the same entry and all write
    // it. The values agree by construction, so the duplicate writes are benign.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            #pragma unroll
            for(int p = 0; p < kittens::TILE_ROW_DIM<T>; p++) {
                const int e = rv_align_elem<T>(p, lane);
                if(e < 0) continue;
                const T val = (e & 1) ? src.data[w][e>>1].y : src.data[w][e>>1].x;
                dst.data[w*kittens::TILE_ROW_DIM<T> + p] = base_types::convertor<U, T>::convert(val);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            dst.data[w*kittens::TILE_ROW_DIM<T> + (lane & 15)] =
                base_types::convertor<U, T>::convert(src[w][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            const int idx = w*WARP_THREADS + lane;
            if(idx < src.length) {
                dst.data[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}

}
