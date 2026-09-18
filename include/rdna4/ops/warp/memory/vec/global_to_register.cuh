/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * Identical lane mapping to the shared-memory vector path next door; see the
 * comment there for the three layouts. Global memory just replaces src.data[]
 * with a raw pointer, and lanes that map to the same entry coalesce instead of
 * broadcasting.
 */

 /**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = typename RV::dtype;
    using U  = typename GL::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    const U *src_ptr = (const U*)&src[(idx.template unit_coord<-1, 3>())];
    const int lane = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            #pragma unroll
            for(int p = 0; p < kittens::TILE_ROW_DIM<T>; p++) {
                const int e = rv_align_elem<T>(p, lane);
                if(e < 0) continue;
                const T val = base_types::convertor<T, U>::convert(src_ptr[w*kittens::TILE_ROW_DIM<T> + p]);
                if(e & 1) dst.data[w][e>>1].y = val;
                else      dst.data[w][e>>1].x = val;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            dst[w][0] = base_types::convertor<T, U>::convert(
                src_ptr[w*kittens::TILE_ROW_DIM<T> + (lane & 15)]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            const int i = w*WARP_THREADS + lane;
            if(i < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[i]);
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = typename RV::dtype;
    using U  = typename GL::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    const int lane = ::kittens::laneid();

    // As in the shared path, replicated layouts have several lanes writing the
    // same entry with the same value.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            #pragma unroll
            for(int p = 0; p < kittens::TILE_ROW_DIM<T>; p++) {
                const int e = rv_align_elem<T>(p, lane);
                if(e < 0) continue;
                const T val = (e & 1) ? src.data[w][e>>1].y : src.data[w][e>>1].x;
                dst_ptr[w*kittens::TILE_ROW_DIM<T> + p] = base_types::convertor<U, T>::convert(val);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            dst_ptr[w*kittens::TILE_ROW_DIM<T> + (lane & 15)] =
                base_types::convertor<U, T>::convert(src[w][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            const int i = w*WARP_THREADS + lane;
            if(i < src.length) {
                dst_ptr[i] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}
} // namespace kittens
