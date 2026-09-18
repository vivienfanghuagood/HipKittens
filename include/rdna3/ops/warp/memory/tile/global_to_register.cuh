/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/*
 * Same gfx11 lane mapping as the shared-memory path (rt_base_coord in
 * types/register/rt_base.cuh), minus the swizzle: global tiles are plain
 * row-major with `row_stride` between rows.
 *
 * Only the row-layout 2-byte operand is contiguous per lane -- 16 elements =
 * 32 B, issued as two 16-byte accesses. Everything else is strided and goes
 * elementwise, which is what the CDNA tree already does for its column path.
 *
 * As there, lanes l and l+16 of an operand tile address identical bytes. That
 * is the mirroring WMMA requires; the coalescer merges them.
 *
 * Unlike LDS, global rows are only as aligned as `row_stride` makes them: the
 * allocation base is 256 B from hipMalloc, but row r starts at r*row_stride
 * elements, so a stride of e.g. 8200 bf16 leaves every odd row 16-byte
 * misaligned. The wide path is therefore selected at runtime (a warp-uniform
 * scalar branch) and falls back to the elementwise loop, which needs no
 * alignment at all, when the tile does not qualify.
 */

/**
 * @brief Load data from a source array into a register tile.
 *
 * @tparam RT The register tile type.
 * @tparam GL The global layout type.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param idx[in] The index of the tile to load data from.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const U *src_ptr = (const U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    const int lane = laneid();
    const int l16  = lane & 15;

    // 16-byte alignment of every row this warp will touch. Uniform across the
    // warp, so the branch below is a scalar branch, not divergence.
    const bool wide = vectorizable &&
        ((reinterpret_cast<uintptr_t>(src_ptr) | (row_stride * sizeof(U))) & 15) == 0;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T val = base_types::convertor<T, U>::convert(
                src_ptr[(row_base + c.x)*row_stride + col_base + c.y]);
            if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
            else       dst.tiles[i][j].data[e>>1].x = val;
        }
    };

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int row_base = i*dst.tile_size_row;
            const int col_base = j*dst.tile_size_col;
            if constexpr (vectorizable) {
                if (wide) {
                    const U *p = &src_ptr[(row_base + l16)*row_stride + col_base];
                    #pragma unroll
                    for(int h = 0; h < 2; h++) {
                        const float4 v = *reinterpret_cast<const float4*>(p + 8*h);
                        __builtin_memcpy((void*)&dst.tiles[i][j].data[4*h], &v, sizeof(v));
                    }
                }
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL, COORD>(dst, src, idx);
}

/**
 * @brief Store data from a register tile to a destination array in global memory.
 *
 * @tparam RT The register tile type.
 * @tparam GL The global layout type.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param[in] idx The index of the tile to store data to.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int lane = laneid();
    const int l16  = lane & 15;

    const bool wide = vectorizable &&
        ((reinterpret_cast<uintptr_t>(dst_ptr) | (row_stride * sizeof(U))) & 15) == 0;

    // Lanes l and l+16 of an operand tile write the same bytes with the same
    // values. That redundancy is exactly the WMMA mirroring invariant, so the
    // write is benign whichever lane lands last.
    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T val = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                  : src.tiles[i][j].data[e>>1].x;
            dst_ptr[(row_base + c.x)*row_stride + col_base + c.y] =
                base_types::convertor<U, T>::convert(val);
        }
    };

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int row_base = i*src.tile_size_row;
            const int col_base = j*src.tile_size_col;
            if constexpr (vectorizable) {
                if (wide) {
                    U *p = &dst_ptr[(row_base + l16)*row_stride + col_base];
                    #pragma unroll
                    for(int h = 0; h < 2; h++) {
                        float4 v;
                        __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[4*h], sizeof(v));
                        *reinterpret_cast<float4*>(p + 8*h) = v;
                    }
                }
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

}
