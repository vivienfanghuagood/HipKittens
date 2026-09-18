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
 * Same gfx12 lane mapping as the shared-memory path (rt_base_coord in
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
                                         && std::is_same_v<T, U> && (sizeof(U) == 2 || sizeof(U) == 1);
    /// A lane's whole fragment in one access: 16 bytes for a 2-byte type, 8 for fp8.
    constexpr int  vec_bytes = E * sizeof(U);

    const U *src_ptr = (const U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    const int lane = laneid();

    // Alignment of every row this warp will touch. Uniform across the warp, so
    // the branch below is a scalar branch, not divergence.
    const bool wide = vectorizable &&
        ((reinterpret_cast<uintptr_t>(src_ptr) | (row_stride * sizeof(U))) & (vec_bytes - 1)) == 0;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        // dtype packs 4 elements for fp8 and 2 otherwise, so index the unpacked
        // type rather than through .x/.y.
        T* elems = reinterpret_cast<T*>(&dst.tiles[i][j].data[0]);
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            elems[e] = base_types::convertor<T, U>::convert(
                src_ptr[(row_base + c.x)*row_stride + col_base + c.y]);
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
                    // One access, not two. The RDNA3 version of this loop ran
                    // h over 0..1 and wrote data[0] and data[4], because there
                    // a lane held all 16 of k in eight packed words; on gfx12
                    // the halves split k, so a lane holds 8 elements in four
                    // words (two for fp8) and its run starts at the column
                    // rt_base_coord reports rather than at col_base.
                    const int2 c0 = rt_base_coord<T, L>(0, lane);
                    const U *p = &src_ptr[(row_base + c0.x)*row_stride + col_base + c0.y];
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[0], p, vec_bytes);
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
                                         && std::is_same_v<T, U> && (sizeof(U) == 2 || sizeof(U) == 1);
    constexpr int  vec_bytes = E * sizeof(U);

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int lane = laneid();

    const bool wide = vectorizable &&
        ((reinterpret_cast<uintptr_t>(dst_ptr) | (row_stride * sizeof(U))) & (vec_bytes - 1)) == 0;

    // On gfx11 lanes l and l+16 of an operand tile wrote the same bytes with the
    // same values, and the store relied on that mirroring being benign. gfx12
    // has no mirroring: the halves hold disjoint k, so the 32 lanes partition
    // the tile and every byte is written exactly once.
    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        const T* elems = reinterpret_cast<const T*>(&src.tiles[i][j].data[0]);
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            dst_ptr[(row_base + c.x)*row_stride + col_base + c.y] =
                base_types::convertor<U, T>::convert(elems[e]);
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
                    // See the matching note in load(): one access at the lane's
                    // own column, not two at col_base.
                    const int2 c0 = rt_base_coord<T, L>(0, lane);
                    U *p = &dst_ptr[(row_base + c0.x)*row_stride + col_base + c0.y];
                    __builtin_memcpy((void*)p, (const void*)&src.tiles[i][j].data[0], vec_bytes);
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
