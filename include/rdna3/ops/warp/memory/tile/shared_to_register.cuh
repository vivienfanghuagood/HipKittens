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
 * gfx11 lane -> address mapping.  See rt_base_coord() in
 * types/register/rt_base.cuh for the layouts these derive from.  In short, for
 * base tile (i,j) and lane l:
 *
 *   row layout, bf16/half operand : row l%16, cols 0..15          (32 B contiguous)
 *   col layout, bf16/half operand : col l%16, rows 0..15          (16 strided elements)
 *   col layout, f32 accumulator   : col l%16, rows {2e + l/16}    (8 strided elements)
 *   row layout, f32 accumulator   : row l%16, cols {2e + l/16}    (8 strided elements)
 *
 * Only the first is a vectorizable access, and only when the shared tile holds
 * the same 2-byte type.  That case is the A operand of a GEMM and is worth a
 * fast path; the rest fall back to elementwise accesses, which is what the CDNA
 * tree already does for its column-major path.
 *
 * The fast path issues four 8-byte reads rather than one 32-byte read.  The XOR
 * swizzle in st::idx() works on an 8-byte granule (`((addr % repeat) >> 7) << 3`),
 * so a single wide read spanning 32 B would pick up four blocks that the swizzle
 * has permuted relative to each other.  Four separate idx() calls are correct
 * under the existing swizzle, and keep CDNA's conflict-free pattern of 16 lanes
 * hitting 16 distinct 8-byte offsets.  A wider swizzle granule is a Phase 7
 * question, to be settled against the LDS bank-conflict counters.
 *
 * Note that lanes l and l+16 of an operand tile issue *identical* addresses.
 * That is the 2x mirroring WMMA requires, and LDS broadcasts the matching
 * addresses rather than serializing them, so it costs no extra cycles.
 */

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;

    // A whole 16-element row, same width in LDS as in registers: 4x ds_read_b64.
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const int lane = laneid();
    const int l16  = lane & 15;

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int row_base = i*base::tile_size_row;
            const int col_base = j*base::tile_size_col;
            if constexpr (vectorizable) {
                // Plain 8-byte copies, not inline asm: with T == U there is no
                // conversion to generate a v_bfi_b32 (the thing the CDNA path's
                // asm was working around), and letting the compiler own the
                // reads lets it cover all four with one s_waitcnt lgkmcnt.
                #pragma unroll
                for(int b = 0; b < 4; b++) {
                    const U *p = src.idx(const_cast<U*>(src.data), {row_base + l16, col_base + 4*b});
                    // float2 rather than uint64_t: `data` is only 4-byte aligned
                    // as a register array, while the LDS side is 8-byte aligned
                    // (the allocator gives 16 and the swizzle only moves bits 3-6).
                    const float2 v = *reinterpret_cast<const float2*>(p);
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[2*b], &v, sizeof(v));
                }
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const T val = base_types::convertor<T, U>::convert(
                        src[{row_base + c.x, col_base + c.y}]);
                    if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
                    else       dst.tiles[i][j].data[e>>1].x = val;
                }
            }
        }
    }
}


/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;

    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const int lane = laneid();
    const int l16  = lane & 15;

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int row_base = i*base::tile_size_row;
            const int col_base = j*base::tile_size_col;
            if constexpr (vectorizable) {
                // Lanes l and l+16 write the same bytes with the same values --
                // the mirroring invariant makes this benign, and skipping the
                // upper half would cost a branch for no bandwidth (LDS writes to
                // a matching address collapse the same way reads broadcast).
                #pragma unroll
                for(int b = 0; b < 4; b++) {
                    U *p = dst.idx(dst.data, {row_base + l16, col_base + 4*b});
                    float2 v;
                    __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[2*b], sizeof(v));
                    *reinterpret_cast<float2*>(p) = v;
                }
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const T val = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                          : src.tiles[i][j].data[e>>1].x;
                    dst[{row_base + c.x, col_base + c.y}] =
                        base_types::convertor<U, T>::convert(val);
                }
            }
        }
    }
}
}
