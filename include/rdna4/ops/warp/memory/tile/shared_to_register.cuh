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
 * gfx12 lane -> address mapping.  See rt_base_coord() in
 * types/register/rt_base.cuh for the layouts these derive from.  In short, for
 * base tile (i,j), lane l and h = l/16:
 *
 *   row layout, bf16/half operand : row l%16, cols 8h..8h+7        (16 B contiguous)
 *   row layout, fp8 operand       : row l%16, cols 8h..8h+7        ( 8 B contiguous)
 *   col layout, any operand       : col l%16, rows 8h..8h+7        (8 strided elements)
 *   col layout, f32 accumulator   : col l%16, rows {2e + h}        (8 strided elements)
 *   row layout, f32 accumulator   : row l%16, cols {2e + h}        (8 strided elements)
 *
 * Only the row-layout operands are vectorizable accesses, and only when the
 * shared tile holds the same type.  That case is the A operand of a GEMM and is
 * worth a fast path; the rest fall back to elementwise accesses, which is what
 * the CDNA tree already does for its column-major path.  The fast path is one
 * ds_load_b128 for a 2-byte operand and one ds_load_b64 for fp8 -- same eight
 * elements, half the bytes.
 *
 * The difference from RDNA3 is that the fast path is ONE ds_load_b128, not two.
 * There a lane held all 16 of k and the two wave halves were mirrors, so 32
 * lanes issued 16 distinct 32-byte rows; here the halves split k, so each lane
 * wants only its own 16 bytes and the 32 lanes issue 32 distinct addresses.
 * Half the instructions and half the registers for the same tile.
 *
 * That also means the LDS traffic is real rather than broadcast: 32 x 16 B =
 * 512 B, a 4-cycle minimum against RDNA3's 2.  The 16-byte-granule swizzle in
 * st::idx() still reaches that minimum.  Working it through for the bf16 case
 * (swizzle_bytes 128, so subtile_cols 64), a lane's tile offset is
 *
 *     off = 128*(X + r) + 16*(cb/8 + h)      r = l%16, cb = col_base % 64
 *
 * and the swizzle XORs the granule index with (X+r)%8, leaving banks
 * 4*gi..4*gi+3 with gi = (cb/8 + h) ^ ((X+r)%8).  Over the 32 lanes each of the
 * 8 granule positions comes up exactly 4 times, at 4 distinct 128-byte blocks:
 * 4 cycles, which is the floor set by the 512 bytes moved.  !! Hand-derived,
 * not measured -- no gfx12 part has run this. !!
 *
 * fp8 halves every byte count in that derivation: 32 x 8 B = 256 B, and with
 * swizzle_bytes 128 the granule index becomes (cb/8 + h)/2, so pairs of lanes
 * share a 16-byte granule and the 32 lanes cover 16 granules over 4 blocks.
 * Still 4 cycles, now for half the data -- the swizzle is tuned for the 16-byte
 * access and an 8-byte one cannot do better than break even against it.  Not a
 * reason to avoid fp8; a reason not to expect the LDS read to get twice as fast.
 */

/**
 * @brief Whether load()/store() for this pair take the vectorized LDS path.
 *
 * ds_load_b128 for a 2-byte element type, ds_load_b64 for fp8: a lane always
 * holds 8 elements, so the access width follows sizeof directly.
 */
template<ducks::rt::all RT, ducks::st::all ST>
static constexpr bool lds_vectorizable =
    std::is_same_v<typename RT::layout, ducks::rt_layout::row>
    && rt_base<typename base_types::packing<typename RT::dtype>::unpacked_type,
               typename RT::layout>::element_stride == 1
    && std::is_same_v<typename base_types::packing<typename RT::dtype>::unpacked_type,
                      typename ST::dtype>
    && (sizeof(typename ST::dtype) == 2 || sizeof(typename ST::dtype) == 1);

/**
 * @brief How many LDS instructions load()/load_async() issues for this pair.
 *
 * Callers pipelining loads against math need this to size their dscnt wait: see
 * lds_wait() in the memory utilities. Zero means the pair does not take the
 * vectorized path, in which case load_async() is not available.
 */
template<ducks::rt::all RT, ducks::st::all ST>
static constexpr int lds_loads = lds_vectorizable<RT, ST> ? RT::height * RT::width : 0;

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam wait Whether to retire the reads before returning. Pass false to keep
 *              them in flight and wait later with lds_wait<>(), which is how a
 *              caller overlaps the reads for one K-slice with the math of the
 *              previous one. Only the vectorized path can do this -- the
 *              elementwise fallback goes through the compiler, which inserts
 *              its own waits, so it is a static_assert rather than a silent
 *              downgrade.
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<bool wait=true, ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    static_assert(wait || lds_vectorizable<RT, ST>,
                  "load<false> requires the vectorized path; check lds_loads<RT,ST> != 0");

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;

    // A lane's half-row of k, same width in LDS as in registers: one ds_load of
    // E*sizeof(U) bytes, which is 16 for a 2-byte type and 8 for fp8.
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && (sizeof(U) == 2 || sizeof(U) == 1);
    constexpr int  vec_bytes    = E * sizeof(U);

    const int lane = laneid();
    const uint32_t src_ptr = (uint32_t)(uintptr_t)&src.data[0];

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int row_base = i*base::tile_size_row;
            const int col_base = j*base::tile_size_col;
            if constexpr (vectorizable) {
                // Element 0's coordinate is the start of the lane's run, so ask
                // rt_base_coord for it rather than restating the half-split.
                // The alignment this requires holds at both widths: tiles are
                // KITTENS_DEFAULT_ALIGN'd, the byte offset of an 8-element
                // column boundary is a multiple of vec_bytes, and the swizzle
                // only moves bits 4-6, so it never splits a run that started
                // inside a 16-byte granule.  The register side goes through
                // memcpy because `data` is only 4-byte aligned as an array.
                const int2 c0 = rt_base_coord<T, L>(0, lane);
                const uint32_t p = src.idx(src_ptr, {row_base + c0.x, col_base + c0.y});
                if constexpr (vec_bytes == 16) {
                    const float4 v = load_shared_vec4_async(p);
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[0], &v, sizeof(v));
                } else {
                    const float2 v = load_shared_vec_async(p);
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[0], &v, sizeof(v));
                }
            }
            else {
                // `data` is dtype[], and dtype packs 2 elements for bf16/half/
                // float but 4 for fp8, so index the unpacked type directly
                // rather than through .x/.y. Same bytes either way -- the
                // vectorized path above already memcpys over the whole array.
                T* elems = reinterpret_cast<T*>(&dst.tiles[i][j].data[0]);
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    elems[e] = base_types::convertor<T, U>::convert(
                        src[{row_base + c.x, col_base + c.y}]);
                }
            }
        }
    }
    // One wait for the whole tile: the reads above were issued back to back so
    // that their latencies overlap, and the compiler cannot insert this itself
    // for an inline-asm ds_load.
    if constexpr (vectorizable && wait) lds_wait<0>();
}

/**
 * @brief Issue a tile's LDS reads without waiting for them. See load<false>.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load_async(RT &dst, const ST &src) { load<false>(dst, src); }


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
                                         && std::is_same_v<T, U> && (sizeof(U) == 2 || sizeof(U) == 1);
    constexpr int  vec_bytes    = E * sizeof(U);

    const int lane = laneid();
    const uint32_t dst_ptr = (uint32_t)(uintptr_t)&dst.data[0];

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int row_base = i*base::tile_size_row;
            const int col_base = j*base::tile_size_col;
            if constexpr (vectorizable) {
                // Unlike RDNA3 there is no mirroring here, so the 32 lanes
                // write 32 disjoint runs and cover the tile exactly once.
                // Nothing to deduplicate and nothing to collapse.
                const int2 c0 = rt_base_coord<T, L>(0, lane);
                const uint32_t p = dst.idx(dst_ptr, {row_base + c0.x, col_base + c0.y});
                if constexpr (vec_bytes == 16) {
                    float4 v;
                    __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[0], sizeof(v));
                    store_shared_vec4(p, v);
                } else {
                    float2 v;
                    __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[0], sizeof(v));
                    store_shared_vec(p, v);
                }
            }
            else {
                // See the note in load(): index the unpacked type, because
                // dtype holds 4 elements for fp8 and 2 for everything else.
                const T* elems = reinterpret_cast<const T*>(&src.tiles[i][j].data[0]);
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    dst[{row_base + c.x, col_base + c.y}] =
                        base_types::convertor<U, T>::convert(elems[e]);
                }
            }
        }
    }
    // Retire the writes before the caller's barrier, and before it can reuse
    // the VGPRs the asm reads from -- neither is something the compiler tracks
    // across an inline-asm ds_store.
    if constexpr (vectorizable) lds_wait<0>();
}
}
