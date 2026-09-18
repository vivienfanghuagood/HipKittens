/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
 */

/*
 * Row-sliced version of the warp-level path in
 * ops/warp/memory/tile/shared_to_register.cuh: warp w owns rows
 * [w*RT::height, (w+1)*RT::height) of subtiles and runs exactly the same lane
 * mapping within them. See that file for the gfx12 layouts and for why the
 * vectorized case issues two 16-byte reads.
 *
 * Note that unlike the CDNA version this goes through src.idx() / operator[]
 * rather than indexing src.data[] directly, so the shared tile's XOR swizzle is
 * applied. Raw indexing is only correct for an unswizzled layout.
 */

/**
 * @brief Collaboratively load data from a shared tile into register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && (sizeof(U) == 2 || sizeof(U) == 1);
    /// A lane's whole fragment in one access: 16 bytes for a 2-byte type, 8 for fp8.
    constexpr int  vec_bytes    = E * sizeof(U);

    const int lane = ::kittens::laneid();
    const int warp_row_offset = warpid() * warp_height;
    const uint32_t src_ptr = (uint32_t)(uintptr_t)&src.data[0];

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int row_base = (warp_row_offset + i) * base::tile_size_row;
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            const int col_base = j * base::tile_size_col;
            if constexpr (vectorizable) {
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
                // dtype packs 4 elements for fp8 and 2 otherwise, so index the
                // unpacked type rather than through .x/.y.
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
    if constexpr (vectorizable) lds_wait<0>();
}


/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void store(ST &dst, const RT &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && (sizeof(U) == 2 || sizeof(U) == 1);
    /// A lane's whole fragment in one access: 16 bytes for a 2-byte type, 8 for fp8.
    constexpr int  vec_bytes    = E * sizeof(U);

    const int lane = ::kittens::laneid();
    const int warp_row_offset = warpid() * warp_height;
    const uint32_t dst_ptr = (uint32_t)(uintptr_t)&dst.data[0];

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        const int row_base = (warp_row_offset + i) * base::tile_size_row;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col_base = j * base::tile_size_col;
            if constexpr (vectorizable) {
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
                // See the note in load(): index the unpacked type.
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
    if constexpr (vectorizable) lds_wait<0>();
}
