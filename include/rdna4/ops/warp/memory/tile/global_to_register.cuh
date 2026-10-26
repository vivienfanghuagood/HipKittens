/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

/*
 * On gfx11 a 16-bit row tile held each row twice, in lanes l and l + 16, and
 * HK_RT_HALF_SPLIT chose whether a bounded access moved each run once or
 * twice. Nothing on gfx12 is mirrored -- every lane already moves only its own
 * bytes -- so the knob is accepted and has no effect here.
 */
#ifndef HK_RT_HALF_SPLIT
#define HK_RT_HALF_SPLIT 0
#endif

namespace kittens {

namespace detail {

// __builtin_nontemporal_store takes only native scalar/vector types, and the
// tile element types here are classes (__hip_bfloat16, HIP_vector_type). These
// are the same-width native types to memcpy through; nothing is converted, the
// bits go out unchanged.
template<int Bytes> struct nt_word;
template<> struct nt_word<1> { using type = unsigned char;  };
template<> struct nt_word<2> { using type = unsigned short; };
template<> struct nt_word<4> { using type = unsigned int;   };
template<int Bytes> using nt_word_t = typename nt_word<Bytes>::type;
using nt_vec2 = unsigned int __attribute__((ext_vector_type(2)));
using nt_vec4 = unsigned int __attribute__((ext_vector_type(4)));

/*
 * A lane's run.
 *
 * Every gfx12 fragment puts a lane's E = 8 elements in one line of the base
 * tile (row l%16 of a row-layout tile), contiguous, the upper wave half 8
 * further along (rt_base_coord). So in a row-layout tile the lane's elements
 * are E consecutive elements of one row in memory -- 8 bytes of int8 or fp8, 16
 * of a 2-byte type, 32 of fp32 -- whatever the tile's type and whatever the
 * memory's. That is the whole difference from the gfx11 tree, where only the
 * 2-byte operand was contiguous (and mirrored) and the fp32 accumulator held
 * every other column, so a narrowing store had to assemble its runs across the
 * halves (pack_interleaved_run) and a widening load read twice what it kept.
 * Here a load or store of any type pair is one run a lane, converted in
 * registers: no exchange, and no byte moved twice.
 *
 * The run is moved in the widest accesses that tile it -- one b64 for 8
 * bytes, one b128 for 16, two for 32.
 */
template<typename U, int E> struct run {
    static constexpr int bytes = E * int(sizeof(U));
    static constexpr int chunk = bytes < 16 ? bytes : 16;   // bytes an access
    static constexpr int chunks = bytes / chunk;
    static_assert(chunk == 8 || chunk == 16, "a run is 8, 16 or 32 bytes");
    static_assert(bytes % chunk == 0, "a run is whole accesses");
};

template<typename U, int E>
__device__ inline void load_run(const U *p, U (&out)[E]) {
    using R = run<U, E>;
    #pragma unroll
    for(int k = 0; k < R::chunks; k++) {
        if constexpr (R::chunk == 16) {
            const nt_vec4 v = reinterpret_cast<const nt_vec4 *>(p)[k];
            __builtin_memcpy((char *)out + 16*k, &v, 16);
        } else {
            const nt_vec2 v = reinterpret_cast<const nt_vec2 *>(p)[k];
            __builtin_memcpy((char *)out + 8*k, &v, 8);
        }
    }
}

template<typename U, int E, bool NT = false>
__device__ inline void store_run(U *p, const U (&in)[E]) {
    using R = run<U, E>;
    #pragma unroll
    for(int k = 0; k < R::chunks; k++) {
        if constexpr (R::chunk == 16) {
            nt_vec4 v; __builtin_memcpy(&v, (const char *)in + 16*k, 16);
            if constexpr (NT) __builtin_nontemporal_store(v, reinterpret_cast<nt_vec4 *>(p) + k);
            else reinterpret_cast<nt_vec4 *>(p)[k] = v;
        } else {
            nt_vec2 v; __builtin_memcpy(&v, (const char *)in + 8*k, 8);
            if constexpr (NT) __builtin_nontemporal_store(v, reinterpret_cast<nt_vec2 *>(p) + k);
            else reinterpret_cast<nt_vec2 *>(p)[k] = v;
        }
    }
}

template<typename U, int E>
__device__ inline void buffer_load_run(i32x4 srsrc, uint32_t off, uint32_t soff, U (&out)[E]) {
    using R = run<U, E>;
    #pragma unroll
    for(int k = 0; k < R::chunks; k++) {
        if constexpr (R::chunk == 16) {
            const __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, off + 16*k, soff, 0);
            __builtin_memcpy((char *)out + 16*k, &raw, 16);
        } else {
            const uint64_t raw = llvm_amdgcn_raw_buffer_load_b64(srsrc, off + 8*k, soff, 0);
            __builtin_memcpy((char *)out + 8*k, &raw, 8);
        }
    }
}

template<typename U, int E>
__device__ inline void buffer_store_run(i32x4 srsrc, uint32_t off, const U (&in)[E]) {
    using R = run<U, E>;
    #pragma unroll
    for(int k = 0; k < R::chunks; k++) {
        if constexpr (R::chunk == 16) {
            __uint128_t raw; __builtin_memcpy(&raw, (const char *)in + 16*k, 16);
            llvm_amdgcn_raw_buffer_store_b128(raw, srsrc, off + 16*k, 0, 0);
        } else {
            uint64_t raw; __builtin_memcpy(&raw, (const char *)in + 8*k, 8);
            llvm_amdgcn_raw_buffer_store_b64(raw, srsrc, off + 8*k, 0, 0);
        }
    }
}

/// Whether a row-layout tile of T moves to and from memory of U one run a lane.
template<typename T, typename U, typename L>
constexpr bool run_path = std::is_same_v<L, ducks::rt_layout::row>
                          && rt_base<T, L>::element_stride == 1
                          && (rt_base<T, L>::elements_per_thread * sizeof(U)) % 8 == 0;

} // namespace detail

/*
 * Same gfx12 lane mapping as the shared-memory path (rt_base_coord in
 * types/register/rt_base.cuh), minus the swizzle: global tiles are plain
 * row-major with `row_stride` between rows.
 *
 * A row-layout tile moves one run a lane (detail::run above); a column-layout
 * one is strided per lane and goes elementwise, which is what the CDNA tree
 * already does for its column path.
 *
 * Unlike LDS, global rows are only as aligned as `row_stride` makes them: the
 * allocation base is 256 B from hipMalloc, but row r starts at r*row_stride
 * elements, so a stride of e.g. 8200 bf16 leaves every odd row 16-byte
 * misaligned. The run path is therefore selected at runtime (a warp-uniform
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

    constexpr int  E    = base::elements_per_thread;
    constexpr bool runs = detail::run_path<T, U, L>;

    const U *src_ptr = (const U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    const int lane = laneid();

    // Alignment of every run this warp will touch: a run starts a multiple of
    // its own size into its row, so the row starts are what can break it.
    // Uniform across the warp, so the branch below is a scalar branch.
    constexpr int A = runs ? detail::run<U, E>::chunk : 1;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(src_ptr) | (row_stride * sizeof(U))) & (A - 1)) == 0;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T val = base_types::convertor<T, U>::convert(
                src_ptr[(row_base + c.x)*row_stride + col_base + c.y]);
            reinterpret_cast<T *>(&dst.tiles[i][j].data[0])[e] = val;   // any packing (int8_4: four a word)
        }
    };

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int row_base = i*dst.tile_size_row;
            const int col_base = j*dst.tile_size_col;
            if constexpr (runs) {
                if (aligned) {
                    const int2 c0 = rt_base_coord<T, L>(0, lane);
                    U r[E];
                    detail::load_run<U, E>(&src_ptr[(row_base + c0.x)*row_stride + col_base + c0.y], r);
                    #pragma unroll
                    for(int e = 0; e < E; e++)
                        reinterpret_cast<T *>(&dst.tiles[i][j].data[0])[e] =
                            base_types::convertor<T, U>::convert(r[e]);
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
 * @brief Store a register tile to a raw global pointer.
 *
 * The addressing half of store() -- turning a gl and a coord into a base
 * pointer and a row stride -- split out so a caller that already has the
 * pointer can skip it. A distributed epilogue is the case that needs this: the
 * destination may be a peer GPU's buffer, which is an ordinary global pointer
 * but is not describable as a gl, because gl has no default constructor and a
 * host-only one, so a kernel argument cannot hold an array of them.
 *
 * Everything below the pointer is identical to the gl version -- which is the
 * point, the two share this body rather than duplicating the vectorization
 * decision.
 *
 * @tparam RT The register tile type.
 * @tparam U The element type of the destination.
 * @tparam NT Bypass the cache hierarchy on the way out (see store_at_nt).
 * @param[out] dst_ptr Where the tile's (0,0) element goes.
 * @param[in] row_stride Elements between consecutive rows at the destination.
 * @param[in] src The source register tile.
 */
template<ducks::rt::all RT, typename U, bool NT = false>
__device__ inline static void store_at(U *dst_ptr, int row_stride, const RT &src) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr int  E    = base::elements_per_thread;
    constexpr bool runs = detail::run_path<T, U, L>;

    const int lane = laneid();

    constexpr int A = runs ? detail::run<U, E>::chunk : 1;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(dst_ptr) | (row_stride * sizeof(U))) & (A - 1)) == 0;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T val = reinterpret_cast<const T *>(&src.tiles[i][j].data[0])[e];
            U *p = &dst_ptr[(row_base + c.x)*row_stride + col_base + c.y];
            const U cv = base_types::store_convertor<U, T>::convert(val);
            // __builtin_nontemporal_store only accepts native scalar/vector
            // types, and U is a class (__hip_bfloat16), so go through an
            // integer of the same width. Pure bit shuffling, no conversion.
            if constexpr (NT) {
                using raw = detail::nt_word_t<sizeof(U)>;
                raw w; __builtin_memcpy(&w, &cv, sizeof(U));
                __builtin_nontemporal_store(w, reinterpret_cast<raw*>(p));
            }
            else *p = cv;
        }
    };

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int row_base = i*src.tile_size_row;
            const int col_base = j*src.tile_size_col;
            if constexpr (runs) {
                if (aligned) {
                    const int2 c0 = rt_base_coord<T, L>(0, lane);
                    U r[E];
                    #pragma unroll
                    for(int e = 0; e < E; e++)
                        r[e] = base_types::store_convertor<U, T>::convert(
                            reinterpret_cast<const T *>(&src.tiles[i][j].data[0])[e]);
                    detail::store_run<U, E, NT>(&dst_ptr[(row_base + c0.x)*row_stride + col_base + c0.y], r);
                }
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}

/**
 * @brief store_at(), but the stores bypass the cache hierarchy.
 *
 * For a destination that is another GPU's memory, written with the
 * nontemporal hint. This is the gfx11 tree's peer-store path carried over; on
 * gfx11 a peer's memory was cached in the *local* L2 and nothing else wrote it
 * back. Whether gfx12 needs it, and whether its nontemporal hint (TH_NT on
 * gfx12, not gfx11's glc slc dlc) has the same effect, is not measured: nothing
 * here runs across GPUs on gfx12 yet. For local memory it is a pure loss.
 *
 * @tparam RT The register tile type.
 * @tparam U The element type of the destination.
 */
template<ducks::rt::all RT, typename U>
__device__ inline static void store_at_nt(U *dst_ptr, int row_stride, const RT &src) {
    store_at<RT, U, true>(dst_ptr, row_stride, src);
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
    using U = typename GL::dtype;
    store_at<RT, U>((U*)&dst[(idx.template unit_coord<axis, 3>())],
                    dst.template stride<axis>(), src);
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

/*
 * Row-predicated load and store.
 *
 * Separate functions rather than a flag on load()/store_at(), and on buffer
 * instructions rather than on those functions' pointers. Both choices were
 * measured (on gfx1100). A predicate threaded through the pointer path -- even
 * one that is a compile-time constant on the unbounded side -- changed how LLVM
 * addresses every store (32 base+offset stores became 32 separate address
 * pairs, 36 -> 91 VGPRs on an elementwise op) and spilled the GEMM. A buffer
 * resource whose range ends at the tensor's last row needs no predicate in the
 * code at all: the hardware returns zero for a load past the range and drops a
 * store past it, with no branch, no masked lane, and the instruction count of a
 * whole tile. The bytes past the row never move.
 *
 * The voffset is 32-bit, so the tile's rows from its first must span less
 * than 4 GB -- true of any tensor this library is handed.
 */
namespace detail {

template<int axis, int ROWS, typename GL, typename COORD>
__device__ inline i32x4 row_bounded_srsrc(const GL &g, const COORD &unit, const void *base,
                                          int limit = ROWS) {
    // `limit` clips further, for a caller whose rows end before the
    // tensor's do: a query tile that runs into the next request's tokens.
    const int in = rows_in_bounds<axis, ROWS>(g, unit);
    const int valid = limit < 0 ? 0 : (limit < in ? limit : in);
    const uint32_t bytes = uint32_t(valid) * uint32_t(g.template stride<axis>()) *
                           uint32_t(sizeof(typename GL::dtype));
    return make_srsrc(base, bytes);
}

template<typename U>
__device__ inline U buffer_load_elem(i32x4 srsrc, uint32_t off, uint32_t soff) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4, "16- or 32-bit elements");
    U v;
    if constexpr (sizeof(U) == 2) {
        const uint16_t raw = llvm_amdgcn_raw_buffer_load_b16(srsrc, off, soff, 0);
        __builtin_memcpy(&v, &raw, 2);
    } else {
        const uint32_t raw = llvm_amdgcn_raw_buffer_load_b32(srsrc, off, soff, 0);
        __builtin_memcpy(&v, &raw, 4);
    }
    return v;
}

template<typename U>
__device__ inline void buffer_store_elem(i32x4 srsrc, uint32_t off, U v) {
    static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4, "8-, 16- or 32-bit elements");
    if constexpr (sizeof(U) == 1) {
        uint8_t raw; __builtin_memcpy(&raw, &v, 1);
        llvm_amdgcn_raw_buffer_store_b8(raw, srsrc, off, 0, 0);
    } else if constexpr (sizeof(U) == 2) {
        uint16_t raw; __builtin_memcpy(&raw, &v, 2);
        llvm_amdgcn_raw_buffer_store_b16(raw, srsrc, off, 0, 0);
    } else {
        uint32_t raw; __builtin_memcpy(&raw, &v, 4);
        llvm_amdgcn_raw_buffer_store_b32(raw, srsrc, off, 0, 0);
    }
}

} // namespace detail

/**
 * @brief load(), predicated on the tensor's rows: rows of the tile past the
 * last row of `src` read as zero, and their memory is never touched.
 *
 * Same lane mapping and the same run a lane as load(); a column-layout tile
 * goes element by element.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_bounded(RT &dst, const GL &src, const COORD &idx,
                                           int limit = RT::rows) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr int  E    = base::elements_per_thread;
    constexpr bool runs = detail::run_path<T, U, L>;

    // The buffer starts at the tile's first row, column 0, and the column
    // offset rides in soffset. In a K loop that makes the descriptor and every
    // lane's voffset loop-invariant and the one moving part a scalar: with the
    // column folded into the base instead, LLVM hoisted eight voffsets out of
    // the decode GEMM's loop and the kernel lost a step of occupancy (gfx1100).
    const auto unit = idx.template unit_coord<axis, 3>();
    auto row0 = unit;
    row0.c = 0;
    const U *row_ptr = (const U*)&src[row0];
    // An SGPR operand, and uniform by the same argument as the descriptor.
    const uint32_t col0 = __builtin_amdgcn_readfirstlane(uint32_t(unit.c) * sizeof(U));
    const int row_stride = src.template stride<axis>();
    const i32x4 srsrc = detail::row_bounded_srsrc<axis, RT::rows>(src, unit, row_ptr, limit);
    const int lane = laneid();

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int row_base = i*dst.tile_size_row;
            const int col_base = j*dst.tile_size_col;
            if constexpr (runs) {
                // No alignment branch, unlike load(): RDNA runs with unaligned
                // buffer access enabled, and a fallback here would be E
                // separate buffer loads whose registers the allocator reserves
                // whether the branch runs or not (+10 VGPRs on the gfx1100
                // decode GEMM).
                const int2 c0 = rt_base_coord<T, L>(0, lane);
                U r[E];
                detail::buffer_load_run<U, E>(
                    srsrc, ((row_base + c0.x)*row_stride + col_base + c0.y) * sizeof(U), col0, r);
                #pragma unroll
                for(int e = 0; e < E; e++)
                    reinterpret_cast<T *>(&dst.tiles[i][j].data[0])[e] =
                        base_types::convertor<T, U>::convert(r[e]);
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const uint32_t off = ((row_base + c.x)*row_stride + col_base + c.y) * sizeof(U);
                    reinterpret_cast<T *>(&dst.tiles[i][j].data[0])[e] =
                        base_types::convertor<T, U>::convert(detail::buffer_load_elem<U>(srsrc, off, col0));
                }
            }
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_bounded(RT &dst, const GL &src, const COORD &idx,
                                           int limit = RT::rows) {
    load_bounded<2, RT, GL, COORD>(dst, src, idx, limit);
}

/**
 * @brief store(), predicated on the tensor's rows: rows of the tile past the
 * last row of `dst` are not written.
 *
 * A row-layout tile writes its runs through the buffer when the destination's
 * rows are aligned to them; the rest go element by element.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_bounded(const GL &dst, const RT &src, const COORD &idx,
                                            int limit = RT::rows) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr int  E    = base::elements_per_thread;
    constexpr bool runs = detail::run_path<T, U, L>;

    const auto unit = idx.template unit_coord<axis, 3>();
    U *dst_ptr = (U*)&dst[unit];
    const int row_stride = dst.template stride<axis>();
    const i32x4 srsrc = detail::row_bounded_srsrc<axis, RT::rows>(dst, unit, dst_ptr, limit);
    const int lane = laneid();
    constexpr int A = runs ? detail::run<U, E>::chunk : 1;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(dst_ptr) | (row_stride * sizeof(U))) & (A - 1)) == 0;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T v = reinterpret_cast<const T *>(&src.tiles[i][j].data[0])[e];
            detail::buffer_store_elem<U>(
                srsrc, ((row_base + c.x)*row_stride + col_base + c.y) * sizeof(U),
                base_types::store_convertor<U, T>::convert(v));
        }
    };

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int row_base = i*src.tile_size_row;
            const int col_base = j*src.tile_size_col;
            if constexpr (runs) {
                if (aligned) {
                    const int2 c0 = rt_base_coord<T, L>(0, lane);
                    U r[E];
                    #pragma unroll
                    for(int e = 0; e < E; e++)
                        r[e] = base_types::store_convertor<U, T>::convert(
                            reinterpret_cast<const T *>(&src.tiles[i][j].data[0])[e]);
                    detail::buffer_store_run<U, E>(
                        srsrc, ((row_base + c0.x)*row_stride + col_base + c0.y) * sizeof(U), r);
                }
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_bounded(const GL &dst, const RT &src, const COORD &idx,
                                            int limit = RT::rows) {
    store_bounded<2, RT, GL, COORD>(dst, src, idx, limit);
}

} // namespace kittens
