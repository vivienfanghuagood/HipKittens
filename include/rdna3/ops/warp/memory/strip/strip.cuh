/**
 * @file
 * @brief Register strips to and from global memory, and their per-row
 * vectors through LDS.
 *
 * A chunk of a strip is SPAN = 32*RUN contiguous elements of a row, and lane
 * l's run of it starts RUN*l elements in, so every access here is one lane
 * reading or writing RUN contiguous elements: one global_load_b128 for 8
 * bf16 or 4 fp32, two for 8 fp32. The element type converts on the way in
 * and out, in registers.
 *
 * Coordinates are in elements (coord<>): a strip has no tile unit to count
 * in.
 *
 * The bounded forms go through a buffer resource that ends at the global's
 * last row, as rt's load_bounded does, and also drop the runs that start at
 * or past its last column: those take an offset past any range, which the
 * hardware reads as zero and does not write. A run is never split by the
 * column bound, because a strip is only used on rows whose length is a
 * whole number of runs.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"
#include "../vec/shared_to_register.cuh"

/*
 * HK_STRIP_NT: bit 1 makes the unbounded loads non-temporal (slc dlc), bit 2
 * the unbounded stores (glc slc dlc). Off by default, and what it measures is
 * the Infinity Cache, not the kernel. A 8192x4096 bf16 RMSNorm run back to
 * back on the same tensors (hk-dsl tools/hk-bench/strip_norm.py's shapes):
 *
 *     plain 173.8 us   nt loads 125.7   nt stores 80.5   both 168.1
 *
 * 80.5 us is past what HBM moves; the non-temporal stores stop the output
 * from evicting the 67 MB input from the 96 MB MALL, so every call after the
 * first reads it from there. In a model the next kernel reads the output, and
 * that is the traffic the hint would send to HBM instead.
 */
#ifndef HK_STRIP_NT
#define HK_STRIP_NT 0
#endif

namespace kittens {

namespace detail {

using u32x4 = unsigned __attribute__((ext_vector_type(4)));
using u32x2 = unsigned __attribute__((ext_vector_type(2)));
template<typename V>
__device__ inline V strip_ld(const V *p) {
    if constexpr (HK_STRIP_NT & 1) return __builtin_nontemporal_load(p);
    else return *p;
}
template<typename V>
__device__ inline void strip_st(V *p, const V &v) {
    if constexpr (HK_STRIP_NT & 2) __builtin_nontemporal_store(v, p);
    else *p = v;
}

/// RUN elements at `p`, in the widest accesses the run's bytes allow.
template<typename U, int RUN>
__device__ inline void strip_read(U (&run)[RUN], const U *p) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            const u32x4 v = strip_ld(reinterpret_cast<const u32x4*>(p) + k);
            __builtin_memcpy((void*)&run[k * 16 / sizeof(U)], &v, 16);
        }
    } else if constexpr (BYTES == 8) {
        const u32x2 v = strip_ld(reinterpret_cast<const u32x2*>(p));
        __builtin_memcpy((void*)&run[0], &v, 8);
    } else if constexpr (BYTES == 4) {
        const uint32_t v = *reinterpret_cast<const uint32_t*>(p);
        __builtin_memcpy((void*)&run[0], &v, 4);
    } else {
        static_assert(BYTES == 2, "a run is 2, 4, 8 or a multiple of 16 bytes");
        const uint16_t v = *reinterpret_cast<const uint16_t*>(p);
        __builtin_memcpy((void*)&run[0], &v, 2);
    }
}

template<typename U, int RUN>
__device__ inline void strip_write(U *p, const U (&run)[RUN]) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            u32x4 v;
            __builtin_memcpy(&v, (const void*)&run[k * 16 / sizeof(U)], 16);
            strip_st(reinterpret_cast<u32x4*>(p) + k, v);
        }
    } else if constexpr (BYTES == 8) {
        u32x2 v; __builtin_memcpy(&v, (const void*)&run[0], 8);
        strip_st(reinterpret_cast<u32x2*>(p), v);
    } else if constexpr (BYTES == 4) {
        uint32_t v; __builtin_memcpy(&v, (const void*)&run[0], 4);
        *reinterpret_cast<uint32_t*>(p) = v;
    } else {
        static_assert(BYTES == 2, "a run is 2, 4, 8 or a multiple of 16 bytes");
        uint16_t v; __builtin_memcpy(&v, (const void*)&run[0], 2);
        *reinterpret_cast<uint16_t*>(p) = v;
    }
}

template<typename U, int RUN>
__device__ inline void strip_buffer_read(U (&run)[RUN], i32x4 srsrc, uint32_t voff,
                                         uint32_t soff) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            const __uint128_t v = llvm_amdgcn_raw_buffer_load_b128(srsrc, voff + 16 * k, soff, 0);
            __builtin_memcpy((void*)&run[k * 16 / sizeof(U)], &v, 16);
        }
    } else if constexpr (BYTES == 8) {
        const uint64_t v = llvm_amdgcn_raw_buffer_load_b64(srsrc, voff, soff, 0);
        __builtin_memcpy((void*)&run[0], &v, 8);
    } else if constexpr (BYTES == 4) {
        const uint32_t v = llvm_amdgcn_raw_buffer_load_b32(srsrc, voff, soff, 0);
        __builtin_memcpy((void*)&run[0], &v, 4);
    } else {
        static_assert(BYTES == 2, "a run is 2, 4, 8 or a multiple of 16 bytes");
        const uint16_t v = llvm_amdgcn_raw_buffer_load_b16(srsrc, voff, soff, 0);
        __builtin_memcpy((void*)&run[0], &v, 2);
    }
}

template<typename U, int RUN>
__device__ inline void strip_buffer_write(i32x4 srsrc, uint32_t voff, uint32_t soff,
                                          const U (&run)[RUN]) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            __uint128_t v;
            __builtin_memcpy(&v, (const void*)&run[k * 16 / sizeof(U)], 16);
            llvm_amdgcn_raw_buffer_store_b128(v, srsrc, voff + 16 * k, soff, 0);
        }
    } else if constexpr (BYTES == 8) {
        uint64_t v; __builtin_memcpy(&v, (const void*)&run[0], 8);
        llvm_amdgcn_raw_buffer_store_b64(v, srsrc, voff, soff, 0);
    } else if constexpr (BYTES == 4) {
        uint32_t v; __builtin_memcpy(&v, (const void*)&run[0], 4);
        llvm_amdgcn_raw_buffer_store_b32(v, srsrc, voff, soff, 0);
    } else {
        static_assert(BYTES == 2, "a run is 2, 4, 8 or a multiple of 16 bytes");
        uint16_t v; __builtin_memcpy(&v, (const void*)&run[0], 2);
        llvm_amdgcn_raw_buffer_store_b16(v, srsrc, voff, soff, 0);
    }
}

/// Past any buffer range this library builds: reads zero, writes nothing.
static constexpr uint32_t STRIP_OOB = 0x80000000u;

/// The descriptor over a strip's rows that are in bounds, from column 0.
template<int ROWS, typename GL>
__device__ inline i32x4 strip_srsrc(const GL &g, const coord<> &idx) {
    using U = typename GL::dtype;
    coord<> row0 = idx;
    row0.c = 0;
    const int left = int(g.rows()) - idx.r;
    const int valid = left <= 0 ? 0 : (left < ROWS ? left : ROWS);
    return make_srsrc(&g[row0], uint32_t(valid) * uint32_t(g.row_stride()) * uint32_t(sizeof(U)));
}

} // namespace detail

/**
 * @brief Global -> strip. Row r of the strip is row idx.r + r of `src`,
 * from column idx.c.
 */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const U *base = &src[idx] + laneid() * RS::run;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            detail::strip_read(run, base + r * stride + j * RS::span);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/**
 * @brief Strip -> global, the inverse of load().
 */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void store(const GL &dst, const RS &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    U *base = &dst[idx] + laneid() * RS::run;
    const int stride = dst.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                run[e] = base_types::store_convertor<U, T>::convert(src.data[r][j][e]);
            detail::strip_write(base + r * stride + j * RS::span, run);
        }
    }
}

/**
 * @brief load(), predicated on the global's extent: rows past its last row
 * and runs at or past its last column read as zero and touch no memory.
 */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load_bounded(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const i32x4 srsrc = detail::strip_srsrc<RS::rows>(src, idx);
    const uint32_t soff = __builtin_amdgcn_readfirstlane(uint32_t(idx.c) * sizeof(U));
    const int lane0 = laneid() * RS::run;
    const int cols_left = int(src.cols()) - idx.c;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            const int c = j * RS::span + lane0;
            const uint32_t voff = c < cols_left ? uint32_t(r * stride + c) * sizeof(U)
                                                : detail::STRIP_OOB;
            U run[RS::run];
            detail::strip_buffer_read(run, srsrc, voff, soff);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/**
 * @brief store(), predicated as load_bounded(): rows past the global's last
 * row and runs at or past its last column are not written.
 */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void store_bounded(const GL &dst, const RS &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const i32x4 srsrc = detail::strip_srsrc<RS::rows>(dst, idx);
    const uint32_t soff = __builtin_amdgcn_readfirstlane(uint32_t(idx.c) * sizeof(U));
    const int lane0 = laneid() * RS::run;
    const int cols_left = int(dst.cols()) - idx.c;
    const int stride = dst.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            const int c = j * RS::span + lane0;
            const uint32_t voff = c < cols_left ? uint32_t(r * stride + c) * sizeof(U)
                                                : detail::STRIP_OOB;
            U run[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                run[e] = base_types::store_convertor<U, T>::convert(src.data[r][j][e]);
            detail::strip_buffer_write(srsrc, voff, soff, run);
        }
    }
}

/**
 * @brief Global -> one value per row: entry r is element (idx.r + r, idx.c).
 * Every lane reads the same addresses.
 */
template<ducks::rs::cv CV, ducks::gl::all GL>
__device__ inline static void load(CV &dst, const GL &src, const coord<> &idx) {
    using U = typename GL::dtype;
    const U *base = &src[idx];
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = base_types::convertor<typename CV::T, U>::convert(base[r * stride]);
}

/**
 * @brief One entry of a per-row vector -> one element, from lane 0. Entry 0;
 * the caller's precondition is that the vector is uniform.
 */
template<ducks::gl::all GL, ducks::rs::cv CV>
__device__ inline static void store_scalar(const GL &dst, const CV &src, const coord<> &idx) {
    using U = typename GL::dtype;
    if (laneid() != 0) return;
    *(U*)&dst[idx] = base_types::store_convertor<U, typename CV::T>::convert(src.data[0][0]);
}

/**
 * @brief A per-row vector -> the first CV::length entries of an LDS vector.
 *
 * Lane 0 writes: every lane holds the same values, and 32 lanes writing one
 * address is a conflict where 32 reading it is a broadcast.
 */
template<ducks::sv::all SV, ducks::rs::cv CV>
__device__ inline static void store(SV &dst, const CV &src) {
    static_assert(SV::length >= CV::length, "the LDS vector holds every row's value");
    using U = typename SV::dtype;
    if (laneid() == 0) {
        #pragma unroll
        for (int r = 0; r < CV::length; r++)
            detail::lds_write(&dst.data[r],
                              base_types::convertor<U, typename CV::T>::convert(src.data[r][0]));
    }
}

/**
 * @brief The first CV::length entries of an LDS vector -> a per-row vector,
 * in every lane.
 */
template<ducks::rs::cv CV, ducks::sv::all SV>
__device__ inline static void load(CV &dst, const SV &src) {
    static_assert(SV::length >= CV::length, "the LDS vector holds every row's value");
    using U = typename SV::dtype;
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = base_types::convertor<typename CV::T, U>::convert(
            detail::lds_read(&src.data[r]));
}

} // namespace kittens
