/**
 * @file
 * @brief General utilities for ThunderKittens.
 */

#pragma once

#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

#include <hip/hip_runtime.h>

#include "base_types.cuh"
#include "../types/register/rt_layout.cuh"
#include "../types/shared/st_layout.cuh"

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif

/**
 * @namespace kittens
 *
 * @brief The main namespace of ThunderKittens.
 */
namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

/**
 * @brief Tile dimension constant.
 */
template<typename T>
concept all_layouts = ducks::rt_layout::all<T> || ducks::st_layout::all<T>;

template<all_layouts layout>
constexpr bool is_col_lt = std::is_same_v<layout, ducks::rt_layout::col> || std::is_same_v<layout, ducks::st_layout::col>;

// Like RDNA3, RDNA4 has exactly one matrix shape -- 16x16x16 WMMA -- for every
// supported type.  gfx12 does add fp8 WMMA, but at the same 16x16x16 shape
// rather than CDNA's 32-wide fp8 variant, so there is still nothing to
// special-case here.
template<typename T, all_layouts layout=ducks::st_layout::row>
constexpr int TILE_ROW_DIM = 16;

template<typename T, all_layouts layout=ducks::st_layout::row>
constexpr int TILE_COL_DIM = 16;

/**
 * @brief Tile num elements constant calculated as TILE_DIM squared.
 */
template<typename T> constexpr int TILE_ELEMENTS{TILE_COL_DIM<T>*TILE_ROW_DIM<T>};

/* ----------  gfx12 WMMA FRAGMENT SHAPE  ---------- */

/*
 * !! NOT HARDWARE-VERIFIED !!  No gfx1200/gfx1201 part was available when this
 * was written.  The shapes below are read off the builtin signatures, which are
 * unambiguous about *size*, plus the RDNA4 ISA guide's description of how the
 * two wave halves split K, which is what fixes the *arrangement*.  Before
 * trusting a number out of this tree, run the PROBE_GFX12_W32 variant of
 * hk-rdna/probe/wmma_layout.hip, which tests exactly this table.
 *
 * Where gfx12 differs from gfx11, and why every constant below is a separate
 * knob rather than the single `replication` RDNA3 needed:
 *
 *   A operand, gfx11 (v16f16, 8 VGPRs): lane l feeds row l%16 and holds all 16
 *       of k.  Lanes 0-15 and 16-31 must carry identical data.
 *   A operand, gfx12 (v8f16, 4 VGPRs):  lane l feeds row l%16 and holds
 *       k = 8*(l/16) + e, e in 0..7.  The halves split K instead of mirroring
 *       it, so an operand costs half the registers it does on RDNA3 -- the
 *       single biggest reason a gfx12 GEMM should outrun a gfx11 one at the
 *       same tile size.  fp8 is the same shape, 8 bytes and 2 VGPRs per lane.
 *   C/D accumulator (v8f32, 8 VGPRs): unchanged from gfx11.  Lane l register r
 *       holds (row 2r + l/16, col l%16).
 *
 * So on gfx12 nothing is replicated, but *both* operands and accumulators are
 * split across the wave halves -- just along different axes and by different
 * amounts.  That is the pair of constants below.
 */
template<typename T> constexpr int WMMA_REPLICATION = 1;  // gfx12 duplicates nothing

/// Distance along a lane's element axis between element `e` and element `e+1`.
template<typename T> constexpr int WMMA_ELEMENT_STRIDE = 1;  // operands: contiguous in k
template<> constexpr int WMMA_ELEMENT_STRIDE<float> = 2;     // accumulator: rows interleave

/// How far along that axis the upper wave half is shifted from the lower one.
template<typename T> constexpr int WMMA_HALF_SHIFT = 8;  // operands: k 0..7 vs 8..15
template<> constexpr int WMMA_HALF_SHIFT<float> = 1;     // accumulator: odd vs even rows
/**
 * @brief Constant representing number of threads in a warp.
 */
constexpr int WARP_THREADS{32};
/**
 * @brief Constant representing number of threads in a warpgroup of four warps.
 */
constexpr int WARPGROUP_THREADS{128};
/**

 * @brief Constant representing number of warps in a warpgroup of four warps.
 */
constexpr int WARPGROUP_WARPS{4};
/**

 * @brief Get the warp ID of the current thread.
 * @return The warp ID.
 */
__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; }
/**
 * @brief Get the warpgroup ID of the current thread.
 * @return The warpgroup ID.
 */
__device__ __forceinline__ int warpgroupid() { return threadIdx.x >> 7; }

/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }

/**
 * @brief Which half of the wave this lane is in (0 for lanes 0-15, 1 for 16-31).
 *
 * gfx12 WMMA splits a wave32 into two 16-lane halves that divide k between them
 * for A/B operands, and that index the two halves of the accumulator's rows.
 * Nearly every RDNA4-specific layout computation needs this and `lane16()`.
 */
__device__ __forceinline__ int wavehalf() { return (threadIdx.x & 0x1f) >> 4; }
/**
 * @brief This lane's index within its 16-lane half.
 */
__device__ __forceinline__ int lane16() { return threadIdx.x & 0xf; }


/**
 * @brief Compute the ceiling division of a by b.
 * @param a The dividend.
 * @param b The divisor.
 * @return The ceiling division of a by b.
 */
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }
  
  /**
   * @brief Transform a workgroup ID to a new workgroup ID based on the chunk size and number of XCDs.
   * @param workgroup_id The original workgroup ID.
   * @param num_workgroups The total number of workgroups.
   * @param num_xcds The number of XCDs.
   * @param chunk_size The chunk size.
   * @return The new workgroup ID.
   */
  __host__ __device__ inline int chiplet_transform_chunked(
      int workgroup_id, 
      int num_workgroups,
      int num_xcds,
      int chunk_size 
  ) {
      // Current XCD
      int xcd = workgroup_id % num_xcds;
  
      // Largest full (NUM_XCDS*CHUNK_SIZE)-aligned block
      int block = num_xcds * chunk_size;
      int limit = (num_workgroups / block) * block;
  
      // If pid beyond the last full block, leave unchanged
      if (workgroup_id > limit) return workgroup_id;
  
      // Local PID (within round-robin assignment)
      int local_pid    = workgroup_id / num_xcds;
      int chunk_idx    = local_pid / chunk_size;
      int pos_in_chunk = local_pid % chunk_size;
  
      // New PID
      return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
  }

// gfx1200/gfx1201 (Navi 44/48, RX 9060 XT / RX 9070 XT) are single monolithic
// dies, so as on RDNA3 there is nothing to swizzle workgroups across.
// NUM_XCDS=1 makes chiplet_transform_chunked() the identity, which keeps kernel
// code that calls it portable between CDNA and RDNA.
//
// 64KB is the per-workgroup LDS limit, unchanged from RDNA3 -- confirmed by
// compiling a 65540-byte __shared__ array for gfx1201 and reading the error.
// The CU count is the top gfx1201 part (RX 9070 XT, 64 CUs); it only feeds
// launch heuristics, and a 9070 (56) or a gfx1200 (32) will simply under-fill.
constexpr int MAX_SHARED_MEMORY = 65536;
constexpr int NUM_XCDS = 1;
constexpr int CUS_PER_XCD = 64;
constexpr int NUM_CUS = CUS_PER_XCD * NUM_XCDS;

/* ----------  CUSTOM TYPES  ---------- */
typedef uint32_t      uint2_t __attribute__((ext_vector_type(2)));

/* ----------  TYPE HELPERS  ---------- */

/**
 * @namespace ducks
 *
 * @brief ThunderKittens' namespace for template metaprogramming..
 * 
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
namespace ducks {

/**
 * @brief A type representing an empty default for a template.
 */
struct default_type {};

// This macro can't be done as a template, so it doesn't really have a location in kittens.
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

}

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads in a warp.
 */
static constexpr uint64_t MASK_ALL = 0xFFFFFFFF;

/**
 * @brief Perform a shuffle down operation on a packed type synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param delta[in] The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 */
 template<typename T>
 __device__ static inline T packed_shfl_down(uint64_t mask, const T &f, int delta) {
     return __shfl_down(f, delta, WARP_THREADS);
 }
template<>
__device__ inline float2 packed_shfl_down<float2>(uint64_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down(f.x, delta, WARP_THREADS);  // Add the width parameter here
    r.y = __shfl_down(f.y, delta, WARP_THREADS);  // And here
    return r;
}
template<>
__device__ inline bf16 packed_shfl_down(uint64_t mask, const bf16 &f, int delta) {
    float r = __shfl_down(base_types::convertor<float, bf16>::convert(f), delta, WARP_THREADS);
    return base_types::convertor<bf16, float>::convert(r);
}

template<>
__device__ inline bf16_2 packed_shfl_down(uint64_t mask, const bf16_2 &f, int delta) {
    float2 r;
    r.x = __shfl_down(base_types::convertor<float, bf16>::convert(f.x), delta, WARP_THREADS);
    r.y = __shfl_down(base_types::convertor<float, bf16>::convert(f.y), delta, WARP_THREADS);
    return base_types::convertor<bf16_2, float2>::convert(r);
}
// Add these specializations after the existing packed_shfl_down implementations:

/**
 * @brief Perform a packed shuffle operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param src[in] The source lane from which to shuffle.
 * @return The result of the shuffle operation.
 */
 
 template<typename T>
 __device__ static inline T packed_shfl(uint64_t mask, const T &f, int src) {
     return __shfl(f, src, WARP_THREADS);
 }
 template<>
 __device__ inline bf16 packed_shfl(uint64_t mask, const bf16 &f, int src) {
     float r = __shfl(base_types::convertor<float, bf16>::convert(f), src, WARP_THREADS);
     return base_types::convertor<bf16, float>::convert(r);
 }
 
 template<>
 __device__ inline bf16_2 packed_shfl(uint64_t mask, const bf16_2 &f, int src) {
     float2 r;
     r.x = __shfl(base_types::convertor<float, bf16>::convert(f.x), src, WARP_THREADS);
     r.y = __shfl(base_types::convertor<float, bf16>::convert(f.y), src, WARP_THREADS);
     return base_types::convertor<bf16_2, float2>::convert(r);
 }
 
 template<>
 __device__ inline half packed_shfl(uint64_t mask, const half &f, int src) {
     float r = __shfl(base_types::convertor<float, half>::convert(f), src, WARP_THREADS);
     return base_types::convertor<half, float>::convert(r);
 }
 
 template<>
 __device__ inline half_2 packed_shfl(uint64_t mask, const half_2 &f, int src) {
     float2 r;
     r.x = __shfl(base_types::convertor<float, half>::convert(f.x), src, WARP_THREADS);
     r.y = __shfl(base_types::convertor<float, half>::convert(f.y), src, WARP_THREADS);
     return base_types::convertor<half_2, float2>::convert(r);
 }
 
 template<>
 __device__ inline float2 packed_shfl<float2>(uint64_t mask, const float2 &f, int src) {
     float2 r;
     r.x = __shfl(f.x, src, WARP_THREADS);
     r.y = __shfl(f.y, src, WARP_THREADS);
     return r;
 }

/* ----------  LANE EXCHANGE  ---------- */

/*
 * Butterfly exchange: every lane swaps with lane `laneid ^ lanemask`.
 *
 * This is the shape RDNA4 reductions want. Along the lane axis the register
 * layouts are replicated across the 16 lanes of a half, so a butterfly with a
 * mask below 16 leaves the answer in every lane of that half and needs no
 * broadcast afterwards.
 *
 * The mask is a template parameter because gfx12 can encode all of these in the
 * instruction rather than computing an address:
 *
 *   mask 1,2,4,8 : DPP16 ROW_XMASK, a free modifier on the consuming VALU op
 *   mask 16      : v_permlanex16_b32, the cross-half exchange
 *
 * __shfl_xor would instead emit ds_bpermute_b32 for all of them, which runs on
 * the LDS pipe and has to be waited on. In a softmax-shaped kernel that is the
 * difference between a reduction that hides under the WMMA stream and one that
 * does not.
 *
 * These read from the exchange partner unconditionally, so like any shuffle they
 * are only meaningful with the whole wave active.
 */
static constexpr int HALF_WAVE_SWAP = 16;

namespace detail {
template<int lanemask>
__device__ static inline int lane_xor_b32(int x) {
    static_assert(lanemask > 0 && lanemask <= 16 && (lanemask & (lanemask-1)) == 0,
                  "lane_xor mask must be a power of two no greater than 16");
    if constexpr (lanemask == HALF_WAVE_SWAP) {
        // Identity lane selects: a plain swap of the two halves.
        return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, true, false);
    }
    else {
        constexpr int DPP_ROW_XMASK0 = 0x160;
        return __builtin_amdgcn_update_dpp(0, x, DPP_ROW_XMASK0 | lanemask, 0xf, 0xf, true);
    }
}

/// Select operands for a permlanex16 that also flips bit B of the row position.
/// Nibble i of (sel_lo, sel_hi) is the source row position for the lane at row
/// position i (i+8 for sel_hi), so both spell out p -> p ^ (1<<B).
template<int B> __device__ static constexpr uint32_t permlanex16_sel(int base) {
    uint32_t s = 0;
    for(int i = 0; i < 8; i++) s |= (uint32_t)((base + i) ^ (1 << B)) << (4*i);
    return s;
}

/// Exchange with lane `laneid ^ ((1<<B) | 16)`: the other half, and within it the
/// neighbour whose row position differs in bit B. permlanex16 already crosses the
/// halves, so the selects only have to supply the ^(1<<B) within the row.
template<int B>
__device__ static inline int lane_half_xor_b32(int x) {
    static_assert(B >= 0 && B < 4, "row position is 4 bits wide");
    return __builtin_amdgcn_permlanex16(x, x, permlanex16_sel<B>(0), permlanex16_sel<B>(8), true, false);
}

/// Apply a 32-bit lane exchange to a value of any 2, 4 or 8 byte type.
///
/// The void* casts are deliberate: bf16_2 and friends have constructors, so the
/// compiler warns about memcpy'ing them even though moving their bits between
/// lanes is exactly what we mean.
template<typename T, typename F>
__device__ static inline T lane_apply_b32(const T &f, F &&op_b32) {
    if constexpr (sizeof(T) == 8) {
        int lo, hi;
        __builtin_memcpy(&lo, (const void*)((const char*)&f),     4);
        __builtin_memcpy(&hi, (const void*)((const char*)&f + 4), 4);
        lo = op_b32(lo);
        hi = op_b32(hi);
        T r;
        __builtin_memcpy((void*)((char*)&r),     &lo, 4);
        __builtin_memcpy((void*)((char*)&r + 4), &hi, 4);
        return r;
    }
    else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2, "lane exchange supports 2, 4 and 8 byte values");
        int x = 0;
        __builtin_memcpy(&x, (const void*)&f, sizeof(T)); // 16-bit types ride in the low half
        x = op_b32(x);
        T r;
        __builtin_memcpy((void*)&r, &x, sizeof(T));
        return r;
    }
}
} // namespace detail

/**
 * @brief Exchange a value with lane `laneid ^ lanemask`.
 *
 * Operates on raw bits, so a packed pair (bf16_2, half_2) moves as one dword
 * rather than being unpacked to two floats the way packed_shfl has to.
 */
template<int lanemask, typename T>
__device__ static inline T lane_xor(const T &f) {
    return detail::lane_apply_b32(f, [](int v) { return detail::lane_xor_b32<lanemask>(v); });
}

/**
 * @brief Exchange a value with lane `laneid ^ ((1<<B) | 16)`.
 *
 * Not a power of two, so it is not a lane_xor. Every gfx12 fragment layout needs
 * one of these: exactly one bit of a value's position along the element axis is
 * carried by the wave-half bit rather than by the element index, so one stage of
 * a transpose has to swap a lane bit with lane bit 4. Which lane bit depends on
 * the shape -- bit 0 for the f32 accumulator (position = 2e+h), bit 3 for a WMMA
 * operand (position = e+8h) -- and B names it. Applied only to the lanes where
 * the two bits disagree. See transpose() in ops/warp/register/tile/conversions.cuh.
 */
template<int B, typename T>
__device__ static inline T lane_half_xor(const T &f) {
    return detail::lane_apply_b32(f, [](int v) { return detail::lane_half_xor_b32<B>(v); });
}

/**
 * @brief Reduce across the 16 lanes of a wave half, leaving the result in all of them.
 *
 * The masks stay below 16, so the two halves never mix and each ends up with its
 * own copy -- which is what the replicated register layouts expect.
 */
template<typename op, typename T>
__device__ static inline T half_wave_butterfly(T v) {
    v = op::template op<T>(v, lane_xor<8>(v));
    v = op::template op<T>(v, lane_xor<4>(v));
    v = op::template op<T>(v, lane_xor<2>(v));
    v = op::template op<T>(v, lane_xor<1>(v));
    return v;
}

using bytes_4  = HIP_vector_type<float, 1>;
using bytes_8  = HIP_vector_type<float, 2>;
using bytes_16 = HIP_vector_type<float, 4>;

/* ----------  SHARED MEMORY UTILS  ---------- */

// namespace ducks {
// namespace sb {
// struct identifier {};
// }
// }

// template<typename Args...>
// struct sb {
//     using identifier = ducks::sb::identifier;
//     Args... args;
// };

// namespace ducks {
// namespace sb {
// template<typename T> concept all = requires {
//     typename T::identifier;
// } && std::is_same_v<T::identifier, identifier>;
// }
// }

// Joyously stolen from https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/include/cute/container/alignment.hpp#L51
#if defined(__CUDACC__)
#define KITTENS_ALIGN_AS(n) __align__(n)
#else
#define KITTENS_ALIGN_AS(n) alignas(n)
#endif

#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
template<int default_alignment=16> 
struct shared_allocator {
    int *ptr;

    private:
        // Recursive template to generate N-dimensional array type
        template<typename A, size_t... dims>
        struct variadic_array;
        template<typename A, size_t first_dim, size_t... rest_dims>
        struct variadic_array<A, first_dim, rest_dims...> {
            using type = typename variadic_array<A, rest_dims...>::type[first_dim];
        };
        template<typename A>
        struct variadic_array<A> {
            using type = A;
        };
        template<typename A, size_t... dims> 
        using variadic_array_t = typename variadic_array<A, dims...>::type;

        template<int alignment>
        __device__ inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                if(p % alignment != 0) {
                    ptr = (int*)(p + (alignment-(p%alignment)));
                }
            }
        }

    public:
        /**
        * @brief Construct a new shared allocator using a pointer to extern shared memory.
        * @param[in] _ptr Pointer to the start of the extern shared memory.
        */
        __device__ shared_allocator(int *_ptr): ptr(_ptr) {}
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<default_alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam alignment An alignment to enforce for this particular object.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<int alignment, typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
};

} // namespace kittens