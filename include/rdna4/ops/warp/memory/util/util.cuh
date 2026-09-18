/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/hip_ldg.h>

namespace kittens {

enum class coherency {
    cache_all = 0,
    cache_global = 1,
    cache_stream = 2,
    non_temporal = 3
};

/* ----------   Wait counters  ---------- */

/*
 * gfx12 split S_WAITCNT into one instruction per counter.  This is the single
 * most mechanical difference between this tree and include/rdna3, and it is the
 * kind that fails silently, so it is worth writing down what was checked.
 *
 *   gfx11                      gfx12                        encoding
 *   s_waitcnt lgkmcnt(N)  ->   s_wait_dscnt N               BFC6_00NN
 *   s_waitcnt vmcnt(N)    ->   s_wait_loadcnt N             BFC0_00NN   (loads)
 *                         ->   s_wait_storecnt N            BFC1_00NN   (stores)
 *                              s_wait_kmcnt N               BFC7_00NN   (scalar)
 *
 * The trap: `s_waitcnt lgkmcnt(0)` still *assembles* for gfx1201 -- it encodes
 * to BF89FC07, the legacy opcode -- so the rdna3 spelling compiles here without
 * a diagnostic.  But LLVM's own memory legalizer never emits it for gfx12; a
 * plain __shared__ load compiled for gfx1201 comes out as `ds_load_b32` followed
 * by `s_wait_dscnt 0x0` (BFC60000).  Mixing the two would mean hand-written asm
 * waiting on a counter the compiler-generated code no longer maintains.
 * Verified by assembling both forms for gfx1201 and disassembling them.
 *
 * The LDS and global mnemonics below were renamed too (ds_read_b128 ->
 * ds_load_b128, global_load_dwordx4 -> global_load_b128).  Unlike the waits,
 * those really are pure aliases -- LLVM's assembler accepts the gfx9/11 spelling
 * and encodes the gfx12 instruction -- but the tree uses the native names so the
 * asm reads the same as the disassembly.
 */

/// Wait until at most `N` vector *loads* are outstanding (gfx11's vmcnt).
template<int N=0> __device__ inline void vmem_load_wait() {
    static_assert(N >= 0 && N <= 63, "loadcnt is 6 bits");
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

/// Wait until at most `N` vector *stores* are outstanding. Separate from the
/// load counter on gfx12; on gfx11 both lived in vmcnt/vscnt.
template<int N=0> __device__ inline void vmem_store_wait() {
    static_assert(N >= 0 && N <= 63, "storecnt is 6 bits");
    asm volatile("s_wait_storecnt %0" :: "i"(N) : "memory");
}

/* ----------   Shared memory utilities  ---------- */
__device__ inline float2 load_shared_vec(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_load_b64 %0, %1\n"
        "s_wait_dscnt 0\n"
        : "=v"(result)              // Output: store result in float2
        : "v"(lds_off)              // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

__device__ inline void store_shared_vec(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_store_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

__device__ inline float2 load_global_vec2(const float2* gptr) {
    float2 v;
    // Use global_load_b64 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_b64 %0, %1, off\n"
        "s_wait_loadcnt 0\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline float4 load_global_vec4(const float4* gptr) {
    float4 v;
    // Use global_load_b128 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_b128 %0, %1, off\n"
        "s_wait_loadcnt 0\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}

/*
 * gfx12 buffer resource descriptor (V#).
 *
 * Word 3 is *not* the gfx9 config word the CDNA tree uses. Measured on gfx1100
 * (see tools/rdna-probes/srsrc3.hip), a descriptor is only usable if both hold:
 *
 *   OOB_SELECT (bits 29:28) == 3.  At any smaller value NumRecords is counted in
 *     units of STRIDE, and STRIDE is 0 for a raw buffer -- so *every* access is
 *     out of bounds. Loads return zero and stores are dropped, silently, with no
 *     fault and no warning. Both gfx9 words (0x00020000 from global_to_register
 *     and 0x00110000 from the old make_srsrc) fail exactly this way on gfx1100.
 *   FORMAT (bits 18:12) != 0.  Non-format buffer ops ignore what the format is,
 *     but 0 is BUF_FMT_INVALID and invalidates the descriptor: 0x30000000 reads
 *     back zeros while 0x30004000 works.
 *
 * The gfx1100 measurement is the rdna3 tree's; the open question for this tree
 * was whether gfx12 reorganized word 3.  It did not, and the evidence is AMD's
 * own, not inferred: Composable Kernel (shipped in ROCm, see ck/ck.hpp and
 * ck_tile/core/config.hpp) selects CK_BUFFER_RESOURCE_3RD_DWORD per arch and
 * puts gfx11 and gfx12 on the *same* line --
 *
 *     gfx9   0x00020000
 *     gfx103 0x31014000
 *     gfx11 || gfx12   0x31004000
 *
 * Corroborating from the compiler side, LLVM has UfmtGFX10 and UfmtGFX11 unified
 * format tables but no UfmtGFX12 (gfx12 reuses gfx11's), and
 * SIInstrInfo::getDefaultRsrcDataFormat() places FORMAT, RESOURCE_LEVEL and
 * OOB_SELECT at the same bits for everything >= GFX10 with no GFX12 branch.
 *
 * So the value below is CK's gfx12 word rather than the rdna3 tree's
 * 0x31014000. They differ only in FORMAT (4 vs 20), which raw buffer ops ignore
 * -- either would do, and taking the one AMD ships for this arch is free.
 */
static constexpr uint32_t GFX12_BUFFER_CONFIG = 0x31004000u;

/**
 * @brief Build a raw buffer descriptor over `range_bytes` starting at `ptr`.
 *
 * Accesses at or past `range_bytes` are clamped by the hardware: loads read 0,
 * stores are dropped. That bounds check is the whole reason to prefer buffer
 * ops over plain pointers here -- it makes ragged edge tiles free.
 *
 * Unlike the CDNA version this takes no row stride. The stride field only means
 * anything for *structured* buffers, and turning swizzling on would change the
 * address computation out from under the raw byte offsets every caller passes.
 */
__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr);   // width = sizeof(void*)
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, GFX12_BUFFER_CONFIG);
    return *reinterpret_cast<const i32x4*>(&rsrc);
}

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b8(uint8_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b16(uint16_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");


__device__ inline float2 load_global_vec2_async(const float2* gptr) {
    float2 v;
    asm volatile(
        "global_load_b64 %0, %1, off\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline float4 load_global_vec4_async(const float4* gptr) {
    float4 v;
    asm volatile(
        "global_load_b128 %0, %1, off\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__device__ inline void store_global_b128_async(void* gptr, __uint128_t value) {
    asm volatile(
        "global_store_b128 %0, %1, off\n"
        :
        : "v"(gptr), "v"(value)
        : "memory"
    );
}

__device__ inline float2 load_shared_vec_async(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_load_b64 %0, %1\n"
        // no wait: the caller batches these and calls lds_wait<>()
        : "=v"(result)              // Output: store result in float2
        : "v"(lds_off)              // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

/**
 * @brief Wait until at most `N` LDS operations are still outstanding.
 *
 * LDS returns in order, so lds_wait<N>() after issuing a batch of M+N reads
 * retires exactly the first M of them. That is what lets a caller issue the
 * reads for K-slice k+1, then wait only on slice k's, and run slice k's math
 * with the next slice's reads still in flight.
 *
 * On gfx11 this was lgkmcnt, which also counted scalar memory; gfx12 gives LDS
 * its own counter, so a wait here no longer accidentally waits on an s_load.
 */
template<int N=0> __device__ inline void lds_wait() {
    static_assert(N >= 0 && N <= 63, "dscnt is 6 bits");
    asm volatile("s_wait_dscnt %0" :: "i"(N) : "memory");
}

/*
 * 128-bit LDS access.
 *
 * These have to be inline asm for the same reason the b64 pair above does, and
 * the reason is worth stating because it is not obvious from the C++: a shared
 * tile reached through shared_allocator is a *generic* pointer as far as the
 * compiler is concerned, so writing `*(float4*)p` gets you flat_load_b128, not
 * ds_load_b128.  Both address the same bytes, but flat goes out through the
 * memory pipe and counts against loadcnt rather than dscnt.  Taking the 32-bit
 * LDS offset and naming the instruction is what pins the access to the LDS pipe.
 *
 * Neither waits: the callers issue a batch and then a single lds_wait<>().
 */
// The asm operands are the native vector type rather than HIP's float4: a
// 16-byte HIP_vector_type is not something clang will place in a register for a
// "v" *input* constraint ("indirect register inputs"), though it accepts it as
// an output.  The memcpys are free; both types are four consecutive dwords.
typedef float raw_v4f __attribute__((ext_vector_type(4)));

__device__ inline float4 load_shared_vec4_async(uint32_t lds_off) {
    raw_v4f v;
    asm volatile(
        "ds_load_b128 %0, %1\n"
        : "=v"(v)
        : "v"(lds_off)
        : "memory"
    );
    float4 result;
    __builtin_memcpy(&result, &v, sizeof(v));
    return result;
}

__device__ inline void store_shared_vec4(uint32_t lds_off, float4 val) {
    raw_v4f v;
    __builtin_memcpy(&v, &val, sizeof(v));
    asm volatile(
        "ds_store_b128 %0, %1\n"
        :
        : "v"(lds_off), "v"(v)
        : "memory"
    );
}

/* ----------   To prevent generic addressing  ---------- */

template<typename T> struct move {
    __device__ static inline void lds(T& dst, uint32_t src);
    __device__ static inline void sts(uint32_t dst, const T& src);
    __device__ static inline void ldg(T& dst, T* src);
    __device__ static inline void stg(T* dst, const T& src);
};

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uint32_t bytes    = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args>                       inline constexpr uint32_t size_bytes             = 0; // base case
template<typename T, typename... Args>           inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens
