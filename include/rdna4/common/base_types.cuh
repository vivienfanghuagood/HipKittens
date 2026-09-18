/**
 * @file
 * @brief Declarations, manipulations, and wrappers for basic types.
 * 
 * This file is a bunch of utilities for going back and forth between different types.
 * 
 * Many of them are for the compiler, so as to clean up the code. It unfortunately
 * seems necessary when we have types we really care about that are less than word width.
 */

#pragma once

#include <hip_bf16.h>
#include <hip_fp16.h>
#include <hip_fp8.h>
#include <string>
#include <bit>


namespace kittens {

// /**
//  * @brief Bfloat16 floating-point type.
//  */
using bf16 = __hip_bfloat16;
/**
 * @brief Half-precision floating-point type.
 */
using half = __half;
// /**
//  * @brief Packed word of two bfloat16 floating-point values.
//  */
using bf16_2 = __hip_bfloat162;
/**
 * @brief Packed word of two half-precision floating-point values.
 */
using half_2 = __half2;
/*
 * fp8 on gfx12 is OCP, not fnuz -- and this is not a style preference, it is
 * the only spelling that compiles.  HIP gates the two encodings by target
 * (amd_hip_fp8.h:41): gfx942 sets HIP_FP8_TYPE_FNUZ and clears
 * HIP_FP8_TYPE_OCP, gfx1200/gfx1201/gfx950 do the reverse.  On gfx1201 the
 * fnuz types are still *declared* -- so `using fp8e4m3 = __hip_fp8_e4m3_fnuz`
 * inherited from the CDNA tree compiles for as long as nothing instantiates it
 * -- but they have no __device__ constructor from float, so the first convertor
 * that touches one fails.  Which is exactly why the inherited typedef survived
 * unnoticed until fp8 was wired up.
 *
 * The difference is not only in the name.  fnuz has no infinities and one NaN
 * (0x80, negative zero's slot); OCP e4m3 has no infinities either but reserves
 * 0x7F/0xFF for NaN, and OCP e5m2 does have infinities.  The exponent bias
 * differs too, fnuz e4m3 being 8 against OCP's 7, so the same byte means a
 * different number.  Anything crossing between a gfx942 kernel and a gfx12 one
 * has to convert, not reinterpret.
 */

/**
 * @brief float8 E4M3 (OCP) floating-point type.
 */
using fp8e4m3 = __hip_fp8_e4m3;
/**
 * @brief Packed word of two E4M3 values.
 */
using fp8e4m3_2 = __hip_fp8x2_e4m3;
/**
 * @brief Packed word of four E4M3 values.
 */
using fp8e4m3_4 = __hip_fp8x4_e4m3;
/**
 * @brief float8 E5M2 (OCP) floating-point type, AMD's "bf8".
 */
using fp8e5m2 = __hip_fp8_e5m2;
/**
 * @brief Packed word of two E5M2 values.
 */
using fp8e5m2_2 = __hip_fp8x2_e5m2;
/**
 * @brief Packed word of four E5M2 values.
 */
using fp8e5m2_4 = __hip_fp8x4_e5m2;

namespace ducks {
/**
 * @namespace base_types
 *
 * @brief A namespace for concepts for basic data types.
 */
namespace base_types {

template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2> || std::is_same_v<T, fp8e4m3_4> || std::is_same_v<T, fp8e5m2_4>;
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>;

/// The two 8-bit operand formats gfx12 WMMA accepts, in either A or B position.
template<typename T>
concept fp8 = std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>;

} // namespace base_types
} // namespace ducks

/**
 * @namespace base_types
 *
 * @brief A namespace for ThunderKittens basic data types.
 */
namespace base_types {

/**
 * @brief Provides compile-time constants for different types.
 *
 * @tparam T The type for which to provide constants.
 */
template<typename T> struct constants {
    /**
     * @brief Zero
     * @return Constexpr zero with type T
     */
    static __device__ inline constexpr T zero()      { return T{0}; }
    /**
     * @brief One
     * @return Constexpr one with type T
     */
    static __device__ inline constexpr T one()       { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static __device__ inline constexpr T neg_infty() { return T{-INFINITY}; }
};
template<> struct constants<float2> {
    static __device__ inline constexpr float2 zero()      { return float2{0.f, 0.f}; }
    static __device__ inline constexpr float2 one()       { return float2{1.f, 1.f}; }
    static __device__ inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __device__ inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};
template<> struct constants<bf16> {
    static __device__ inline constexpr bf16 zero()      { return std::bit_cast<bf16>(uint16_t(0x0000)); } // unfortunately __float2bf16_rn is not constexpr
    static __device__ inline constexpr bf16 one()       { return std::bit_cast<bf16>(uint16_t(0x3F80)); }
    static __device__ inline constexpr bf16 pos_infty() { return std::bit_cast<bf16>(uint16_t(0x7F80)); }
    static __device__ inline constexpr bf16 neg_infty() { return std::bit_cast<bf16>(uint16_t(0xFF80)); }
};
template<> struct constants<bf16_2> {
    static __device__ inline bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static __device__ inline bf16_2 one()       { return bf16_2{constants<bf16>::one(),       constants<bf16>::one()};       }
    static __device__ inline bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __device__ inline bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template<> struct constants<half> {
    static __device__ inline constexpr half zero()      { return std::bit_cast<half>(uint16_t(0x0000)); }
    static __device__ inline constexpr half one()       { return std::bit_cast<half>(uint16_t(0x3C00)); }
    static __device__ inline constexpr half pos_infty() { return std::bit_cast<half>(uint16_t(0x7C00)); }
    static __device__ inline constexpr half neg_infty() { return std::bit_cast<half>(uint16_t(0xFC00)); }
};
template<> struct constants<half_2> {
    static __device__ inline constexpr half_2 zero()      { return std::bit_cast<half_2>(uint32_t(0x00000000)); }
    static __device__ inline constexpr half_2 one()       { return std::bit_cast<half_2>(uint32_t(0x3C003C00)); }
    static __device__ inline constexpr half_2 pos_infty() { return std::bit_cast<half_2>(uint32_t(0x7C007C00)); }
    static __device__ inline constexpr half_2 neg_infty() { return std::bit_cast<half_2>(uint32_t(0xFC00FC00)); }
};
// OCP e4m3 has no infinities -- the largest finite value is 0x7E (448) and
// 0x7F/0xFF are NaN -- so pos_infty()/neg_infty() are deliberately absent
// rather than approximated. A specialization replaces the primary template
// whole, so asking for either is a compile error, which is the right answer:
// fp8 is an operand format here, and a reduction that needs an identity should
// be running on the f32 accumulator.
template<> struct constants<fp8e4m3> {
    static __device__ inline constexpr fp8e4m3 zero() { return std::bit_cast<fp8e4m3>(uint8_t(0x00)); }
    static __device__ inline constexpr fp8e4m3 one() { return std::bit_cast<fp8e4m3>(uint8_t(0x38)); } // 0_0111_000, bias 7
};
template<> struct constants<fp8e4m3_2> {
    static __device__ inline constexpr fp8e4m3_2 zero() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp8e4m3_2 one() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x3838)); }
};
template<> struct constants<fp8e4m3_4> {
    static __device__ inline constexpr fp8e4m3_4 zero() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr fp8e4m3_4 one() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x38383838)); }
};
// e5m2 does have infinities (0x7C / 0xFC), so unlike e4m3 it can supply them.
template<> struct constants<fp8e5m2> {
    static __device__ inline constexpr fp8e5m2 zero()      { return std::bit_cast<fp8e5m2>(uint8_t(0x00)); }
    static __device__ inline constexpr fp8e5m2 one()       { return std::bit_cast<fp8e5m2>(uint8_t(0x3C)); } // 0_01111_00, bias 15
    static __device__ inline constexpr fp8e5m2 pos_infty() { return std::bit_cast<fp8e5m2>(uint8_t(0x7C)); }
    static __device__ inline constexpr fp8e5m2 neg_infty() { return std::bit_cast<fp8e5m2>(uint8_t(0xFC)); }
};
template<> struct constants<fp8e5m2_2> {
    static __device__ inline constexpr fp8e5m2_2 zero()      { return std::bit_cast<fp8e5m2_2>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp8e5m2_2 one()       { return std::bit_cast<fp8e5m2_2>(uint16_t(0x3C3C)); }
    static __device__ inline constexpr fp8e5m2_2 pos_infty() { return std::bit_cast<fp8e5m2_2>(uint16_t(0x7C7C)); }
    static __device__ inline constexpr fp8e5m2_2 neg_infty() { return std::bit_cast<fp8e5m2_2>(uint16_t(0xFCFC)); }
};
template<> struct constants<fp8e5m2_4> {
    static __device__ inline constexpr fp8e5m2_4 zero()      { return std::bit_cast<fp8e5m2_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr fp8e5m2_4 one()       { return std::bit_cast<fp8e5m2_4>(uint32_t(0x3C3C3C3C)); }
    static __device__ inline constexpr fp8e5m2_4 pos_infty() { return std::bit_cast<fp8e5m2_4>(uint32_t(0x7C7C7C7C)); }
    static __device__ inline constexpr fp8e5m2_4 neg_infty() { return std::bit_cast<fp8e5m2_4>(uint32_t(0xFCFCFCFC)); }
};
template<> struct constants<int> {
    static __device__ inline constexpr int zero()      { return 0; }
    static __device__ inline constexpr int one()       { return 1; }
};
template<> struct constants<int2> {
    static __device__ inline constexpr int2 zero()      { return int2{0, 0}; }
    static __device__ inline constexpr int2 one()       { return int2{1, 1}; }
};

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
    /**
     * @brief The number of elements packed together.
     *
     * @return constexpr int representing number of elements within the type.
     */
    static __device__ inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static __device__ inline constexpr T pack(const auto &i);
};
template<> struct packing<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<half> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float4> {
    static __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<fp8e4m3> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e4m3_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e5m2> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e5m2;
    using packed_type = fp8e5m2_4;
};
template<> struct packing<fp8e5m2_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e5m2;
    using packed_type = fp8e5m2_4;
};

/**
 * @brief Pack four float8 into 32-bits.
 */
static __host__ __device__ inline fp8e4m3_4 make_fp8e4m3_4(const fp8e4m3 & x, const fp8e4m3 & y, const fp8e4m3 & z, const fp8e4m3 & w) {
    return std::bit_cast<fp8e4m3_4>(
        static_cast<uint32_t>(
            std::bit_cast<uint8_t>(x) | 
            (std::bit_cast<uint8_t>(y) << 8) | 
            (std::bit_cast<uint8_t>(z) << 16) | 
            (std::bit_cast<uint8_t>(w) << 24)
        )
    );
}

/**
 * @brief Provides templated functionality to convert between different types.
 *
 * @tparam T The target type for conversion.
 * @tparam U The source type for conversion.
 */
template<typename T, typename U> struct convertor {
    /**
     * @brief Converts a value of type U to type T.
     *
     * @param u[in] The value of type U to convert.
     * @return T The converted value of type T.
     */
    static __host__ __device__ inline T convert(const U & u) {
        return (T)u;
    }
};
template<> struct convertor<float, bf16> {
    static __host__ __device__ inline float convert(const bf16 & u) {
        return 	__bfloat162float(u);
    }
};
// template<> struct convertor<bf16, float> {
//     static __host__ __device__ inline bf16 convert(const float & u) {
//         return 	__float2bfloat16(u);
//     }
// };
template<> struct convertor<bf16, float> {
    static __host__ __device__ inline bf16 convert(const float &u) {
        // Fast unsafe conversion (truncation only)
        return std::bit_cast<bf16>(
            static_cast<uint16_t>(
                std::bit_cast<uint32_t>(u) >> 16
            )
        );
    }
};
template<> struct convertor<float2, bf16_2> {
    static __host__ __device__ inline float2 convert(const bf16_2 & u) {
        return 	__bfloat1622float2(u);
    }
};
template<> struct convertor<bf16_2, float2> {
    static __host__ __device__ inline bf16_2 convert(const float2 &u) {
        return bf16_2{
            std::bit_cast<bf16>(static_cast<uint16_t>(std::bit_cast<uint32_t>(u.x) >> 16)),
            std::bit_cast<bf16>(static_cast<uint16_t>(std::bit_cast<uint32_t>(u.y) >> 16))
        };
    }
};
// template<> struct convertor<bf16_2, float2> {
//     static __host__ __device__ inline bf16_2 convert(const float2 &u) {
//         uint32_t result;
//         asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" 
//                      : "=v"(result) 
//                      : "v"(u.x), "v"(u.y));
//         return *reinterpret_cast<bf16_2*>(&result);
//     }
// };


template<> struct convertor<float, half> {
    static __host__ __device__ inline float convert(const half & u) {
        return __half2float(u);
    }
};
template<> struct convertor<half, float> {
    static __host__ __device__ inline half convert(const float & u) {
        return __float2half(u);
    }
};
template<> struct convertor<float2, half_2> {
    static __host__ __device__ inline float2 convert(const half_2 & u) {
        return __half22float2(u);
    }
};
template<> struct convertor<half_2, float2> {
    static __host__ __device__ inline half_2 convert(const float2 & u) {
        return __float22half2_rn(u);
    }
};
template<> struct convertor<bf16, half> {
    static __host__ __device__ inline bf16 convert(const half & u) {
        return __float2bfloat16(__half2float(u));
    }
};
template<> struct convertor<half, bf16> {
    static __host__ __device__ inline half convert(const bf16 & u) {
        return __float2half(__bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static __host__ __device__ inline bf16_2 convert(const half_2 & u) {
        return __float22bfloat162_rn(__half22float2(u));
    }
};
template<> struct convertor<half_2, bf16_2> {
    static __host__ __device__ inline half_2 convert(const bf16_2 & u) {
        return __float22half2_rn(__bfloat1622float2(u));
    }
};
template<> struct convertor<fp8e4m3_4, float4> {
    static __host__ __device__ inline fp8e4m3_4 convert(const float4& u) {
        return fp8e4m3_4(u);
    }
};
template<> struct convertor<float4, fp8e4m3_4> {
    static __host__ __device__ inline float4 convert(const fp8e4m3_4& u) {
        fp8e4m3 *vals = reinterpret_cast<fp8e4m3*>(const_cast<fp8e4m3_4*>(&u));
        return make_float4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]));
    }
};
template<> struct convertor<fp8e4m3_2, float2> {
    static __host__ __device__ inline fp8e4m3_2 convert(const float2& u) {
        return fp8e4m3_2(u);
    }
};
template<> struct convertor<float2, fp8e4m3_2> {
    static __host__ __device__ inline float2 convert(const fp8e4m3_2& u) {
        fp8e4m3 *vals = reinterpret_cast<fp8e4m3*>(const_cast<fp8e4m3_2*>(&u));
        return make_float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e4m3, float> {
    static __host__ __device__ inline fp8e4m3 convert(const float & u) {
        return fp8e4m3(u);
    }
};
template<> struct convertor<float, fp8e4m3> {
    static __host__ __device__ inline float convert(const fp8e4m3 & u) {
        return float(u);
    }
};
// Same six for e5m2. On gfx1200/gfx1201 HIP_FP8_CVT_FAST_PATH is set
// (amd_hip_fp8.h:33), so these lower to v_cvt_pk_fp8_f32 / v_cvt_pk_f32_fp8
// rather than the software fallback the host path uses.
template<> struct convertor<fp8e5m2_4, float4> {
    static __host__ __device__ inline fp8e5m2_4 convert(const float4& u) {
        return fp8e5m2_4(u);
    }
};
template<> struct convertor<float4, fp8e5m2_4> {
    static __host__ __device__ inline float4 convert(const fp8e5m2_4& u) {
        fp8e5m2 *vals = reinterpret_cast<fp8e5m2*>(const_cast<fp8e5m2_4*>(&u));
        return make_float4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]));
    }
};
template<> struct convertor<fp8e5m2_2, float2> {
    static __host__ __device__ inline fp8e5m2_2 convert(const float2& u) {
        return fp8e5m2_2(u);
    }
};
template<> struct convertor<float2, fp8e5m2_2> {
    static __host__ __device__ inline float2 convert(const fp8e5m2_2& u) {
        fp8e5m2 *vals = reinterpret_cast<fp8e5m2*>(const_cast<fp8e5m2_2*>(&u));
        return make_float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e5m2, float> {
    static __host__ __device__ inline fp8e5m2 convert(const float & u) {
        return fp8e5m2(u);
    }
};
template<> struct convertor<float, fp8e5m2> {
    static __host__ __device__ inline float convert(const fp8e5m2 & u) {
        return float(u);
    }
};
}
}