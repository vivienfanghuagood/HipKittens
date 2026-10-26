/**
 * @file
 * @brief Register strips: rows held so that each lane owns contiguous runs.
 *
 * An rt's lane layout is the one WMMA gives it. A 16-bit row tile holds each
 * row twice, once in each wave half; an fp32 tile's lanes own alternate
 * columns. That is the right shape for an mma and the wrong one for a kernel
 * that only streams rows -- a norm, a softmax -- which wants each lane to own a
 * contiguous run of a row, so that one 16-byte access fills it and nothing is
 * replicated. Triton calls that a blocked layout; this is it, for one warp.
 *
 * rs<T, ROWS, COLS, RUN>: ROWS rows of COLS columns. Lane l owns, of every
 * row, the runs
 *
 *     [j*SPAN + l*RUN, j*SPAN + l*RUN + RUN)     j = 0 .. CHUNKS-1
 *
 * where SPAN = 32*RUN is what one chunk of the wave covers and CHUNKS =
 * COLS/SPAN. A chunk is one contiguous SPAN-element stretch of the row, so a
 * load of it is one fully coalesced access per lane: 8 bf16 or 4 fp32 per
 * 16 bytes. Nothing is replicated across lanes: a lane's data is
 * ROWS*CHUNKS*RUN elements.
 *
 * Its vectors follow from that:
 *   col_vec, one value per row, is rs_cv<T, ROWS>: every lane holds all ROWS
 *     values -- a row reduction ends in a butterfly, which leaves the answer in
 *     every lane.
 *   row_vec, one value per column, is rs<T, 1, COLS, RUN>: the strip's own
 *     column layout, so a weight vector loads with the same 16-byte accesses
 *     as the rows it scales.
 */

#pragma once

#include <type_traits>

#include "../../common/common.cuh"

namespace kittens {

namespace ducks {
/**
 * @namespace rs
 * @brief Concepts for register strips and their per-row vectors.
 */
namespace rs {
struct identifier {};
struct cv_identifier {};
} // namespace rs
} // namespace ducks

/**
 * @brief One value per row of a strip, held by every lane of the warp.
 *
 * data[r][0] is row r's value. The trailing extent of 1 is there so code
 * written against rv's data[outer][inner] walks this too.
 */
template<typename _T, int _length>
struct rs_cv {
    using identifier = ducks::rs::cv_identifier;
    static_assert(_length > 0, "a strip has at least one row");
    using T = typename base_types::packing<_T>::unpacked_type;
    using dtype = T;
    static constexpr int length    = _length;
    static constexpr int outer_dim = _length;
    static constexpr int inner_dim = 1;
    T data[outer_dim][inner_dim];
    __device__ inline       T* operator[](size_t idx)       { return &data[idx][0]; }
    __device__ inline const T* operator[](size_t idx) const { return &data[idx][0]; }
};

/**
 * @brief A register strip: ROWS rows of COLS columns, RUN contiguous
 * elements of a row per lane per chunk.
 */
template<typename _T, int _rows, int _cols, int _run = 8>
struct rs {
    using identifier = ducks::rs::identifier;
    static_assert(kittens::ducks::base_types::T1<_T>);
    using T = typename base_types::packing<_T>::unpacked_type;
    using dtype = T;
    static constexpr int rows   = _rows;
    static constexpr int cols   = _cols;
    static constexpr int run    = _run;
    static_assert(rows > 0, "a strip has at least one row");
    static_assert(run == 1 || run == 2 || run == 4 || run == 8 || run == 16,
                  "a run is a power of two of at most 16 elements");
    static constexpr int span   = run * WARP_THREADS;  ///< columns one chunk covers
    static_assert(cols % span == 0, "a strip is a whole number of chunks: COLS % (32*RUN) == 0");
    static constexpr int chunks = cols / span;
    static constexpr int elements_per_thread = rows * chunks * run;
    T data[rows][chunks][run];
    using col_vec = rs_cv<T, rows>;
    using row_vec = rs<T, 1, cols, run>;
};

namespace ducks {
namespace rs {
template<typename T>
concept all = requires { typename T::identifier; } &&
              std::is_same_v<typename T::identifier, identifier>;
template<typename T>
concept cv = requires { typename T::identifier; } &&
             std::is_same_v<typename T::identifier, cv_identifier>;
} // namespace rs
} // namespace ducks

template<int R, int C, int RUN = 8> using rs_fl = rs<float, R, C, RUN>;
template<int R, int C, int RUN = 8> using rs_bf = rs<bf16,  R, C, RUN>;
template<int R, int C, int RUN = 8> using rs_hf = rs<half,  R, C, RUN>;

} // namespace kittens
