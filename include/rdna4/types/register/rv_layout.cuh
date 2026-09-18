/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace rv_layout
 * 
 * @brief A namespace for template metaprogramming with register vector layouts.
 */
namespace rv_layout {

/**
 * @brief A dummy type used to identify an aligned layout.
 *
 * "Aligned" means the vector's index runs along a tile's *element* axis, so a
 * lane holds several entries and the whole vector is replicated across the 16
 * lanes of a wave half.  Every gfx12 fragment puts 8 entries in a lane, so the
 * two element types agree on the count -- but not on *which* entries: an f32
 * accumulator's halves split the axis even/odd, while a bf16/half operand's
 * halves split it into the runs 0..7 and 8..15.  rv_align_elem() in rt_base.cuh
 * is the inverse map.  inner_dim is still computed rather than fixed, for the
 * same reason as on gfx11; on gfx12 it just happens to come out the same for
 * every type.
 */
struct align {};
/**
 * @brief A dummy type used to identify an orthogonal layout.
 *
 * "Orthogonal" means the vector's index runs along the *lane* axis (l%16), so
 * each lane holds exactly one entry per subtile, replicated across the two wave
 * halves.
 */
struct ortho { constexpr static int inner_dim = 1; };
/**
 * @brief A dummy type used to identify an unreplicated layout, for better coalesced loads and vector operations like layernorm.
 */
struct naive { constexpr static int inner_dim = 1; };

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;

} // namespace rv_layout
} // namespace ducks
} // namespace kittens