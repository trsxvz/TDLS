#ifndef TDLS_SOLVERS_TILED_LU_CONFIG_HPP
#define TDLS_SOLVERS_TILED_LU_CONFIG_HPP



/// \file
/// \brief Compile-time configuration of the tiled LU solver family.
/// \author Tristan Chenaille
///
/// Every knob is a `static constexpr` member of a config type passed as the
/// `Cfg` template argument of the solvers. TiledLuDefaultConfig holds the
/// tuned defaults; a caller overrides them by providing its own type with
/// the same members.



#include <limits>
#include <type_traits>

#include <tdls/core/macros.hpp>



namespace tdls {



/// \brief Elimination schedule of the tiled factorization.
enum class TiledLuSchedule {
    RightLooking, ///< Factor the diagonal tile, push updates into the trailing matrix.
    LeftLooking   ///< Pull updates from prior tiles when a tile is visited.
};



/// \brief Default compile-time knobs of the tiled LU solvers.
/// \tparam T scalar type (float or double)
template<typename T>
struct TiledLuDefaultConfig {

    /// Acceptable-pivot threshold of the out-of-block search. An in-tile
    /// pivot candidate whose magnitude reaches this value is accepted
    /// without looking outside the tile; below it, the search extends to
    /// the rows under the tile (out-of-block pivoting) and the best
    /// corrected candidate wins.
    static constexpr T oob_threshold = std::is_same_v<T, float> ? T(1e-4f) : T(1e-10);

    /// Singularity floor: the factorization is declared singular when even
    /// the best candidate stays below it. `numeric_limits<T>::min()`
    /// rejects only a zero/subnormal pivot - a genuine structural
    /// singularity. A merely small pivot is kept on purpose: the loss of
    /// stability is surfaced by the backward error and overflow is caught
    /// downstream by the caller, whereas an absolute floor wrongly flags
    /// well-conditioned matrices at small scale.
    static constexpr T singular_eps = std::numeric_limits<T>::min();

    /// Out-of-block pivot search strategy. When true, the below-tile scan
    /// stops at the first candidate whose corrected magnitude reaches
    /// oob_threshold instead of scanning the whole panel for the maximum;
    /// the running maximum is still kept as the fallback when no candidate
    /// is acceptable. Cheaper in the OOB-heavy regime - especially
    /// left-looking, where every candidate replays the prior tiles - at
    /// the cost of a possibly smaller (but still >= oob_threshold) pivot.
    /// Set false to restore the full-panel partial-pivoting scan.
    static constexpr bool oob_first_acceptable = true;

    /// Unroll policy of the in-tile scalar loops, applied through a
    /// two-branch `if constexpr` (the pragma dialect itself lives in
    /// core/macros.hpp). true: loops indexing register tiles carry a
    /// forced-unroll pragma - the guard that keeps tiles in registers on
    /// GPU backends, where a rolled loop indexes the tile dynamically and
    /// demotes it to slow local memory. false: no unroll pragma anywhere -
    /// faster compiles, GPU performance not guaranteed. Outer tile-sweep
    /// loops never carry a pragma in either branch.
    static constexpr bool unroll_inner = true;
};



} // namespace tdls



#endif // TDLS_SOLVERS_TILED_LU_CONFIG_HPP
