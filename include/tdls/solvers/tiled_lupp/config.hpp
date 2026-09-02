#ifndef TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP
#define TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP



/// \file
/// \brief Compile-time configuration of the TiledLUpp solver family.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every knob is a member of the TiledLUppConfig aggregate. A constexpr
/// instance of it is passed as the `Config` non-type template argument
/// of the TiledLUpp solvers. The default-constructed value carries the
/// tuned defaults; a caller overrides individual knobs with designated
/// initializers, in declaration order:
///
///   constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 4};
///
/// The aggregate is a structural type, as class-type non-type template
/// parameters require: every member is public, and two configurations
/// with equal members name the same solver instantiation. The two
/// floating-point knobs are held as tdls::StructuralReal values: exact,
/// built and read back implicitly, and admissible as template arguments
/// on every compiler supporting class-type template parameters (a
/// plain floating-point member would need P1907R1, absent from GCC 10
/// and from every nvcc before CUDA 13.0).



#include <limits>
#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/core/structural_real.hpp>



namespace tdls {



/// \brief Elimination schedule of the tiled factorization.
enum class TiledLUppSchedule {
    RightLooking, ///< Factor the diagonal tile, push updates into the trailing matrix.
    LeftLooking   ///< Pull updates from prior tiles when a tile is visited.
};



/// \brief Memory layout of the factor matrix.
enum class TiledLUppLayout {
    RowMajor, ///< Element (r, c) at flat index r * N + c, the TFEL convention.
    ColMajor  ///< Element (r, c) at flat index c * N + r.
};



/// \brief Compile-time knobs of the TiledLUpp solvers, carrying the tuned
/// defaults.
/// \tparam T scalar type (float or double)
template<typename T>
struct TiledLUppConfig {

    /// Tile extent: the matrix is processed as a grid of tile_size x
    /// tile_size register tiles. This is the main performance axis of the
    /// solvers. Tune it per system dimension (measured optima in the
    /// source project: 3, 4 or 6 depending on N). Both TiledLUpp solvers only
    /// require tile_size >= 1. The tile size may exceed the dimension (the
    /// grid is then a single partial tile), and tile_size = 1 degenerates
    /// into an untiled scalar elimination.
    int tile_size = 3;

    /// Elimination schedule of the tiled factorization, see
    /// TiledLUppSchedule.
    TiledLUppSchedule schedule = TiledLUppSchedule::RightLooking;

    /// Acceptable-pivot threshold of the out-of-tile search. Both
    /// thresholds carry the scalar type T by construction, stored
    /// exactly as tdls::StructuralReal values so that the configuration
    /// stays a template argument everywhere; write them as plain T
    /// values, the conversions are implicit. An in-tile
    /// pivot candidate whose magnitude reaches this value is accepted
    /// without looking outside the tile; below it, the search extends to
    /// the rows under the tile (out-of-tile pivoting) and the best
    /// corrected candidate wins.
    StructuralReal<T> oot_threshold = std::is_same_v<T, float> ? T(1e-4f) : T(1e-10);

    /// Singularity floor: the factorization is declared singular when even
    /// the best candidate of the out-of-tile recovery stays below it. The
    /// floor only guards the recovery path (a pivot reaching oot_threshold
    /// is accepted directly), so it must not exceed oot_threshold; the
    /// solvers enforce this contract at compile time. `numeric_limits<T>::min()`
    /// rejects only a zero/subnormal pivot, a genuine structural
    /// singularity. A merely small pivot is kept on purpose: the loss of
    /// stability is surfaced by the backward error and overflow is caught
    /// downstream by the caller, whereas an absolute floor wrongly flags
    /// well-conditioned matrices at small scale.
    StructuralReal<T> singular_eps = std::numeric_limits<T>::min();

    /// Out-of-tile pivot search strategy. When true, the below-tile scan
    /// stops at the first candidate whose corrected magnitude reaches
    /// oot_threshold instead of scanning the whole panel for the maximum;
    /// the running maximum is still kept as the fallback when no candidate
    /// is acceptable. Cheaper in the OOT-heavy regime (especially
    /// left-looking, where every candidate replays the prior tiles), at
    /// the cost of a possibly smaller (but still >= oot_threshold) pivot.
    /// Set false to restore the full-panel partial-pivoting scan.
    bool oot_first_acceptable = true;

    /// Unroll policy of the in-tile scalar loops, applied through a
    /// two-branch `if constexpr` (the pragma dialect itself lives in
    /// core/macros.hpp). true: loops indexing register tiles carry a
    /// forced-unroll pragma, the guard that keeps tiles in registers on
    /// GPU backends, where a rolled loop indexes the tile dynamically and
    /// demotes it to slow local memory. false: no unroll pragma anywhere,
    /// for faster compiles, GPU performance not guaranteed. Outer tile-sweep
    /// loops never carry a pragma in either branch.
    bool unroll_inner = true;

    /// Memory layout of the factor matrix, in both solvers and both
    /// residency modes. The knob only remaps the flat element index that
    /// the element stride scales: the arithmetic sequence is unchanged,
    /// so both layouts produce bitwise-identical results on identical
    /// inputs. Row-major is the TFEL convention and the default. A
    /// factorization keeps the layout of its configuration, so its
    /// substitutions share it by construction. The vector operands and
    /// the pivot are one-dimensional and unaffected.
    TiledLUppLayout layout = TiledLUppLayout::RowMajor;
};



} // namespace tdls



#endif // TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP
