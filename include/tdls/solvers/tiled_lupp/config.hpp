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
/// with equal members name the same solver instantiation. The
/// floating-point knobs are held as tdls::StructuralReal values: exact,
/// built and read back implicitly, and admissible as template arguments
/// on every compiler supporting class-type template parameters (a
/// plain floating-point member would need P1907R1, absent from GCC 10
/// and from every nvcc before CUDA 13.0).



#include <limits>
#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/core/structural_real.hpp>
#include <tdls/solvers/options.hpp>



namespace tdls {



/// \brief Compile-time knobs of the TiledLUpp solvers, carrying the tuned
/// defaults.
/// \tparam T scalar type (float, double or long double)
template<typename T>
struct TiledLUppConfig {

    /// Tile extent: the matrix is processed as a grid of tile_size x
    /// tile_size register tiles. This is the main performance axis of the
    /// solvers. Tune it per system dimension: the optimum measured in
    /// tfelGPU, the source project, depends on N. On an RTX 5080, one
    /// thread per system inside implicit MFront behaviours (N = 12 to 19,
    /// double), tiles of 4 or 5 were the fastest, and 3 was up to 28 %
    /// slower at N = 19 (tile 4, left-looking, best there); 4, the default
    /// (3 up to version 0.3.0). Both TiledLUpp solvers
    /// only require tile_size >= 1. The tile size may exceed the
    /// dimension (the grid is then a single partial tile), and
    /// tile_size = 1 degenerates into an untiled scalar elimination.
    int tile_size = 4;

    /// Elimination schedule of the tiled factorization, see Schedule.
    Schedule schedule = Schedule::RightLooking;

    /// Row interchanges of the partial pivoting. Logical, the default:
    /// rows never move in the stored matrix, the pivot array maps each
    /// position to its row, and the tiles go back to the rows it gives, an
    /// index known at run time only. On GPU, a caller-local matrix is then
    /// indexed at run time and lives in the local memory of the thread.
    ///
    /// Physical: the whole matrix is a single tile, and the rows are
    /// interchanged as in LAPACK, through a test of every row index against
    /// the pivot row, so that once the loops are unrolled every index is
    /// known at compile time: a caller-local matrix, its pivots and its
    /// right-hand sides stay in registers. tile_size and schedule are
    /// ignored. The pivots are those of Logical over a single tile (a
    /// tile_size of at least N), and so are the operations of the
    /// factorization: the factors agree bitwise, rows reordered; the
    /// substitutions sum in another order, the solutions agree within
    /// rounding errors. The pivot
    /// array then holds the interchanges: piv[j] is the row exchanged with
    /// row j at step j. A factorization keeps the scheme of its
    /// configuration, so its substitutions share it by construction.
    ///
    /// For small systems only: the elimination is unrolled over the whole
    /// matrix, N^3 operations in the code, and it pays only while the
    /// matrix fits in the registers of a thread, about N^2 sizeof(T) / 4 <=
    /// 200 on NVIDIA GPUs, less what the caller keeps alive. Beyond it, the
    /// compiler spills the matrix, and Logical keeps only the working tiles
    /// in registers.
    ///
    /// Auto is rejected at compile time: it is the default of the blocks of
    /// ArrowheadLUpp only (ArrowheadLUppConfig::block_row_interchange),
    /// which choose by block size, precision and target and hand TiledLUpp
    /// Physical or Logical.
    RowInterchange row_interchange = RowInterchange::Logical;

    /// Acceptable-pivot threshold of the out-of-tile search. The
    /// floating-point knobs carry the scalar type T by construction, stored
    /// exactly as tdls::StructuralReal values so that the configuration
    /// stays a template argument everywhere; write them as plain T
    /// values, the conversions are implicit (a long double value must
    /// fit a 63-bit odd mantissa, as every double literal does). An in-tile
    /// pivot candidate whose magnitude reaches this value is accepted
    /// without looking outside the tile; below it, the search extends to
    /// the rows under the tile (out-of-tile pivoting) and the best
    /// corrected candidate wins.
    StructuralReal<T> oot_pivot_threshold = std::is_same_v<T, float> ? T(1e-4f) : T(1e-10);

    /// Out-of-tile pivot search strategy. When true, the below-tile scan
    /// stops at the first candidate whose corrected magnitude reaches
    /// oot_pivot_threshold instead of scanning the whole panel for the maximum;
    /// the running maximum is still kept as the fallback when no candidate
    /// is acceptable. Cheaper in the OOT-heavy regime (especially
    /// left-looking, where every candidate replays the prior tiles), at
    /// the cost of a possibly smaller (but still >= oot_pivot_threshold) pivot.
    /// Set false to restore the full-panel partial-pivoting scan.
    bool oot_first_acceptable = true;

    /// Relative threshold of the pivot choice inside the diagonal tile,
    /// in (0, 1]. At each column, the row in place keeps the pivot when
    /// its magnitude reaches both this fraction of the largest in-tile
    /// candidate and oot_pivot_threshold. Otherwise the largest in-tile
    /// candidate wins, the first one on ties. This threshold governs the
    /// in-tile choice only: the out-of-tile recovery keeps its absolute
    /// trigger, oot_pivot_threshold. The in-tile multipliers stay bounded
    /// by its inverse.
    ///
    /// The default 0.1 is the one of CooperativeLUpp. It keeps most rows
    /// in place: fewer row swaps, and pivot arrays that stay alike across
    /// the threads of a warp, whose pivot-driven reads then hit the same
    /// offsets. A value of 1 takes the largest in-tile candidate, the
    /// choice of TiledLUpp up to version 0.2.0, and the solvers then
    /// compile the test out. The solvers enforce the range at compile
    /// time.
    StructuralReal<T> relative_pivot_threshold = T(0.1);

    /// Singularity floor: the factorization is declared singular when even
    /// the best candidate of the out-of-tile recovery stays below it, or
    /// when the best pivot of the last diagonal tile does, where no row is
    /// left to search. The floor only guards the pivots below
    /// oot_pivot_threshold (a pivot reaching it is accepted directly), so
    /// it must not exceed oot_pivot_threshold, and it must be positive, a
    /// zero floor letting a zero pivot through; the solvers enforce both
    /// contracts at compile time. `numeric_limits<T>::min()` rejects only
    /// a zero/subnormal pivot, a genuine structural singularity. A merely
    /// small pivot is kept on purpose: the loss of stability is surfaced
    /// by the backward error and overflow is caught downstream by the
    /// caller, whereas an absolute floor wrongly flags well-conditioned
    /// matrices at small scale.
    StructuralReal<T> singular_floor = std::numeric_limits<T>::min();

    /// Unroll policy, applied through a two-branch `if constexpr` (the
    /// pragma dialect itself lives in core/macros.hpp). The knob does not
    /// govern every loop on purpose: only the loops where offering the
    /// choice can noticeably change the performance. These are the loops
    /// indexing the register tiles (the in-tile scalar loops) and, in the
    /// compile-time solver, the loops over the N entries of the pivot and
    /// vector operands. true: they carry a forced-unroll pragma, the guard
    /// that keeps the tiles in registers on GPU backends, where a rolled
    /// loop indexes them dynamically and demotes them to slow local
    /// memory. A caller-local matrix stays in local memory anyway: the
    /// pivots index its rows at run time. Caller-local pivots and
    /// right-hand sides may too, depending on the dimension and the
    /// compiler. Inside implicit behaviours, false was 10 to 23 % slower
    /// (RTX 5080, double). false: no unroll pragma
    /// anywhere, for faster compiles, GPU performance not guaranteed. The
    /// tile sweeps walk memory, not registers, and carry no pragma in
    /// either branch. The runtime solver carries no pragma and ignores
    /// the knob. Every TDLS family names this knob unroll_loops.
    bool unroll_loops = true;

    /// Memory layout of the matrix, in both solvers and both residency
    /// modes. The solver adapts its design to it, at compile time. It
    /// factors S, the stored matrix read row-major, so that the tiles are
    /// read in address order in both layouts. Row-major, the TFEL
    /// convention and the default: S = A and P A = L U. The forward pass
    /// is folded into the factorization by solve_inplace, then U x = y.
    /// Column-major: S = A^T, and the pivoting interchanges columns of A.
    /// This is partial pivoting of A^T, as stable as partial pivoting of
    /// A. Since A = U^T L^T P, the solve runs through the transposed
    /// factors. First U^T z = b, folded into the factorization by
    /// solve_inplace. Then L^T w = z, which divides by nothing, and
    /// x = P^T w. A matrix stored column-major and its transpose stored
    /// row-major are thus the same array, factored the same way bit for
    /// bit. The first solves A x = b, the second A^T x = b. A
    /// factorization keeps the layout of its configuration, so its
    /// substitutions share it by construction.
    /// The vector operands and the pivot are one-dimensional and
    /// unaffected.
    MatrixLayout layout = MatrixLayout::RowMajor;
};



} // namespace tdls



#endif // TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP
