#ifndef TDLS_SOLVERS_COOPERATIVE_LUPP_CONFIG_HPP
#define TDLS_SOLVERS_COOPERATIVE_LUPP_CONFIG_HPP



/// \file
/// \brief Compile-time configuration of the CooperativeLUpp solver family.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every knob is a member of the CooperativeLUppConfig aggregate. A
/// constexpr instance of it is passed as the `Config` non-type template
/// argument of the CooperativeLUpp solvers. The default-constructed value
/// carries the defaults; a caller overrides individual knobs with
/// designated initializers, in declaration order:
///
///   constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3};
///
/// The aggregate is a structural type, as class-type non-type template
/// parameters require: every member is public, and two configurations
/// with equal members name the same solver instantiation. The
/// floating-point knobs are held as tdls::StructuralReal values: exact,
/// built and read back implicitly, and admissible as a template argument
/// on every compiler supporting class-type template parameters (a plain
/// floating-point member would need P1907R1, absent from GCC 10 and from
/// every nvcc before CUDA 13.0).



#include <limits>

#include <tdls/core/macros.hpp>
#include <tdls/core/structural_real.hpp>
#include <tdls/solvers/options.hpp>



namespace tdls {



/// \brief Compile-time knobs of the CooperativeLUpp solvers, carrying the
/// defaults.
/// \tparam T scalar type (float, double or long double)
template<typename T>
struct CooperativeLUppConfig {

    /// Number of matrix rows held by each thread of the group. A system
    /// of dimension N is solved by ceil(N / rows_per_thread) threads, and
    /// each thread keeps its rows in registers. This is the main
    /// performance axis of the solver: a larger value packs more systems
    /// per warp and wastes fewer lanes, at the price of more registers per
    /// thread. The default 1 is the original mapping of MAGMA, one row per
    /// thread. When N is not a multiple of rows_per_thread, the last
    /// thread holds phantom rows, skipped at compile time. A value
    /// reaching N gives one thread per system, the setting of sequential
    /// CPU code, and a larger value acts as N. Requires
    /// rows_per_thread >= 1.
    int rows_per_thread = 1;

    /// Row interchanges of the partial pivoting. Logical, the default, is
    /// the scheme of MAGMA: rows never move, and each thread tracks the
    /// position of its rows in the pivoted order. Physical is the scheme of
    /// LAPACK: rows are exchanged between the threads, so that the row at
    /// position k always lives in slot k / threads_per_system of thread
    /// k % threads_per_system. The owner of every row of U is then known in
    /// advance: the pivot search publishes one candidate per thread instead
    /// of one magnitude per row, and each step of the factorization takes
    /// two barriers instead of three. In exchange, a column whose pivot is
    /// not in place moves two whole rows through the workspace. Which
    /// scheme is faster depends on the device and on how often the
    /// matrices pivot. Both choose the same pivots and run the same
    /// operations. On a matrix they do not declare singular, their
    /// solutions are bitwise identical when no multiply-add is fused. GPU
    /// compilers fuse them by default, possibly differently in the two
    /// schemes: the solutions may then differ in the last bits. The
    /// factored matrix, the pivot array and the workspace size differ: see
    /// the solvers.
    RowInterchange row_interchange = RowInterchange::Logical;

    /// Relative threshold of the pivot choice, in (0, 1]. At each column,
    /// the row in place keeps the pivot when its magnitude reaches both
    /// this fraction of the largest magnitude of the column and the
    /// singularity floor. Otherwise the largest magnitude wins, the first
    /// one on ties. The default 1 is the partial pivoting of LAPACK, and
    /// the solvers then compile the test out. A smaller value saves row
    /// interchanges, at the price of multipliers bounded by its inverse
    /// instead of 1. The threshold is relative: unlike the absolute
    /// oot_pivot_threshold of TiledLUpp, a value such as 1e-10 would
    /// practically disable the pivoting. The solvers enforce the range at
    /// compile time.
    StructuralReal<T> relative_pivot_threshold = T(1);

    /// Singularity floor: the factorization is declared singular when the
    /// best pivot of a column falls below it. `numeric_limits<T>::min()`
    /// rejects only a zero or subnormal pivot, a genuine structural
    /// singularity, with the same criterion as the TiledLUpp solvers. It
    /// must be finite and positive, a zero floor letting a zero pivot
    /// through; the solver enforces both contracts at compile time.
    StructuralReal<T> singular_floor = std::numeric_limits<T>::min();

    /// Unroll policy, applied through a two-branch `if constexpr` (the
    /// pragma dialect itself lives in core/macros.hpp). The knob does not
    /// govern every loop on purpose: only the loops where offering the
    /// choice can noticeably change the performance, those indexing the
    /// register rows (the column sweeps of the factorization and of the
    /// substitutions, the loops along a row, the loads and stores). true:
    /// they carry a forced-unroll pragma, the guard that keeps the rows in
    /// registers on GPU backends, where a rolled loop indexes them
    /// dynamically and demotes them to slow local memory. false: no unroll
    /// pragma anywhere. On CPU, where a whole system held by one thread
    /// does not fit the registers anyway, false gives smaller code, faster
    /// builds and measured faster solves. The pivot search reads the
    /// workspace, not the registers, and carries no pragma in either
    /// branch; the compiler unrolls it on its own once the column sweep
    /// is unrolled. The runtime solver holds no register rows, carries no
    /// pragma and ignores this knob. Every TDLS family names this knob
    /// unroll_loops.
    bool unroll_loops = true;

    /// Memory layout of the matrix, in both residency modes. The knob only
    /// remaps the flat element index that the element stride scales: the
    /// arithmetic sequence is unchanged, so both layouts produce
    /// bitwise-identical results on identical inputs. Row-major is the
    /// TFEL convention and the default; column-major is the convention of
    /// MAGMA. The vector operands and the pivot are one-dimensional and
    /// unaffected.
    MatrixLayout layout = MatrixLayout::RowMajor;
};



} // namespace tdls



#endif // TDLS_SOLVERS_COOPERATIVE_LUPP_CONFIG_HPP
