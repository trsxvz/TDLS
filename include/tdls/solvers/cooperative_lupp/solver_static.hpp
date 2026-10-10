#ifndef TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_STATIC_HPP
#define TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_STATIC_HPP



/* The elimination and substitution kernels of this file are derived from
   MAGMA 2.10.0, magmablas/zgesv_batched_small.cu, function
   zgesv_batched_small_device (authors Azzam Haidar and Ahmad Abdelfattah).
   Those portions are distributed under the MAGMA license, reproduced
   below as required by it:

   Copyright (c) 2009-2023, The University of Tennessee
   All rights reserved.

     Redistribution and use in source and binary forms, with or without
     modification, are permitted provided that the following conditions
     are met:

     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of the University of Tennessee, Knoxville nor the
       names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

     This software is provided by the copyright holders and contributors
     ``as is'' and any express or implied warranties, including, but not
     limited to, the implied warranties of merchantability and fitness for
     a particular purpose are disclaimed. In no event shall the copyright
     holders or contributors be liable for any direct, indirect, incidental,
     special, exemplary, or consequential damages (including, but not
     limited to, procurement of substitute goods or services; loss of use,
     data, or profits; or business interruption) however caused and on any
     theory of liability, whether in contract, strict liability, or tort
     (including negligence or otherwise) arising in any way out of the use
     of this software, even if advised of the possibility of such damage.
*/



/// \file
/// \brief CooperativeLUpp solver: LU with partial pivoting held in the
/// registers of a group of threads, one system per group.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
/// \copyright The elimination and substitution kernels of this file are
/// derived from MAGMA 2.10.0 (Univ. of Tennessee, Knoxville; Univ. of
/// California, Berkeley; Univ. of Colorado, Denver), file
/// magmablas/zgesv_batched_small.cu, function zgesv_batched_small_device,
/// authors Azzam Haidar and Ahmad Abdelfattah. Those portions remain
/// subject to the MAGMA license (BSD 3-Clause), reproduced in full in the
/// comment at the top of this file as required by it.
///
/// One system is solved by a group of threads_per_system threads. The
/// solver factors S, the stored matrix read column-major: S = A under the
/// column-major layout, S = A^T under the row-major one, the default.
/// Thread tx holds rows tx, tx + threads_per_system,
/// tx + 2 * threads_per_system and so on of S, rows_per_thread of them,
/// in registers: rows of A under the column-major layout, columns of A
/// under the row-major one. In both layouts the threads of a group thus
/// read consecutive addresses, the access pattern the design of MAGMA
/// rests on. The threads cooperate through a small workspace (shared
/// memory on GPU) and a barrier, deduced or provided by the caller. The
/// factorization is the right-looking LU with partial pivoting of
/// MAGMA's register-blocked small-system kernel, applied to S: same
/// operations, same order and, with Config.relative_pivot_threshold = 1,
/// same pivot choice; a smaller threshold keeps the row in place more
/// often, and the pivots then differ. Under the column-major layout,
/// P A = L U and the solve is MAGMA's: L y = P b folded into the
/// factorization, then U x = y. Under the row-major layout, P A^T = L U,
/// so A = U^T L^T P: the pivoting interchanges columns of A, which is
/// partial pivoting of A^T and as stable, and the solve is the mirror of
/// MAGMA's: U^T z = b folded into the factorization, then L^T w = z,
/// which divides by nothing since L has a unit diagonal, and x = P^T w.
/// The rest of this documentation speaks of the rows of S. By default
/// pivoting is logical: rows never move, each thread tracks the position
/// of its rows in the pivoted order (MAGMA's rowid) and the pivot array
/// records it.
/// Config.row_interchange selects physical row interchanges instead,
/// the scheme of LAPACK: the rows move between the threads, so that the
/// row at position k always lives in slot k / threads_per_system of
/// thread k % threads_per_system. Both schemes choose the same pivots
/// and run the same operations in the same order. On a matrix they do
/// not declare singular, their solutions are bitwise identical, up to
/// the multiply-adds a compiler may fuse differently in each (see
/// CooperativeLUppConfig::row_interchange). The mapping of rows to
/// threads changes nothing in the arithmetic either, so every value of
/// rows_per_thread produces bitwise-identical results on identical
/// inputs. Config.relative_pivot_threshold below 1 relaxes the pivot
/// choice, the same way in both schemes. The runtime dimension variant
/// is CooperativeLUppSolverDynamic (solver_dynamic.hpp).
///
/// **Calling convention.** Every thread of the group calls the same entry
/// point, with the same template arguments, the same operands, the same
/// workspace and its own rank tx in [0, threads_per_system). The solver
/// never sees a global thread index: callers pre-offset every remote
/// pointer with the system index and pass a runtime stride, as for the
/// TiledLUpp solvers.
///
/// **Barrier.** The last argument, `sync`, is the barrier of the group,
/// and most calls omit it. By default, tdls::AutoSync, the solver deduces
/// it from the compilation target: nothing with one thread per system;
/// on GPU, the lanes of the group within their warp or wavefront,
/// checked on entry; on CPU, a barrier held in the workspace. A caller
/// passes its own barrier only where the compiler asks for one, or for a
/// group that the deduced barrier does not know, such as a group larger
/// than a warp. Any callable taking no argument serves, invoked by every
/// thread of the group, that makes the memory writes of each thread
/// visible to all of them, in the workspace and in the operands: a block,
/// SYCL group, Kokkos team or CPU thread barrier. It is then used as is,
/// without any check. make_sync returns the deduced barrier, for a caller
/// that also needs it for its own exchanges. tdls::NoSync, no barrier at
/// all, is refused at compile time for a group of several threads.
/// core/group.hpp details these cases, and what the solver checks.
///
/// **Fixed barrier sequence.** Every entry point executes a sequence of
/// barriers set by N, rows_per_thread, the row interchanges and the entry
/// point alone: it never depends on the data, not even on a singular
/// verdict, which is returned at the end instead of by an early exit. A
/// barrier wider than the group (a whole warp, a work-group, a team) is
/// therefore valid as well, provided every thread of its scope makes the
/// same sequence of calls.
/// Between two barriers, no workspace element is written by one thread
/// while another thread accesses it, so the solver is free of data races
/// even when the threads of a group do not run in lockstep.
///
/// **Inputs on entry.** Each thread reads only the entries of its own
/// rows of S and of the right-hand side: the matrix entries
/// S(tx + K * threads_per_system, c) and the right-hand-side entries
/// tx + K * threads_per_system. A caller that builds them in that thread
/// needs no barrier before the call: under the row-major layout, the
/// columns of A of the thread.
///
/// **Results on return.** Every entry point ends with a barrier. When a
/// thread returns, every thread of the group has written its results
/// (factored rows, pivot entries, solution entries), and they are
/// visible to the whole group, whatever the residency of the operands: a
/// thread may read the solution entries of the others at once. The
/// workspace is free again on return as well, so the caller may reuse it
/// immediately, for instance to exchange data between the threads.
///
/// **Residency.** Each operand has an *internal* mode, selected by the
/// `internal_rhs` / `internal_piv` / `internal_matrix` template booleans:
/// the operand is then the slice of the calling thread, a plain
/// caller-local array, and the stride is ignored. In *external* mode the
/// operand is the whole object, in memory reachable by every thread of
/// the group, walked with the stride.
///   - matrix: external element (r, c) of A lives at
///     `A[((r)*N+(c))*A_stride]` under the default row-major layout,
///     `A[((c)*N+(r))*A_stride]` under the column-major one. Element
///     (v, c) of S thus lives at `A[((c)*N+(v))*A_stride]` in both. The
///     internal slice holds the rows_per_thread rows of S of the thread:
///     element c of row tx + K * threads_per_system lives at
///     `A[c*rows_per_thread+K]`. It is the block of A made of these
///     vectors, stored in the layout of Config.layout: the
///     N x rows_per_thread block of the columns of the thread, row-major,
///     or the rows_per_thread x N block of its rows, column-major. With
///     one thread per system, the slice is the whole matrix in that
///     layout.
///   - pivot: under logical row interchanges, external entry r holds the
///     position of physical row r of S in the pivoted order. This maps
///     physical to logical rows, the inverse of the TiledLUpp convention.
///     Under physical ones, entry k holds the original index of the row of
///     S at position k, as in the TiledLUpp solvers. Under the row-major
///     layout, the rows of S are the columns of A. The internal slice holds
///     the entries of the rows of the thread, entry K for row
///     tx + K * threads_per_system.
///   - right-hand side and solution: external entry r is row r; the
///     internal slice holds entry tx + K * threads_per_system at index K.
///
/// Offsets are computed in 32-bit arithmetic, with the bounds of the
/// TiledLUpp solvers. The single form `base[e*stride]` covers every
/// batched layout: AoS (stride 1), SoA (stride = batch size), AoSoA
/// (stride = W).
///
/// **Factored format.** S is factored in place, in the stored array.
/// Under logical row interchanges, the factorization stays in the
/// physical rows of S: physical row r holds, before the column of its
/// pivot step, the multipliers of L, and from that column on, its row of
/// U. Under physical ones, the rows end in the pivoted order, as in
/// LAPACK: row k holds the multipliers of L before column k, and its row
/// of U from column k on. Both formats hold the same rows, bit for bit,
/// at different places. Under the row-major layout, these rows of S are
/// the columns of the stored array. The diagonal holds the pivots
/// themselves, not their reciprocals. A factorization produced here must
/// be consumed by the substitution routines of this solver, under the
/// same configuration.
///
/// **Workspace.** Under logical row interchanges, 3 * N elements per
/// group, MAGMA's sB, sx and dsx, in this order. Under physical ones,
/// 2 * N + 2 * threads_per_system + 5 elements: the candidates of the
/// threads and their positions, the magnitude of the row in place, then
/// two exchange records of N + 2 elements, the rows that leave and
/// enter the position of the step. The substitutions use the first
/// 2 * N + 2 elements at most. Two more elements come first, those of
/// the deduced CPU barrier: on CPU, they must be zero before the first
/// call of a group of several threads, and the barrier keeps them in
/// that state between calls. They keep their place whatever the dimension and the
/// configuration, so that a workspace may serve other solves.
/// workspace_size counts them all. The workspace
/// pointer carries no restrict qualifier: the other threads of the
/// group write the workspace between two barriers, and a restrict
/// pointer would let the compiler reuse a value read before the
/// barrier. Every other operand keeps the qualifier, since each thread
/// only accesses its own rows of it.
///
/// **Departures from the MAGMA kernel.** The arithmetic is MAGMA's, but
/// for the relative threshold of the pivot choice; the other changes are
/// structural.
///   - rows_per_thread rows per thread instead of one, phantom rows
///     skipped at compile time, so that several systems share a warp.
///   - Device-callable entry points with a deduced or caller-provided
///     barrier, instead of a kernel with one system per block.
///   - The back substitution keeps the solution entry x_i in a register of
///     the thread owning entry i. MAGMA writes it to sB[i] in the very
///     barrier interval where every thread reads sB[i]: correct in
///     lockstep execution only, a data race otherwise.
///   - Under logical row interchanges, the pivot output is the logical
///     position of each row (rowid), not the LAPACK swap sequence, and the
///     matrix keeps its physical row order: a separate substitution can
///     then rebuild its state from the factored rows alone. The swap
///     sequence (sipiv) is not computed.
///   - Under the row-major layout, MAGMA's factorization of A^T and the
///     mirror solve U^T z = b, L^T w = z, x = P^T w: the threads hold
///     columns of A, so that a group still reads consecutive addresses.
///     Each substitution step takes one barrier instead of two, the
///     exchange being double-buffered.
///   - Physical row interchanges as an option, with the same pivots and
///     the same operations: one candidate per thread instead of one
///     magnitude per row, two barriers per column instead of three, and no
///     row moved while the pivot is in place. Under them a zero pivot row
///     is not scaled to zero: the factorization of a singular matrix is
///     unspecified, and a pivot reciprocal of 1 keeps it finite.
///   - A relative threshold of the pivot choice, threshold pivoting: 1
///     keeps the choice of MAGMA and LAPACK, a smaller value keeps the
///     row in place more often, and the pivots and the results then
///     differ.
///   - A pivot below Config.singular_floor is singular, instead of an
///     exactly zero one. The default floor only adds the subnormal pivots,
///     the criterion of every TDLS solver.
///   - Real scalar types: the magnitude of the pivot search is the
///     absolute value, as in the real-precision variants of MAGMA.



#include <type_traits>
#include <utility>

#include <tdls/core/group.hpp>
#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/solvers/cooperative_lupp/config.hpp>



// Older gcc releases cannot prove that the rows of a thread are loaded
// before being read: the load of a phantom-capable row slot is guarded by
// the row index and its later uses by rowid, which starts equal to it and
// stays at or above N for a phantom row. The resulting
// -Wmaybe-uninitialized reports are spurious. Under UBSan instrumentation
// at -O2 and above, gcc additionally loses the value ranges of the fully
// unrolled loop indices and reports impossible subscripts through
// -Warray-bounds, exactly as in the TiledLUpp headers; the exercised paths
// are certified free of out-of-bounds accesses by the constexpr suite.
// Both suppressions are scoped to this header and to those warnings only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif



// clang reports a forced unrolling that the optimizer could not perform
// through -Wpass-failed. The unrolling requested by TDLS_UNROLL_FORCE is
// a performance hint: a failed hint does not affect correctness. The
// suppression is scoped to this header and to that warning only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif



// nvc++ reports through no_device_stack, under the managed memory of
// -stdpar=gpu, every lambda that captures local objects by reference:
// such a lambda handed to a parallel algorithm would read the host stack.
// The lambdas of this header are called on the spot, on the stack of the
// thread that runs the solver: the report is a false positive. The
// suppression is scoped to this header and to that warning only.
#if defined(__NVCOMPILER)
#pragma diag_push
#pragma diag_suppress no_device_stack
#endif

namespace tdls {



namespace detail {

/// \brief Calls fn once per index of the sequence, in increasing order,
/// with the index as a std::integral_constant.
/// \tparam Fn callable type
/// \tparam Ks indices
/// \param[in] fn callable, invoked as fn(std::integral_constant<int, K>{})
template<typename Fn, int... Ks>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void
unroll_K_impl(Fn& fn, std::integer_sequence<int, Ks...>) noexcept {
    (fn(std::integral_constant<int, Ks>{}), ...);
}

/// \brief Compile-time iteration over the row slots K = 0 .. R-1 of a
/// thread.
///
/// Inside fn, K is a constant expression, unlike the index of a runtime
/// loop: the slot classification goes through `if constexpr`, and the
/// register rows rA[K] are indexed at compile time even where no unroll
/// pragma applies. The solver passes it only lambdas that never call the
/// barrier, the one operation that may throw, hence noexcept.
/// \tparam R  number of slots
/// \tparam Fn callable type
/// \param[in] fn callable, invoked as fn(std::integral_constant<int, K>{})
template<int R, typename Fn>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void unroll_K(Fn&& fn) noexcept {
    unroll_K_impl(fn, std::make_integer_sequence<int, R>{});
}

} // namespace detail



/* Addressing macros. They expand inside member functions where the
   parameter names (tx, A, A_stride, piv, piv_stride, b, x, y,
   rhs_stride), the residency template booleans and the class constants
   are in scope. K is a row slot of the calling thread, its row is
   tx + K * threads_per_system. #undef'd at the end of this header. */

/// \def TDLS_COOP_LUPP_A
/// \brief Element (row of slot K, c) of S, the stored matrix read
/// column-major (A under the column-major layout, A^T under the row-major
/// one): slice of the calling thread under internal residency, whole
/// strided matrix otherwise. Consecutive threads hold consecutive rows of
/// S, thus consecutive addresses.
#define TDLS_COOP_LUPP_A(K, c)                                                                     \
    A[internal_matrix ? unsigned((c) * rows_per_thread + (K))                                      \
                      : unsigned((c) * N + tx + (K) * threads_per_system) * unsigned(A_stride)]
/// \def TDLS_COOP_LUPP_PIV
/// \brief Pivot entry of the row of slot K: slice of the calling thread
/// under internal residency, whole strided array otherwise.
#define TDLS_COOP_LUPP_PIV(K)                                                                      \
    piv[internal_piv ? unsigned(K) : unsigned(tx + (K) * threads_per_system) * unsigned(piv_stride)]
/// \def TDLS_COOP_LUPP_B
/// \brief Right-hand-side entry of the row of slot K: slice of the calling
/// thread under internal residency, whole strided vector otherwise.
#define TDLS_COOP_LUPP_B(K)                                                                        \
    b[internal_rhs ? unsigned(K) : unsigned(tx + (K) * threads_per_system) * unsigned(rhs_stride)]
/// \def TDLS_COOP_LUPP_X
/// \brief Solution entry of the row of slot K, addressed as
/// TDLS_COOP_LUPP_B.
#define TDLS_COOP_LUPP_X(K)                                                                        \
    x[internal_rhs ? unsigned(K) : unsigned(tx + (K) * threads_per_system) * unsigned(rhs_stride)]
/// \def TDLS_COOP_LUPP_Y
/// \brief Entry of the row of slot K of the right-hand side solved in
/// place, addressed as TDLS_COOP_LUPP_B.
#define TDLS_COOP_LUPP_Y(K)                                                                        \
    y[internal_rhs ? unsigned(K) : unsigned(tx + (K) * threads_per_system) * unsigned(rhs_stride)]



/// \brief Dense LU factorization with partial pivoting held in the
/// registers of a group of threads, solving one NxN system per group.
///
/// All entry points are static, host- and device-callable, called by
/// every thread of the group with its rank tx, and take raw pointers
/// pre-offset by the caller plus one runtime stride per array (see the
/// file documentation for the calling convention). Entry points:
///   - factorize:             S := P*L*U in place, pivot entries out (S
///     = A column-major, A^T row-major)
///   - substitute:            x := A^-1 b from a factorization (b and x
///     distinct)
///   - substitute_canonical:  idem with b = e_col (tangent-operator columns)
///   - substitute_inplace:    idem with b == x
///   - solve:                 factorize + substitute, the rows staying in
///     registers in between
///   - solve_inplace:         factorize with the forward pass folded in,
///     the original MAGMA kernel
///
/// factorize/substitute stay separate so a factorization can be reused
/// across several right-hand sides. solve, solve_inplace and factorize
/// followed by substitute produce bitwise-identical results.
///
/// The residency template booleans (internal_rhs / internal_piv /
/// internal_matrix) deliberately have no default: they describe what the
/// passed buffers ARE, so each call site must state them explicitly.
///
/// \tparam T      scalar type (float, double or long double)
/// \tparam N      system dimension (N >= 1)
/// \tparam Config compile-time knobs, passed as a constexpr value: rows per
///         thread, row interchanges, relative pivot threshold, singularity
///         floor, unroll policy, matrix layout; see CooperativeLUppConfig
template<typename T, int N, CooperativeLUppConfig<T> Config = CooperativeLUppConfig<T>{}>
struct CooperativeLUppSolverStatic {

    static_assert(N >= 1, "CooperativeLUppSolverStatic: N must be >= 1");
    static_assert(Config.rows_per_thread >= 1,
                  "CooperativeLUppSolverStatic: rows_per_thread must be >= 1");
    static_assert(Config.relative_pivot_threshold.is_finite() && Config.singular_floor.is_finite(),
                  "CooperativeLUppSolverStatic: relative_pivot_threshold and singular_floor must "
                  "be finite (and fit a 63-bit mantissa)");

    /// \brief Rows held by each thread: Config.rows_per_thread, or N when
    /// it is larger. A value below 1, rejected above, is clamped to 1 so
    /// that the rejection stays the only diagnostic.
    static constexpr int rows_per_thread = Config.rows_per_thread < 1   ? 1
                                           : Config.rows_per_thread < N ? Config.rows_per_thread
                                                                        : N;
    /// \brief Threads solving one system: ceil(N / rows_per_thread).
    static constexpr int threads_per_system = (N + rows_per_thread - 1) / rows_per_thread;
    /// \brief Elements of the workspace shared by the threads of a group:
    /// the two elements of the deduced CPU barrier, then 3 * N under
    /// logical row interchanges, 2 * N + 2 * threads_per_system + 5 under
    /// physical ones.
    static constexpr int workspace_size =
        detail::barrier_elements + (Config.row_interchange == RowInterchange::Logical
                                        ? 3 * N
                                        : 2 * N + 2 * threads_per_system + 5);
    /// \brief Relative threshold of the pivot choice, read once from the
    /// configuration (see CooperativeLUppConfig::relative_pivot_threshold).
    static constexpr T relative_pivot_threshold = Config.relative_pivot_threshold;
    /// \brief Singularity floor, read once from the configuration (see
    /// CooperativeLUppConfig::singular_floor).
    static constexpr T singular_floor = Config.singular_floor;
    /// \brief Whether the solve uses the transpose of the factored matrix.
    /// The solver factors S, the stored matrix read column-major, so that
    /// consecutive threads hold consecutive addresses whatever the layout.
    /// Under the column-major layout S = A and the solve is S x = b, the
    /// kernel of MAGMA. Under the row-major layout S = A^T, the threads hold
    /// columns of A, and the solve is S^T x = b (see the file
    /// documentation).
    static constexpr bool transposed = Config.layout == MatrixLayout::RowMajor;

    static_assert(relative_pivot_threshold > T(0) && relative_pivot_threshold <= T(1),
                  "CooperativeLUppSolverStatic: relative_pivot_threshold must be in (0, 1]");
    static_assert(singular_floor > T(0),
                  "CooperativeLUppSolverStatic: singular_floor must be positive");

    /* =====================================================================
       Row slots.
       Slot K of thread tx holds row tx + K * threads_per_system. When N is
       not a multiple of rows_per_thread, the last slots hold phantom rows
       (index >= N) for some or all threads. The classification is a
       compile-time function of K: a slot without any real row is skipped
       entirely, a slot full for every thread needs no guard, and only a
       mixed slot tests its row at run time.
       ===================================================================== */

    /// \brief Whether slot K holds a real row for at least one thread.
    /// \param[in] K row slot
    /// \return true when some thread of the group has a real row in slot K
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool slot_has_rows(const int K) noexcept {
        return K * threads_per_system < N;
    }

    /// \brief Whether slot K holds a real row for every thread.
    /// \param[in] K row slot
    /// \return true when every thread of the group has a real row in slot K
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool slot_is_full(const int K) noexcept {
        return (K + 1) * threads_per_system <= N;
    }

    /// \brief Calls fn on the slot of static index pos when the calling
    /// thread holds it: the position of a row under physical row
    /// interchanges, the entry of the right-hand side under the transposed
    /// solve.
    ///
    /// Index pos lives in slot pos / threads_per_system of thread
    /// pos % threads_per_system. Inside an unrolled column sweep, a column
    /// index is a constant: the slot is then selected at compile time, and
    /// only the rank of the thread is tested at run time. The solver passes
    /// it only callables that never call the barrier, hence noexcept.
    /// \tparam Fn callable type
    /// \param[in] tx  rank of the calling thread in its group
    /// \param[in] pos position, in [0, N)
    /// \param[in] fn  callable, invoked as fn(std::integral_constant<int, K>{})
    template<typename Fn>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void at_position(const int tx, const int pos,
                                                                        Fn&& fn) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (K == pos / threads_per_system && tx == pos % threads_per_system) fn(K_tag);
            }
        });
    }

    /* =====================================================================
       Operand movement.
       Each thread reads and writes only the entries of its own rows, so
       the operands themselves never carry data between threads: the
       workspace does.
       ===================================================================== */

    /// \brief Load the rows of the calling thread into registers.
    /// \tparam internal_matrix residency of A
    /// \param[in]  tx       rank of the calling thread in its group
    /// \param[in]  A        matrix, or slice of the thread (caller-pre-offset)
    /// \param[in]  A_stride element stride of A (external mode)
    /// \param[out] rA       register rows of the thread
    template<bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_rows(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
              T (&rA)[rows_per_thread][N]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) {
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int c = 0; c < N; ++c)
                            rA[K][c] = TDLS_COOP_LUPP_A(K, c);
                    } else {
                        for (int c = 0; c < N; ++c)
                            rA[K][c] = TDLS_COOP_LUPP_A(K, c);
                    }
                }
            }
        });
    }

    /// \brief Store the register rows of the calling thread.
    /// \tparam internal_matrix residency of A
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in,out] A        matrix, or slice of the thread
    ///                (caller-pre-offset)
    /// \param[in]     A_stride element stride of A (external mode)
    /// \param[in]     rA       register rows of the thread
    template<bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_rows(const int tx, T* TDLS_RESTRICT A, const int A_stride,
               const T (&rA)[rows_per_thread][N]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) {
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int c = 0; c < N; ++c)
                            TDLS_COOP_LUPP_A(K, c) = rA[K][c];
                    } else {
                        for (int c = 0; c < N; ++c)
                            TDLS_COOP_LUPP_A(K, c) = rA[K][c];
                    }
                }
            }
        });
    }

    /// \brief Initial pivot entries: no row has moved yet, so the entry of
    /// each slot is the index of its row, phantom rows included (theirs
    /// stays at or above N). Under logical row interchanges the entry is the
    /// position of the row of the slot, under physical ones the original
    /// index of the row held by the slot: both start there.
    /// \param[in]  tx   rank of the calling thread in its group
    /// \param[out] rpiv pivot entry of each slot of the thread
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    init_piv(const int tx, int (&rpiv)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            rpiv[K]         = tx + K * threads_per_system;
        });
    }

    /// \brief Load the pivot entries recorded by factorize.
    /// \tparam internal_piv residency of piv
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  piv        pivot entries, or slice of the thread
    ///             (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[out] rpiv       pivot entry of each slot of the thread
    template<bool internal_piv>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_piv(const int tx, const int* TDLS_RESTRICT piv, const int piv_stride,
             int (&rpiv)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if (slot_is_full(K) || tx + K * threads_per_system < N)
                rpiv[K] = TDLS_COOP_LUPP_PIV(K);
            else
                rpiv[K] = tx + K * threads_per_system;
        });
    }

    /// \brief Record the pivot entries of the calling thread.
    /// \tparam internal_piv residency of piv
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[out] piv        pivot entries, or slice of the thread
    ///             (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  rpiv       pivot entry of each slot of the thread
    template<bool internal_piv>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_piv(const int tx, int* TDLS_RESTRICT piv, const int piv_stride,
              const int (&rpiv)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N)
                    TDLS_COOP_LUPP_PIV(K) = rpiv[K];
            }
        });
    }

    /// \brief Load the right-hand-side entries of the rows of the calling
    /// thread.
    /// \tparam internal_rhs residency of b
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  b          right-hand side, or slice of the thread
    ///             (caller-pre-offset)
    /// \param[in]  rhs_stride element stride of b (external mode)
    /// \param[out] rB         right-hand-side registers of the thread
    template<bool internal_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_rhs(const int tx, const T* TDLS_RESTRICT b, const int rhs_stride,
             T (&rB)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) rB[K] = TDLS_COOP_LUPP_B(K);
            }
        });
    }

    /// \brief Store the solution entries held by the calling thread.
    /// \tparam internal_rhs residency of x
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[out] x          solution, or slice of the thread
    ///             (caller-pre-offset)
    /// \param[in]  rhs_stride element stride of x (external mode)
    /// \param[in]  rB         solution registers of the thread
    template<bool internal_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_solution(const int tx, T* TDLS_RESTRICT x, const int rhs_stride,
                   const T (&rB)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) TDLS_COOP_LUPP_X(K) = rB[K];
            }
        });
    }

    /* =====================================================================
       FACTORIZATION, right-looking, one column per step, as in MAGMA.
       Logical row interchanges, three barrier intervals per column: the
       magnitudes of the column are published at the positions of their
       rows, every thread finds the same pivot, the owner of the pivot row
       publishes it, and every thread eliminates its own rows below the
       pivot. Physical row interchanges, two barrier intervals per column:
       each thread publishes its best candidate, every thread finds the
       same pivot, the owner of position i publishes its row, the owner of
       the pivot row as well when the pivot is not in place, and every
       thread eliminates its rows below the pivot after the exchange. The
       pivot, the operations and their order are the same in both schemes.
       ===================================================================== */

    /// \brief Relative threshold of the pivot choice: whether the row in
    /// place keeps the pivot (see
    /// CooperativeLUppConfig::relative_pivot_threshold).
    /// \param[in] in_place   magnitude of the row in place in the column
    /// \param[in] column_max largest magnitude of the column
    /// \return true when the row in place keeps the pivot
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    keeps_pivot(const T in_place, const T column_max) noexcept {
        const T bound = relative_pivot_threshold * column_max;
        return in_place >= (bound > singular_floor ? bound : singular_floor);
    }

    /// \brief One column of the factorization under logical row
    /// interchanges.
    ///
    /// With fuse_rhs, the step applies the forward pass to the right-hand
    /// side as well. Solve S x = b (column-major): the owner of the pivot
    /// row publishes its entry y_i and the rows below subtract L(r, i) y_i,
    /// as in MAGMA. Solve S^T x = b (row-major), U^T z = b: the owner of
    /// entry i publishes b'(i), final at this step, every thread computes
    /// z_i = b'(i) / U(i, i) through the reciprocal of the pivot, the owner
    /// of entry i keeps z_i in its place, and each entry c > i subtracts
    /// U(i, c) z_i, row i of U being the published pivot row.
    /// \tparam fuse_rhs apply the forward pass to the right-hand side as well
    ///         (solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     tx    rank of the calling thread in its group
    /// \param[in]     i     column of this step
    /// \param[in,out] rA    register rows of the thread
    /// \param[in,out] rB    right-hand-side registers (fuse_rhs only)
    /// \param[in,out] rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     sB    workspace: broadcast slot of the right-hand-side
    ///                entry of the step (fuse_rhs only)
    /// \param[in]     sx    workspace: pivot row
    /// \param[in]     dsx   workspace: magnitudes of the column
    /// \param[in,out] linfo 0, or 1 + the first column whose best pivot
    ///                fell below singular_floor
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    logical_elimination_step([[maybe_unused]] const int tx, const int i,
                             T (&rA)[rows_per_thread][N], [[maybe_unused]] T (&rB)[rows_per_thread],
                             int (&rowid)[rows_per_thread], T* sB, T* sx, T* dsx, int& linfo,
                             Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        // izamax: the magnitudes of column i, at the positions of their rows.
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || rowid[K] < N) dsx[rowid[K]] = detail::abs(rA[K][i]);
            }
        });
        sync();

        // Every thread scans the same magnitudes and finds the same pivot.
        // The scan reads the workspace, not the registers: no pragma.
        T rx_abs_max = dsx[i];
        int max_id   = i;
        for (int j = i + 1; j < N; j++) {
            if (dsx[j] > rx_abs_max) {
                max_id     = j;
                rx_abs_max = dsx[j];
            }
        }
        // Below the default threshold 1, the row in place keeps the pivot
        // when it is large enough.
        if constexpr (relative_pivot_threshold < T(1)) {
            if (keeps_pivot(dsx[i], rx_abs_max)) max_id = i;
        }
        const bool zero_pivot = (rx_abs_max < singular_floor);
        linfo                 = (zero_pivot && linfo == 0) ? (i + 1) : linfo;
        const T update        = zero_pivot ? T(0) : T(1);

        // The owner of the pivot row publishes it; positions i and max_id
        // are swapped. A phantom row never matches: its position is >= N.
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (rowid[K] == max_id) {
                    rowid[K] = i;
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int j = i; j < N; j++)
                            sx[j] = update * rA[K][j];
                    } else {
                        for (int j = i; j < N; j++)
                            sx[j] = update * rA[K][j];
                    }
                    if constexpr (fuse_rhs && !transposed) sB[0] = rB[K];
                } else if (rowid[K] == i) {
                    rowid[K] = max_id;
                }
            }
        });
        if constexpr (fuse_rhs && transposed)
            at_position(tx, i, [&](auto K_tag) { sB[0] = rB[decltype(K_tag)::value]; });
        sync();

        const T reg            = zero_pivot ? T(1) : T(1) / sx[i];
        [[maybe_unused]] T z_i = T(0);
        if constexpr (fuse_rhs && transposed) z_i = sB[0] * reg;

        // scal and ger: every thread eliminates its rows below the pivot.
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if ((slot_is_full(K) || rowid[K] < N) && rowid[K] > i) {
                    rA[K][i] *= reg;
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int j = i + 1; j < N; j++)
                            rA[K][j] -= rA[K][i] * sx[j];
                    } else {
                        for (int j = i + 1; j < N; j++)
                            rA[K][j] -= rA[K][i] * sx[j];
                    }
                    if constexpr (fuse_rhs && !transposed) rB[K] -= rA[K][i] * sB[0];
                }
                if constexpr (fuse_rhs && transposed) {
                    const int c = tx + K * threads_per_system;
                    if (c == i) rB[K] = z_i;
                    if ((slot_is_full(K) || c < N) && c > i) rB[K] -= sx[c] * z_i;
                }
            }
        });
        sync();
    }

    /// \brief Copy a register row to an exchange record of the workspace:
    /// its N entries, its right-hand-side entry (fuse_rhs only), then its
    /// original index. The index travels as a value of T, exact since the
    /// offset bounds keep N far below the mantissa range of every scalar
    /// type.
    /// \tparam fuse_rhs carry the right-hand-side entry as well
    /// \param[in]  row    register row
    /// \param[in]  rhs    its right-hand-side entry (fuse_rhs only)
    /// \param[in]  origin its original index
    /// \param[out] record exchange record, N + 2 elements
    template<bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    save_row(const T (&row)[N], [[maybe_unused]] const T& rhs, const int origin,
             T* record) noexcept {
        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int c = 0; c < N; c++)
                record[c] = row[c];
        } else {
            for (int c = 0; c < N; c++)
                record[c] = row[c];
        }
        if constexpr (fuse_rhs) record[N] = rhs;
        record[N + 1] = static_cast<T>(origin);
    }

    /// \brief Copy an exchange record of the workspace to a register row,
    /// the converse of save_row.
    /// \tparam fuse_rhs carry the right-hand-side entry as well
    /// \param[out] row    register row
    /// \param[out] rhs    its right-hand-side entry (fuse_rhs only)
    /// \param[out] origin its original index
    /// \param[in]  record exchange record, N + 2 elements
    template<bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_row(T (&row)[N], [[maybe_unused]] T& rhs, int& origin, const T* record) noexcept {
        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int c = 0; c < N; c++)
                row[c] = record[c];
        } else {
            for (int c = 0; c < N; c++)
                row[c] = record[c];
        }
        if constexpr (fuse_rhs) rhs = record[N];
        origin = static_cast<int>(record[N + 1]);
    }

    /// \brief Whether slot K of the calling thread holds a real row that
    /// may sit below column i: the slots a pivot row below the diagonal can
    /// come from, a compile-time choice in an unrolled column sweep but for
    /// the rank test of a mixed slot.
    /// \param[in] tx rank of the calling thread in its group
    /// \param[in] K  row slot
    /// \param[in] i  column of the step
    /// \return true when slot K may hold a row below position i
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool below(const int tx, const int K,
                                                                  const int i) noexcept {
        return (K + 1) * threads_per_system > i + 1 &&
               (slot_is_full(K) || tx + K * threads_per_system < N);
    }

    /// \brief Copy the register row at position pos below column i, held by
    /// the calling thread in a slot known at run time only, to an exchange
    /// record, as save_row does.
    ///
    /// Each entry is selected over the slots, never read through the slot
    /// index: on GPU, a register array indexed at run time is demoted to
    /// local memory, and so is one that the compiler reaches through
    /// branches it merges into such an index.
    /// \tparam fuse_rhs carry the right-hand-side entry as well
    /// \param[in]  tx     rank of the calling thread in its group
    /// \param[in]  i      column of the step
    /// \param[in]  pos    position of the row, below i
    /// \param[in]  rA     register rows of the thread
    /// \param[in]  rB     right-hand-side registers (fuse_rhs only)
    /// \param[in]  origin original index of the row held by each slot
    /// \param[out] record exchange record, N + 2 elements
    template<bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    gather_row(const int tx, const int i, const int pos, const T (&rA)[rows_per_thread][N],
               [[maybe_unused]] const T (&rB)[rows_per_thread],
               const int (&origin)[rows_per_thread], T* record) noexcept {
        const int slot   = pos / threads_per_system;
        const auto entry = [&](const int c) {
            T v = T(0);
            detail::unroll_K<rows_per_thread>([&](auto K_tag) {
                constexpr int K = decltype(K_tag)::value;
                if constexpr (slot_has_rows(K)) {
                    if (below(tx, K, i)) v = K == slot ? rA[K][c] : v;
                }
            });
            return v;
        };
        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int c = 0; c < N; c++)
                record[c] = entry(c);
        } else {
            for (int c = 0; c < N; c++)
                record[c] = entry(c);
        }
        T rhs     = T(0);
        int index = 0;
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (below(tx, K, i)) {
                    if constexpr (fuse_rhs) rhs = K == slot ? rB[K] : rhs;
                    index = K == slot ? origin[K] : index;
                }
            }
        });
        if constexpr (fuse_rhs) record[N] = rhs;
        record[N + 1] = static_cast<T>(index);
    }

    /// \brief Copy an exchange record to the register row at position pos
    /// below column i, held by the calling thread in a slot known at run
    /// time only, the converse of gather_row: each slot takes the record or
    /// keeps its row through a select.
    /// \tparam fuse_rhs carry the right-hand-side entry as well
    /// \param[in]     tx     rank of the calling thread in its group
    /// \param[in]     i      column of the step
    /// \param[in]     pos    position of the row, below i
    /// \param[in,out] rA     register rows of the thread
    /// \param[in,out] rB     right-hand-side registers (fuse_rhs only)
    /// \param[in,out] origin original index of the row held by each slot
    /// \param[in]     record exchange record, N + 2 elements
    template<bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    scatter_row(const int tx, const int i, const int pos, T (&rA)[rows_per_thread][N],
                [[maybe_unused]] T (&rB)[rows_per_thread], int (&origin)[rows_per_thread],
                const T* record) noexcept {
        const int slot   = pos / threads_per_system;
        const auto entry = [&](const int c) {
            const T v = record[c];
            detail::unroll_K<rows_per_thread>([&](auto K_tag) {
                constexpr int K = decltype(K_tag)::value;
                if constexpr (slot_has_rows(K)) {
                    if (below(tx, K, i)) rA[K][c] = K == slot ? v : rA[K][c];
                }
            });
        };
        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int c = 0; c < N; c++)
                entry(c);
        } else {
            for (int c = 0; c < N; c++)
                entry(c);
        }
        [[maybe_unused]] const T rhs = fuse_rhs ? record[N] : T(0);
        const int index              = static_cast<int>(record[N + 1]);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (below(tx, K, i)) {
                    if constexpr (fuse_rhs) rB[K] = K == slot ? rhs : rB[K];
                    origin[K] = K == slot ? index : origin[K];
                }
            }
        });
    }

    /// \brief One column of the factorization under physical row
    /// interchanges.
    ///
    /// The row at position k lives in slot k / threads_per_system of thread
    /// k % threads_per_system. Each thread publishes its candidate: the
    /// largest magnitude of column i among its rows at or below position i,
    /// the first one on ties, with its position. Every thread finds the same
    /// column maximum and the same pivot as the logical scan, ties and NaN
    /// included: the row in place when it reaches the maximum, the first
    /// position holding it otherwise. The owner of position i publishes its
    /// row whole. When the pivot is not in place, the owner of the pivot
    /// row publishes it as well, and both rows are exchanged before the
    /// elimination, their multipliers included: the factored rows end in
    /// the pivoted order, as in LAPACK.
    ///
    /// With fuse_rhs, the forward pass of the right-hand side is that of the
    /// logical scheme on positions. Solve S x = b: the entry of each row
    /// travels with it. Solve S^T x = b: the right-hand side is indexed by
    /// the columns of S, which never move; the owner of position i is the
    /// owner of entry i, publishes b'(i) with its row and keeps z_i in its
    /// place.
    /// \tparam fuse_rhs apply the forward pass to the right-hand side as well
    ///         (solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     tx     rank of the calling thread in its group
    /// \param[in]     i      column of this step
    /// \param[in,out] rA     register rows of the thread
    /// \param[in,out] rB     right-hand-side registers (fuse_rhs only)
    /// \param[in,out] origin original index of the row held by each slot
    /// \param[in]     work   workspace of the group
    /// \param[in,out] linfo  0, or 1 + the first column whose best pivot
    ///                fell below singular_floor
    /// \param[in]     sync   barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    physical_elimination_step(const int tx, const int i, T (&rA)[rows_per_thread][N],
                              [[maybe_unused]] T (&rB)[rows_per_thread],
                              int (&origin)[rows_per_thread], T* work, int& linfo,
                              Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        // The right-hand side moves with the rows only under S x = b.
        constexpr bool carries_rhs = fuse_rhs && !transposed;
        // Workspace: the candidates and their positions, the magnitude of
        // the row in place, then the exchange records of the rows leaving
        // and entering position i. Positions travel as values of T, as in
        // save_row.
        T* const candidate = work;
        T* const where     = candidate + threads_per_system;
        T* const in_place  = where + threads_per_system;
        T* const leaving   = in_place + 1;
        T* const entering  = leaving + N + 2;

        // Candidate of the thread. The owner of position i starts from the
        // row in place, so that a NaN there stops the search, as in the
        // logical scan, and publishes its magnitude.
        T best       = T(-1);
        int best_pos = N;
        T local      = T(0);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                const int pos = tx + K * threads_per_system;
                if ((slot_is_full(K) || pos < N) && pos >= i) {
                    const T v = detail::abs(rA[K][i]);
                    if (pos == i || v > best) {
                        best     = v;
                        best_pos = pos;
                    }
                    if (pos == i) local = v;
                }
            }
        });
        candidate[tx] = best;
        where[tx]     = static_cast<T>(best_pos);
        if (tx == i % threads_per_system) in_place[0] = local;
        sync();

        // Every thread finds the same column maximum. The row in place keeps
        // the pivot when it reaches it, as in the logical scan, which keeps
        // the first maximum; otherwise the pivot is the first position
        // holding it. Only that rare case reads the positions. The loops
        // read the workspace, not the registers: no pragma.
        T rx_abs_max = candidate[i % threads_per_system];
        for (int t = 0; t < threads_per_system; t++)
            rx_abs_max = candidate[t] > rx_abs_max ? candidate[t] : rx_abs_max;
        bool keep = !(in_place[0] < rx_abs_max);
        if constexpr (relative_pivot_threshold < T(1)) keep = keeps_pivot(in_place[0], rx_abs_max);
        int max_id = i;
        if (!keep) {
            max_id = N;
            for (int t = 0; t < threads_per_system; t++) {
                const int pos = static_cast<int>(where[t]);
                if (candidate[t] == rx_abs_max && pos < max_id) max_id = pos;
            }
        }
        const bool zero_pivot = (rx_abs_max < singular_floor);
        linfo                 = (zero_pivot && linfo == 0) ? (i + 1) : linfo;

        // The owner of position i publishes its row whole, from a slot
        // known at compile time in an unrolled sweep. When the pivot is not
        // in place, the owner of position max_id publishes the pivot row as
        // well: it enters position i, and the row in place leaves it for
        // position max_id. The elimination reads the pivot row and its
        // right-hand-side entry from the record of the row that ends at
        // position i, under S x = b; under S^T x = b, the entry b'(i) of
        // the owner of position i, from the record of the row in place. A
        // zero pivot is not scaled away as in the logical scheme: the
        // factorization of a singular matrix is unspecified, and reg = 1
        // keeps it finite.
        at_position(tx, i, [&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            save_row<fuse_rhs>(rA[K], rB[K], origin[K], leaving);
        });
        if (max_id != i && tx == max_id % threads_per_system)
            gather_row<carries_rhs>(tx, i, max_id, rA, rB, origin, entering);
        sync();

        if (max_id != i) {
            at_position(tx, i, [&](auto K_tag) {
                constexpr int K = decltype(K_tag)::value;
                load_row<carries_rhs>(rA[K], rB[K], origin[K], entering);
            });
            if (tx == max_id % threads_per_system)
                scatter_row<carries_rhs>(tx, i, max_id, rA, rB, origin, leaving);
        }

        const T* const prow    = max_id == i ? leaving : entering;
        const T reg            = zero_pivot ? T(1) : T(1) / prow[i];
        [[maybe_unused]] T z_i = T(0);
        if constexpr (fuse_rhs && transposed) z_i = leaving[N] * reg;

        // scal and ger: every thread eliminates its rows below the pivot.
        // The next step writes only the candidates, which no thread reads
        // any more: no barrier closes the step.
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                const int pos = tx + K * threads_per_system;
                if ((slot_is_full(K) || pos < N) && pos > i) {
                    rA[K][i] *= reg;
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int j = i + 1; j < N; j++)
                            rA[K][j] -= rA[K][i] * prow[j];
                    } else {
                        for (int j = i + 1; j < N; j++)
                            rA[K][j] -= rA[K][i] * prow[j];
                    }
                    if constexpr (carries_rhs) rB[K] -= rA[K][i] * prow[N];
                }
                if constexpr (fuse_rhs && transposed) {
                    if (pos == i) rB[K] = z_i;
                    if ((slot_is_full(K) || pos < N) && pos > i) rB[K] -= prow[pos] * z_i;
                }
            }
        });
    }

    /// \brief Factorization of the register rows of the group.
    /// \tparam fuse_rhs apply the forward pass to the right-hand side as well
    ///         (solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     tx   rank of the calling thread in its group
    /// \param[in,out] rA   register rows of the thread
    /// \param[in,out] rB   right-hand-side registers (fuse_rhs only)
    /// \param[in,out] rpiv pivot entry of each slot of the thread,
    ///                initialized by init_piv
    /// \param[in]     work workspace of the group
    /// \param[in]     sync barrier of the group
    /// \return 0, or 1 + the first column whose best pivot fell below
    ///         singular_floor (MAGMA's linfo)
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    eliminate(const int tx, T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
              int (&rpiv)[rows_per_thread], T* work,
              Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        int linfo = 0;
        if constexpr (Config.row_interchange == RowInterchange::Logical) {
            T* const sB  = work;
            T* const sx  = work + N;
            T* const dsx = work + 2 * N;
            if constexpr (Config.unroll_loops) {
                TDLS_UNROLL_FORCE
                for (int i = 0; i < N; i++)
                    logical_elimination_step<fuse_rhs>(tx, i, rA, rB, rpiv, sB, sx, dsx, linfo,
                                                       sync);
            } else {
                for (int i = 0; i < N; i++)
                    logical_elimination_step<fuse_rhs>(tx, i, rA, rB, rpiv, sB, sx, dsx, linfo,
                                                       sync);
            }
        } else {
            if constexpr (Config.unroll_loops) {
                TDLS_UNROLL_FORCE
                for (int i = 0; i < N; i++)
                    physical_elimination_step<fuse_rhs>(tx, i, rA, rB, rpiv, work, linfo, sync);
            } else {
                for (int i = 0; i < N; i++)
                    physical_elimination_step<fuse_rhs>(tx, i, rA, rB, rpiv, work, linfo, sync);
            }
            sync(); // no barrier closes the steps: this one frees the workspace
        }
        return linfo;
    }

    /* =====================================================================
       SUBSTITUTION of S x = b, the column-major layout.
       Forward: the owner of the row at position i publishes its
       right-hand-side entry, every thread updates its rows below. This is
       the elimination of the right-hand side that solve_inplace folds into
       the factorization, replayed from the factored rows. Backward: the
       right-hand side is staged at the positions of its rows, then column
       i of U is published, every thread computes x_i, the owner of entry i
       keeps it and every thread updates its entries above. Under physical
       row interchanges, the right-hand side is first gathered in the
       pivoted order, and both passes run on the positions of the rows,
       which are static: the arithmetic is the logical one by construction,
       down to the contractions a compiler applies, and in an unrolled
       sweep the owner of each position is known at compile time.
       ===================================================================== */

    /// \brief One step of the forward substitution.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     i     position of this step
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    right-hand-side registers of the thread
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     sB    workspace: broadcast slot of the pivot entry
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    forward_step(const int i, const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                 const int (&rowid)[rows_per_thread], T* sB,
                 Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (rowid[K] == i) sB[0] = rB[K];
            }
        });
        sync();

        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if ((slot_is_full(K) || rowid[K] < N) && rowid[K] > i) rB[K] -= rA[K][i] * sB[0];
            }
        });
        sync();
    }

    /// \brief Forward pass: L y = P b on the given row positions, y
    /// overwriting the right-hand-side registers.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    right-hand-side registers of the thread
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    forward_pass(const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                 const int (&rowid)[rows_per_thread], T* work,
                 Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        T* const sB = work;
        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < N; i++)
                forward_step(i, rA, rB, rowid, sB, sync);
        } else {
            for (int i = 0; i < N; i++)
                forward_step(i, rA, rB, rowid, sB, sync);
        }
    }

    /// \brief One step of the backward substitution.
    ///
    /// Every thread computes the same x_i. The owner of entry i keeps it in
    /// its register instead of writing sB[i], which every thread reads in
    /// this very barrier interval: the write MAGMA does here is a data
    /// race as soon as the threads leave lockstep.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     i     position (and solution entry) of this step
    /// \param[in]     tx    rank of the calling thread in its group
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    solution registers of the thread: entry
    ///                tx + K * threads_per_system in slot K
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in,out] sB    workspace: right-hand side being reduced
    /// \param[in]     sx    workspace: column i of U
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    backward_step(const int i, const int tx, const T (&rA)[rows_per_thread][N],
                  T (&rB)[rows_per_thread], const int (&rowid)[rows_per_thread], T* sB, T* sx,
                  Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || rowid[K] < N) sx[rowid[K]] = rA[K][i];
            }
        });
        sync();

        const T reg = sB[i] / sx[i];

        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                const int idx = tx + K * threads_per_system;
                if (slot_is_full(K) || idx < N) {
                    if (idx < i) sB[idx] = sB[idx] - reg * sx[idx];
                    rB[K] = (idx == i) ? reg : rB[K];
                }
            }
        });
        sync();
    }

    /// \brief Backward pass: U x = y on the given row positions, the
    /// forward-reduced right-hand side y in the registers on entry, the
    /// solution entries of the thread on exit.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     tx    rank of the calling thread in its group
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    y entries of the rows of the thread on entry,
    ///                solution entries tx + K * threads_per_system on exit
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    backward_pass(const int tx, const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                  const int (&rowid)[rows_per_thread], T* work,
                  Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        T* const sB = work;
        T* const sx = work + N;
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || rowid[K] < N) sB[rowid[K]] = rB[K];
            }
        });
        sync();

        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int i = N - 1; i >= 0; i--)
                backward_step(i, tx, rA, rB, rowid, sB, sx, sync);
        } else {
            for (int i = N - 1; i >= 0; i--)
                backward_step(i, tx, rA, rB, rowid, sB, sx, sync);
        }
    }

    /// \brief Gather of the right-hand side in the pivoted order, under
    /// physical row interchanges: each slot receives the entry of the
    /// original row it holds.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     tx     rank of the calling thread in its group
    /// \param[in,out] rB     on entry, entry tx + K * threads_per_system of
    ///                the right-hand side in slot K, in original order; on
    ///                exit, the entry of the row held by slot K
    /// \param[in]     origin original index of the row held by each slot
    /// \param[in]     staged workspace: the right-hand side in original
    ///                order
    /// \param[in]     sync   barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    gather_rhs(const int tx, T (&rB)[rows_per_thread], const int (&origin)[rows_per_thread],
               T* staged, Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N)
                    staged[tx + K * threads_per_system] = rB[K];
            }
        });
        sync();

        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) rB[K] = staged[origin[K]];
            }
        });
    }

    /* =====================================================================
       SUBSTITUTION of S^T x = b, the row-major layout.
       With P S = L U, S^T = U^T L^T P. Both passes run on the positions of
       the rows, held at the entries of the same index: z_p and w_p live in
       the slot of entry p, which b'(p) leaves when it is consumed, so no
       second vector is needed. Forward, U^T z = b: the row at position i
       publishes its row of U, the owner of entry i its b'(i), every thread
       computes z_i = b'(i) / U(i, i) through the reciprocal of the pivot,
       the owner of entry i keeps z_i, and each entry c > i loses
       U(i, c) z_i. This is the forward pass that solve_inplace folds into
       the factorization, replayed from the factored rows. Backward,
       L^T w = z: the row at position k publishes its multipliers L(k, c),
       c < k, the owner of entry k its w_k, and each entry c < k loses
       L(k, c) w_k; L has a unit diagonal, so no division sits on the chain
       of the steps. One barrier per step: each step writes its own half of
       a double buffer, which no thread still reads (see segment). Then
       x = P^T w, one exchange through the workspace: under logical row
       interchanges, row r of S takes w at its position; under physical
       ones, w_p goes to the original index of the row at position p. These
       are the mirror of the staging of the right-hand side that the
       substitutions of S x = b make at the start.
       ===================================================================== */

    /// \brief Half of the double buffer of the transposed substitutions
    /// used by a step: N + 1 elements, a row segment and one value. A thread
    /// reads the half of step s after the barrier of step s; the next write
    /// to that half comes at step s + 2, after the barrier of step s + 1,
    /// which every thread reaches only once its reads of step s are done.
    /// Both halves take 2 * N + 2 elements, within the workspace for
    /// N >= 2; a dimension of 1 uses the first half only.
    /// \param[in] work workspace of the group
    /// \param[in] step index of the step
    /// \return the half of the step
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr T* segment(T* work,
                                                                  const int step) noexcept {
        return work + (step & 1) * (N + 1);
    }

    /// \brief One step of the forward substitution U^T z = b.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     tx    rank of the calling thread in its group
    /// \param[in]     i     position of this step
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    entries tx + K * threads_per_system: b' below
    ///                position i, z from it on
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    transposed_forward_step(const int tx, const int i, const T (&rA)[rows_per_thread][N],
                            T (&rB)[rows_per_thread], const int (&rowid)[rows_per_thread], T* work,
                            Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        T* const sx = segment(work, i);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (rowid[K] == i) {
                    // Unrolled, the copy runs over all N entries under a
                    // guard: a trip count set by the step lets the AMD
                    // compiler unroll it before the step loop, and the
                    // rows then fall to scratch memory.
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int j = 0; j < N; j++)
                            if (j >= i) sx[j] = rA[K][j];
                    } else {
                        for (int j = i; j < N; j++)
                            sx[j] = rA[K][j];
                    }
                }
            }
        });
        at_position(tx, i, [&](auto K_tag) { sx[N] = rB[decltype(K_tag)::value]; });
        sync();

        const T z_i = sx[N] * (T(1) / sx[i]);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                const int c = tx + K * threads_per_system;
                if (c == i) rB[K] = z_i;
                if ((slot_is_full(K) || c < N) && c > i) rB[K] -= sx[c] * z_i;
            }
        });
    }

    /// \brief One step of the backward substitution L^T w = z.
    ///
    /// The step uses the half of step k + 1: the first step, k = N - 1,
    /// then writes the half that the last step of the forward pass does not
    /// read, so that no barrier separates the passes.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     tx    rank of the calling thread in its group
    /// \param[in]     k     position of this step
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    entries tx + K * threads_per_system: z below
    ///                position k, w from it on
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    transposed_backward_step(const int tx, const int k, const T (&rA)[rows_per_thread][N],
                             T (&rB)[rows_per_thread], const int (&rowid)[rows_per_thread], T* work,
                             Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        T* const sx = segment(work, k + 1);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (rowid[K] == k) {
                    // Unrolled over all N entries under a guard, as in the
                    // forward step.
                    if constexpr (Config.unroll_loops) {
                        TDLS_UNROLL_FORCE
                        for (int c = 0; c < N; c++)
                            if (c < k) sx[c] = rA[K][c];
                    } else {
                        for (int c = 0; c < k; c++)
                            sx[c] = rA[K][c];
                    }
                }
            }
        });
        at_position(tx, k, [&](auto K_tag) { sx[N] = rB[decltype(K_tag)::value]; });
        sync();

        const T w_k = sx[N];
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                const int c = tx + K * threads_per_system;
                if (c < k) rB[K] -= sx[c] * w_k;
            }
        });
    }

    /// \brief Forward then backward pass of S^T x = b on the given row
    /// positions: b in the entries of the thread on entry, w at the entries
    /// of the same positions on exit. Position 0 updates no entry in the
    /// backward pass: its step is skipped.
    /// \tparam backward_only skip the forward pass, folded into the
    ///         factorization (solve_inplace)
    /// \tparam Sync          callable type of the barrier
    /// \param[in]     tx    rank of the calling thread in its group
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    entries tx + K * threads_per_system
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool backward_only, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    transposed_passes(const int tx, const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                      const int (&rowid)[rows_per_thread], T* work,
                      Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        if constexpr (Config.unroll_loops) {
            if constexpr (!backward_only) {
                TDLS_UNROLL_FORCE
                for (int i = 0; i < N; i++)
                    transposed_forward_step(tx, i, rA, rB, rowid, work, sync);
            }
            TDLS_UNROLL_FORCE
            for (int k = N - 1; k >= 1; k--)
                transposed_backward_step(tx, k, rA, rB, rowid, work, sync);
        } else {
            if constexpr (!backward_only) {
                for (int i = 0; i < N; i++)
                    transposed_forward_step(tx, i, rA, rB, rowid, work, sync);
            }
            for (int k = N - 1; k >= 1; k--)
                transposed_backward_step(tx, k, rA, rB, rowid, work, sync);
        }
    }

    /// \brief x = P^T w, through the workspace. Logical row interchanges:
    /// the slot of row r of S takes w at the position of the row. Physical
    /// ones: w_p goes to the original index of the row at position p. The
    /// exchange is staged in the half of the double buffer that the last
    /// backward step does not read.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     tx   rank of the calling thread in its group
    /// \param[in,out] rB   w of the positions tx + K * threads_per_system on
    ///                entry, solution entries tx + K * threads_per_system
    ///                on exit
    /// \param[in]     rpiv pivot entry of each slot of the thread
    /// \param[in]     work workspace of the group
    /// \param[in]     sync barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    unpermute_solution(const int tx, T (&rB)[rows_per_thread], const int (&rpiv)[rows_per_thread],
                       T* work, Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        constexpr bool logical = Config.row_interchange == RowInterchange::Logical;
        T* const staged        = segment(work, 1);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N)
                    staged[logical ? tx + K * threads_per_system : rpiv[K]] = rB[K];
            }
        });
        sync();

        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N)
                    rB[K] = staged[logical ? rpiv[K] : tx + K * threads_per_system];
            }
        });
    }

    /* =====================================================================
       SUBSTITUTION, dispatch on the layout.
       ===================================================================== */

    /// \brief Substitution from the factored rows, the right-hand side in
    /// the registers on entry, the solution entries of the thread on exit:
    /// L y = P b then U x = y under the column-major layout, U^T z = b,
    /// L^T w = z and x = P^T w under the row-major one.
    /// \tparam backward_only skip the forward pass, folded into the
    ///         factorization (solve_inplace)
    /// \tparam Sync          callable type of the barrier
    /// \param[in]     tx   rank of the calling thread in its group
    /// \param[in]     rA   factored register rows of the thread
    /// \param[in,out] rB   right-hand-side registers of the thread on entry
    ///                (forward-reduced with backward_only), solution
    ///                entries tx + K * threads_per_system on exit
    /// \param[in]     rpiv pivot entry of each slot of the thread
    /// \param[in]     work workspace of the group
    /// \param[in]     sync barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool backward_only, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitution(const int tx, const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                 const int (&rpiv)[rows_per_thread], T* work,
                 Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        constexpr bool logical = Config.row_interchange == RowInterchange::Logical;
        if constexpr (transposed) {
            if constexpr (logical) {
                transposed_passes<backward_only>(tx, rA, rB, rpiv, work, sync);
            } else {
                int position[rows_per_thread];
                init_piv(tx, position);
                transposed_passes<backward_only>(tx, rA, rB, position, work, sync);
            }
            unpermute_solution(tx, rB, rpiv, work, sync);
        } else if constexpr (logical) {
            if constexpr (!backward_only) forward_pass(rA, rB, rpiv, work, sync);
            backward_pass(tx, rA, rB, rpiv, work, sync);
        } else {
            int position[rows_per_thread];
            init_piv(tx, position);
            if constexpr (!backward_only) {
                // Staged apart from the broadcast slot of the pass, which a
                // faster thread may already write.
                gather_rhs(tx, rB, rpiv, work + N, sync);
                forward_pass(rA, rB, position, work, sync);
            }
            backward_pass(tx, rA, rB, position, work, sync);
        }
    }

    /* =====================================================================
       Entry points.
       ===================================================================== */

    /// \brief The barrier deduced for the group of the calling thread.
    ///
    /// The entry points deduce it on every call when no barrier is passed.
    /// A caller that also needs the barrier for its own exchanges between
    /// the threads of the group builds it once with this function, calls
    /// it at will, and passes it to the entry points, which then use it as
    /// is. On GPU, every thread of the group must call this function
    /// together, as the entry points (see tdls::AutoSync).
    /// \param[in] work workspace of the group
    /// \return tdls::NoSync for one thread per system, the deduced barrier
    ///         otherwise
    TDLS_EXEC_CHECK_DISABLE
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr auto make_sync(T* work) noexcept {
        AutoSync deduce;
        return detail::resolve_sync<threads_per_system>(deduce, work, threads_per_system);
    }

    /// \brief In-place LU factorization with partial pivoting.
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          matrix, or slice of the thread, pre-offset
    ///                by the caller; its factorization on exit
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        pivot entries, or slice of the thread
    ///                (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group, deduced by default (see
    ///                           tdls::AutoSync)
    /// \return false on a singular matrix (factorization unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_piv, bool internal_matrix, typename Sync = tdls::AutoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    factorize(const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
              const int piv_stride, T* work,
              Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        auto&& group = detail::resolve_sync<threads_per_system>(sync, work, threads_per_system);
        T* const exchange = work + detail::barrier_elements;
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        init_piv(tx, rpiv);
        const int linfo = eliminate<false>(tx, rA, rB, rpiv, exchange, group);
        store_rows<internal_matrix>(tx, A, A_stride, rA);
        store_piv<internal_piv>(tx, piv, piv_stride, rpiv);
        group(); // every output visible to the whole group on return
        return linfo == 0;
    }

    /// \brief Solve from a factorization produced by factorize (b and x
    /// distinct).
    /// \tparam internal_rhs    residency of b and x
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  A          factored matrix, or slice of the thread
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        pivot entries produced by factorize
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  b          right-hand side, in original order
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of b and x (external mode)
    /// \param[in]  work       workspace of workspace_size elements, shared
    ///             by the group
    /// \param[in]  sync       barrier of the group, deduced by default (see
    ///                        tdls::AutoSync)
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::AutoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
               const int* TDLS_RESTRICT piv, const int piv_stride, const T* TDLS_RESTRICT b,
               T* TDLS_RESTRICT x, const int rhs_stride, T* work,
               Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        auto&& group = detail::resolve_sync<threads_per_system>(sync, work, threads_per_system);
        T* const exchange = work + detail::barrier_elements;
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        load_piv<internal_piv>(tx, piv, piv_stride, rpiv);
        load_rhs<internal_rhs>(tx, b, rhs_stride, rB);
        substitution<false>(tx, rA, rB, rpiv, exchange, group);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
    }

    /// \brief Solve with b = e_col generated on the fly: the
    /// consistent-tangent-operator path.
    /// \tparam internal_rhs    residency of x
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  A          factored matrix, or slice of the thread
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        pivot entries produced by factorize
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  col        index of the canonical column e_col
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of x (external mode)
    /// \param[in]  work       workspace of workspace_size elements, shared
    ///             by the group
    /// \param[in]  sync       barrier of the group, deduced by default (see
    ///                        tdls::AutoSync)
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::AutoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_canonical(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                         const int* TDLS_RESTRICT piv, const int piv_stride, const int col,
                         T* TDLS_RESTRICT x, const int rhs_stride, T* work,
                         Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        auto&& group = detail::resolve_sync<threads_per_system>(sync, work, threads_per_system);
        T* const exchange = work + detail::barrier_elements;
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        load_piv<internal_piv>(tx, piv, piv_stride, rpiv);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            rB[K]           = (tx + K * threads_per_system == col) ? T(1) : T(0);
        });
        substitution<false>(tx, rA, rB, rpiv, exchange, group);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
    }

    /// \brief Solve in place from a factorization produced by factorize:
    /// x holds the right-hand side on entry and the solution on exit.
    ///
    /// Each thread reads and writes only the entries of its own rows, the
    /// same set before and after, so the in-place form needs no copy.
    /// \tparam internal_rhs    residency of x
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in]     A          factored matrix, or slice of the thread
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        pivot entries produced by factorize
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] x          right-hand side on entry, solution on
    ///                exit
    /// \param[in]     rhs_stride element stride of x (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group, deduced by default (see
    ///                           tdls::AutoSync)
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::AutoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_inplace(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                       const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                       const int rhs_stride, T* work,
                       Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        auto&& group = detail::resolve_sync<threads_per_system>(sync, work, threads_per_system);
        T* const exchange = work + detail::barrier_elements;
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        load_piv<internal_piv>(tx, piv, piv_stride, rpiv);
        load_rhs<internal_rhs>(tx, x, rhs_stride, rB);
        substitution<false>(tx, rA, rB, rpiv, exchange, group);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
    }

    /// \brief factorize + substitute in one call. The rows stay in
    /// registers between the two, so the factors are stored once and
    /// never reloaded.
    ///
    /// The barrier sequence does not depend on the verdict: a singular
    /// matrix is substituted as well, and x is then unspecified.
    /// \tparam internal_rhs    residency of b and x
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          on entry the matrix, or slice of the
    ///                thread, pre-offset by the caller; on exit its
    ///                factorization (usable for further substitute* calls)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        pivot entries, or slice of the thread
    ///                (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group, deduced by default (see
    ///                           tdls::AutoSync)
    /// \return false on a singular matrix (x unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::AutoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve(const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
          const int piv_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          T* work, Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        auto&& group = detail::resolve_sync<threads_per_system>(sync, work, threads_per_system);
        T* const exchange = work + detail::barrier_elements;
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        init_piv(tx, rpiv);
        const int linfo = eliminate<false>(tx, rA, rB, rpiv, exchange, group);
        store_rows<internal_matrix>(tx, A, A_stride, rA);
        store_piv<internal_piv>(tx, piv, piv_stride, rpiv);
        load_rhs<internal_rhs>(tx, b, rhs_stride, rB);
        substitution<false>(tx, rA, rB, rpiv, exchange, group);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
        return linfo == 0;
    }

    /// \brief Factorization with the forward substitution folded in: the
    /// original MAGMA kernel.
    ///
    /// y holds the unpermuted right-hand side on entry and the solution on
    /// exit. The elimination is applied to y while the factorization runs,
    /// so the separate forward pass disappears and only the backward pass
    /// remains. The result is bitwise-identical to factorize +
    /// substitute_inplace. The barrier sequence does not depend on the
    /// verdict: a singular matrix is substituted as well, and y is then
    /// unspecified.
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          on entry the matrix, or slice of the
    ///                thread, pre-offset by the caller; on exit its
    ///                factorization
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        pivot entries, or slice of the thread
    ///                (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group, deduced by default (see
    ///                           tdls::AutoSync)
    /// \return false on a singular matrix (y unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::AutoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve_inplace(const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                  const int piv_stride, T* TDLS_RESTRICT y, const int rhs_stride, T* work,
                  Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        auto&& group = detail::resolve_sync<threads_per_system>(sync, work, threads_per_system);
        T* const exchange = work + detail::barrier_elements;
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        init_piv(tx, rpiv);
        load_rhs<internal_rhs>(tx, y, rhs_stride, rB);
        const int linfo = eliminate<true>(tx, rA, rB, rpiv, exchange, group);
        store_rows<internal_matrix>(tx, A, A_stride, rA);
        store_piv<internal_piv>(tx, piv, piv_stride, rpiv);
        substitution<true>(tx, rA, rB, rpiv, exchange, group);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) TDLS_COOP_LUPP_Y(K) = rB[K];
            }
        });
        group(); // every output visible to the whole group on return
        return linfo == 0;
    }
};



#undef TDLS_COOP_LUPP_A
#undef TDLS_COOP_LUPP_PIV
#undef TDLS_COOP_LUPP_B
#undef TDLS_COOP_LUPP_X
#undef TDLS_COOP_LUPP_Y



} // namespace tdls

#if defined(__NVCOMPILER)
#pragma diag_pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_STATIC_HPP
