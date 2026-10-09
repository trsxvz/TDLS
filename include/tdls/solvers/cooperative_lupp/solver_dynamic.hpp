#ifndef TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_DYNAMIC_HPP
#define TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_DYNAMIC_HPP



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
/// \brief Runtime-size variant of the CooperativeLUpp solver.
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
/// Same algorithm as CooperativeLUppSolverStatic (solver_static.hpp):
/// the LU with partial pivoting of MAGMA's register-blocked
/// small-system kernel, logical or physical row interchanges, the rows
/// of a system shared by a group of threads, a workspace and a deduced
/// or caller-provided barrier, the pivot entries as pivot output and
/// the pivots on the factored diagonal. The difference is that the
/// system dimension n is a runtime function parameter instead of a
/// template parameter. Only the number of rows per thread
/// (Config.rows_per_thread) stays compile-time: it sizes the pivot
/// entries and the right-hand-side entries kept by each thread, while
/// the number of threads, threads_per_system(n), and every loop bound
/// are runtime values.
///
/// Deliberate differences with the compile-time solver:
///   - No unroll pragma anywhere: with runtime bounds nothing can be
///     register-resident by full unrolling, so the unroll_loops
///     knob of CooperativeLUppConfig is ignored.
///   - The rows of a thread are updated in place in the matrix instead of
///     a register copy: a row of runtime length cannot live in registers.
///   - No internal_rhs / internal_piv / internal_matrix booleans: every
///     operand is the whole object, plain pointer + stride, in memory
///     reachable by every thread of the group.
///   - The size of the group depends on n, so the deduced barrier checks
///     it at run time, and a pass that offers no deduced barrier refuses
///     tdls::AutoSync at compile time. With tdls::NoSync, the caller must
///     keep n <= Config.rows_per_thread (one thread per system).
///
/// For equal shapes (n = N, same configuration), results are bitwise
/// identical to the compile-time solver: factored rows, pivot entries and
/// solution. The arithmetic sequence is the same, only the storage of the
/// rows and the loop mechanics differ. Under physical row interchanges,
/// this holds up to the multiply-adds a compiler may fuse differently in
/// each (see CooperativeLUppConfig::row_interchange).
///
/// The compile-time solver is the performance path; this variant is the
/// flexibility path (dimensions unknown at compile time, fast builds).
///
/// The calling convention, the factored format and the pivot convention
/// are those of the compile-time solver, described in solver_static.hpp.
/// So are the guarantees of the group: a fixed sequence of barriers, no
/// data race, inputs read by their owners on entry, results visible to
/// the whole group on return. The workspace holds workspace_size(n)
/// elements, laid out as there with n for N, without the restrict
/// qualifier, for the reason given there. The matrix layout follows
/// Config.layout: row-major by default, the flat index remapped for
/// column-major storage, resolved at compile time.
///
/// Preconditions: n >= 1 and 0 <= tx < threads_per_system(n). Offsets are
/// computed in 32-bit arithmetic: the flat element index (for the matrix,
/// n*n) must stay below 2^31 and the largest element offset of every
/// array (for the matrix, (n*n-1)*A_stride) below 2^32.



#include <type_traits>

#include <tdls/core/group.hpp>
#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/solvers/cooperative_lupp/config.hpp>



// gcc's flow analysis cannot prove that the right-hand-side entries of a
// thread are loaded before being read: the load stops at the first
// phantom row and every later use stops at the same row, a property
// validated bitwise against the compile-time solver. The resulting
// -Wmaybe-uninitialized reports are spurious. Under UBSan instrumentation
// at -O2 and above, gcc additionally loses the value ranges of the loop
// indices and reports impossible subscripts through -Warray-bounds,
// exactly as in the TiledLUpp headers; the exercised paths are certified
// free of out-of-bounds accesses by the constexpr suite. Both
// suppressions are scoped to this header and to those warnings only,
// exactly as in solver_static.hpp.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace tdls {



/* Addressing macros. They expand inside member functions where n, the
   parameter names (A, A_stride, piv, piv_stride, b, x, y, rhs_stride)
   and the Config value are in scope. #undef'd at the end of this
   header. */

/// \def TDLS_COOP_LUPP_DYN_A
/// \brief Strided element (r, c) of the matrix, flat index remapped by
/// Config.layout.
#define TDLS_COOP_LUPP_DYN_A(r, c) A[TDLS_LAYOUT_INDEX(r, c, n) * unsigned(A_stride)]
/// \def TDLS_COOP_LUPP_DYN_PIV
/// \brief Strided pivot entry of row r.
#define TDLS_COOP_LUPP_DYN_PIV(r) piv[unsigned(r) * unsigned(piv_stride)]
/// \def TDLS_COOP_LUPP_DYN_B
/// \brief Strided entry r of the right-hand side.
#define TDLS_COOP_LUPP_DYN_B(r) b[unsigned(r) * unsigned(rhs_stride)]
/// \def TDLS_COOP_LUPP_DYN_X
/// \brief Strided entry r of the solution.
#define TDLS_COOP_LUPP_DYN_X(r) x[unsigned(r) * unsigned(rhs_stride)]
/// \def TDLS_COOP_LUPP_DYN_Y
/// \brief Strided entry r of the right-hand side solved in place.
#define TDLS_COOP_LUPP_DYN_Y(r) y[unsigned(r) * unsigned(rhs_stride)]



/// \brief Runtime-size dense LU factorization with partial pivoting
/// shared by a group of threads, solving one n x n system per group.
///
/// All entry points are static, host- and device-callable, called by
/// every thread of the group with the dimension n and its rank tx, and
/// take raw pointers pre-offset by the caller with one runtime stride per
/// array. Entry points:
///   - factorize:             A := P*L*U in place, pivot entries out
///   - substitute:            x := A^-1 b from a factorization (b and x
///     distinct)
///   - substitute_canonical:  idem with b = e_col (tangent-operator columns)
///   - substitute_inplace:    idem with b == x
///   - solve:                 factorize + substitute
///   - solve_inplace:         factorize with the forward pass folded in,
///     the original MAGMA kernel
///
/// The factored matrix and the pivot array follow the format of
/// CooperativeLUppSolverStatic under the same configuration: a
/// factorization produced here must be consumed by the substitution
/// routines of this family.
///
/// \tparam T      scalar type (float, double or long double)
/// \tparam Config compile-time knobs, passed as a constexpr value: rows per
///         thread, row interchanges, relative pivot threshold, singularity
///         floor, matrix layout; see CooperativeLUppConfig (unroll_loops
///         is ignored by this variant)
template<typename T, CooperativeLUppConfig<T> Config = CooperativeLUppConfig<T>{}>
struct CooperativeLUppSolverDynamic {

    static_assert(Config.rows_per_thread >= 1,
                  "CooperativeLUppSolverDynamic: rows_per_thread must be >= 1");
    static_assert(Config.relative_pivot_threshold.is_finite() && Config.singular_floor.is_finite(),
                  "CooperativeLUppSolverDynamic: relative_pivot_threshold and singular_floor must "
                  "be finite (and fit a 63-bit mantissa)");

    /// \brief Row slots of each thread: Config.rows_per_thread. When n is
    /// smaller, one thread solves the system and its slots beyond n hold
    /// phantom rows. A value below 1, rejected above, is clamped to 1 so
    /// that the rejection stays the only diagnostic.
    static constexpr int rows_per_thread = Config.rows_per_thread < 1 ? 1 : Config.rows_per_thread;
    /// \brief Relative threshold of the pivot choice, read once from the
    /// configuration (see CooperativeLUppConfig::relative_pivot_threshold).
    static constexpr T relative_pivot_threshold = Config.relative_pivot_threshold;
    /// \brief Singularity floor, read once from the configuration (see
    /// CooperativeLUppConfig::singular_floor).
    static constexpr T singular_floor = Config.singular_floor;

    static_assert(relative_pivot_threshold > T(0) && relative_pivot_threshold <= T(1),
                  "CooperativeLUppSolverDynamic: relative_pivot_threshold must be in (0, 1]");
    static_assert(singular_floor > T(0),
                  "CooperativeLUppSolverDynamic: singular_floor must be positive");

    /// \brief Threads solving one system of dimension n:
    /// ceil(n / rows_per_thread).
    /// \param[in] n system dimension
    /// \return the number of threads of a group
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    threads_per_system(const int n) noexcept {
        return (n + rows_per_thread - 1) / rows_per_thread;
    }

    /// \brief Elements of the workspace shared by the threads of a group:
    /// the two elements of the deduced CPU barrier, then 3 * n under
    /// logical row interchanges, 2 * n + 2 * threads_per_system(n) + 5
    /// under physical ones.
    /// \param[in] n system dimension
    /// \return the number of elements of the workspace
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int workspace_size(const int n) noexcept {
        return detail::barrier_elements + (Config.row_interchange == RowInterchange::Logical
                                               ? 3 * n
                                               : 2 * n + 2 * threads_per_system(n) + 5);
    }

    /* =====================================================================
       Pivot entries and right-hand-side entries.
       Slot K of thread tx holds row tx + K * threads, a phantom row when
       it reaches n. The rows themselves stay in the matrix; only their
       pivot entries and their right-hand-side entries are kept by the
       thread. Each thread reads and writes only the entries of its own
       rows, so the operands never carry data between threads: the
       workspace does.
       ===================================================================== */

    /// \brief Initial pivot entries: no row has moved yet, so the entry of
    /// each slot is the index of its row, phantom rows included (theirs
    /// stays at or above n). Under logical row interchanges the entry is the
    /// position of the row of the slot, under physical ones the original
    /// index of the row held by the slot: both start there.
    /// \param[in]  tx      rank of the calling thread in its group
    /// \param[in]  threads threads of the group, threads_per_system(n)
    /// \param[out] rpiv    pivot entry of each slot of the thread
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    init_piv(const int tx, const int threads, int (&rpiv)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K)
            rpiv[K] = tx + K * threads;
    }

    /// \brief Load the pivot entries recorded by factorize.
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  threads    threads of the group, threads_per_system(n)
    /// \param[in]  piv        pivot entries (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv
    /// \param[out] rpiv       pivot entry of each slot of the thread
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_piv(const int n, const int tx, const int threads, const int* TDLS_RESTRICT piv,
             const int piv_stride, int (&rpiv)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            rpiv[K]     = r < n ? TDLS_COOP_LUPP_DYN_PIV(r) : r;
        }
    }

    /// \brief Record the pivot entries of the calling thread.
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  threads    threads of the group, threads_per_system(n)
    /// \param[out] piv        pivot entries (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  rpiv       pivot entry of each slot of the thread
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_piv(const int n, const int tx, const int threads, int* TDLS_RESTRICT piv,
              const int piv_stride, const int (&rpiv)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            TDLS_COOP_LUPP_DYN_PIV(r) = rpiv[K];
        }
    }

    /// \brief Load the right-hand-side entries of the rows of the calling
    /// thread.
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  threads    threads of the group, threads_per_system(n)
    /// \param[in]  b          right-hand side (caller-pre-offset)
    /// \param[in]  rhs_stride element stride of b
    /// \param[out] rB         right-hand-side entries of the thread
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_rhs(const int n, const int tx, const int threads, const T* TDLS_RESTRICT b,
             const int rhs_stride, T (&rB)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            rB[K] = TDLS_COOP_LUPP_DYN_B(r);
        }
    }

    /// \brief Store the solution entries held by the calling thread.
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  threads    threads of the group, threads_per_system(n)
    /// \param[out] x          solution (caller-pre-offset)
    /// \param[in]  rhs_stride element stride of x
    /// \param[in]  rB         solution entries of the thread
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_solution(const int n, const int tx, const int threads, T* TDLS_RESTRICT x,
                   const int rhs_stride, const T (&rB)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            TDLS_COOP_LUPP_DYN_X(r) = rB[K];
        }
    }

    /* =====================================================================
       FACTORIZATION, right-looking, one column per step, as in the
       compile-time solver, in place in the matrix. Logical row
       interchanges: the magnitudes of the column are published at the
       positions of their rows, every thread finds the same pivot, the
       owner of the pivot row publishes it, and every thread eliminates its
       own rows below the pivot. Physical row interchanges: each thread
       publishes its best candidate, every thread finds the same pivot, the
       owner of position i publishes its row, the owner of the pivot row as
       well when the pivot is not in place, and every thread eliminates its
       rows below the pivot after the exchange.
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
    /// \tparam fuse_rhs apply the elimination to the right-hand side as well
    ///         (the forward pass of solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     i        column of this step
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       right-hand-side entries (fuse_rhs only)
    /// \param[in,out] rowid    position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     sB       workspace: broadcast slot of the pivot
    ///                right-hand side (fuse_rhs only)
    /// \param[in]     sx       workspace: pivot row
    /// \param[in]     dsx      workspace: magnitudes of the column
    /// \param[in,out] linfo    0, or 1 + the first column whose best pivot
    ///                fell below singular_floor
    /// \param[in]     sync     barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    logical_elimination_step(const int n, const int i, const int tx, const int threads,
                             T* TDLS_RESTRICT A, const int A_stride,
                             [[maybe_unused]] T (&rB)[rows_per_thread],
                             int (&rowid)[rows_per_thread], T* sB, T* sx, T* dsx, int& linfo,
                             Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        // izamax: the magnitudes of column i, at the positions of their rows.
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            dsx[rowid[K]] = detail::abs(TDLS_COOP_LUPP_DYN_A(r, i));
        }
        sync();

        // Every thread scans the same magnitudes and finds the same pivot.
        T rx_abs_max = dsx[i];
        int max_id   = i;
        for (int j = i + 1; j < n; j++) {
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
        // are swapped.
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            if (rowid[K] == max_id) {
                rowid[K] = i;
                for (int j = i; j < n; j++)
                    sx[j] = update * TDLS_COOP_LUPP_DYN_A(r, j);
                if constexpr (fuse_rhs) sB[0] = rB[K];
            } else if (rowid[K] == i) {
                rowid[K] = max_id;
            }
        }
        sync();

        const T reg = zero_pivot ? T(1) : T(1) / sx[i];

        // scal and ger: every thread eliminates its rows below the pivot.
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            if (rowid[K] > i) {
                TDLS_COOP_LUPP_DYN_A(r, i) *= reg;
                for (int j = i + 1; j < n; j++)
                    TDLS_COOP_LUPP_DYN_A(r, j) -= TDLS_COOP_LUPP_DYN_A(r, i) * sx[j];
                if constexpr (fuse_rhs) rB[K] -= TDLS_COOP_LUPP_DYN_A(r, i) * sB[0];
            }
        }
        sync();
    }

    /// \brief Copy a row of the matrix to the first n entries of an
    /// exchange record of the workspace.
    /// \param[in]  n        system dimension
    /// \param[in]  A        matrix (caller-pre-offset)
    /// \param[in]  A_stride element stride of A
    /// \param[in]  r        row of the matrix
    /// \param[out] record   exchange record
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    save_row(const int n, const T* TDLS_RESTRICT A, const int A_stride, const int r,
             T* record) noexcept {
        for (int c = 0; c < n; c++)
            record[c] = TDLS_COOP_LUPP_DYN_A(r, c);
    }

    /// \brief Copy the first n entries of an exchange record of the
    /// workspace to a row of the matrix, the converse of save_row.
    /// \param[in]     n        system dimension
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in]     r        row of the matrix
    /// \param[in]     record   exchange record
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_row(const int n, T* TDLS_RESTRICT A, const int A_stride, const int r,
             const T* record) noexcept {
        for (int c = 0; c < n; c++)
            TDLS_COOP_LUPP_DYN_A(r, c) = record[c];
    }

    /// \brief One column of the factorization under physical row
    /// interchanges, as in the compile-time solver: the row at position k
    /// is row k of the matrix, held by slot k / threads of thread
    /// k % threads.
    /// \tparam fuse_rhs apply the elimination to the right-hand side as well
    ///         (the forward pass of solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     i        column of this step
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       right-hand-side entries (fuse_rhs only)
    /// \param[in,out] origin   original index of the row held by each slot
    /// \param[in]     work     workspace of the group
    /// \param[in,out] linfo    0, or 1 + the first column whose best pivot
    ///                fell below singular_floor
    /// \param[in]     sync     barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    physical_elimination_step(const int n, const int i, const int tx, const int threads,
                              T* TDLS_RESTRICT A, const int A_stride,
                              [[maybe_unused]] T (&rB)[rows_per_thread],
                              int (&origin)[rows_per_thread], T* work, int& linfo,
                              Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        // Workspace layout of the compile-time solver.
        T* const candidate = work;
        T* const where     = candidate + threads;
        T* const in_place  = where + threads;
        T* const leaving   = in_place + 1;
        T* const entering  = leaving + n + 2;

        // Candidate of the thread. The owner of position i starts from the
        // row in place and publishes its magnitude, as in the compile-time
        // solver.
        T best       = T(-1);
        int best_pos = n;
        T local      = T(0);
        for (int K = 0; K < rows_per_thread; ++K) {
            const int pos = tx + K * threads;
            if (pos >= n) break;
            if (pos < i) continue;
            const T v = detail::abs(TDLS_COOP_LUPP_DYN_A(pos, i));
            if (pos == i || v > best) {
                best     = v;
                best_pos = pos;
            }
            if (pos == i) local = v;
        }
        candidate[tx] = best;
        where[tx]     = static_cast<T>(best_pos);
        if (tx == i % threads) in_place[0] = local;
        sync();

        // Every thread finds the same column maximum and the same pivot, as
        // in the compile-time solver.
        T rx_abs_max = candidate[i % threads];
        for (int t = 0; t < threads; t++)
            rx_abs_max = candidate[t] > rx_abs_max ? candidate[t] : rx_abs_max;
        bool keep = !(in_place[0] < rx_abs_max);
        if constexpr (relative_pivot_threshold < T(1)) keep = keeps_pivot(in_place[0], rx_abs_max);
        int max_id = i;
        if (!keep) {
            max_id = n;
            for (int t = 0; t < threads; t++) {
                const int pos = static_cast<int>(where[t]);
                if (candidate[t] == rx_abs_max && pos < max_id) max_id = pos;
            }
        }
        const bool zero_pivot = (rx_abs_max < singular_floor);
        linfo                 = (zero_pivot && linfo == 0) ? (i + 1) : linfo;

        // As in the compile-time solver: the owner of position i publishes
        // its row whole and, when the pivot is not in place, the owner of
        // position max_id publishes the pivot row. Each record holds the n
        // entries of its row, then its right-hand-side entry (fuse_rhs
        // only) and its original index. The rows are addressed by position,
        // and the entries of the slots move through loops kept small, which
        // the compiler unrolls: a slot indexed at run time would demote rB
        // and origin to the stack.
        if (tx == i % threads) save_row(n, A, A_stride, i, leaving);
        if (max_id != i && tx == max_id % threads) save_row(n, A, A_stride, max_id, entering);
        for (int K = 0; K < rows_per_thread; ++K) {
            const int pos = tx + K * threads;
            if (pos != i && (pos != max_id || max_id == i)) continue;
            T* const record = pos == i ? leaving : entering;
            if constexpr (fuse_rhs) record[n] = rB[K];
            record[n + 1] = static_cast<T>(origin[K]);
        }
        sync();

        if (max_id != i) {
            if (tx == i % threads) load_row(n, A, A_stride, i, entering);
            if (tx == max_id % threads) load_row(n, A, A_stride, max_id, leaving);
            for (int K = 0; K < rows_per_thread; ++K) {
                const int pos = tx + K * threads;
                if (pos != max_id && pos != i) continue;
                const T* const record = pos == i ? entering : leaving;
                if constexpr (fuse_rhs) rB[K] = record[n];
                origin[K] = static_cast<int>(record[n + 1]);
            }
        }

        const T* const prow = max_id == i ? leaving : entering;
        const T reg         = zero_pivot ? T(1) : T(1) / prow[i];

        // scal and ger: every thread eliminates its rows below the pivot.
        // The next step writes only the candidates, which no thread reads
        // any more: no barrier closes the step.
        for (int K = 0; K < rows_per_thread; ++K) {
            const int pos = tx + K * threads;
            if (pos >= n) break;
            if (pos > i) {
                TDLS_COOP_LUPP_DYN_A(pos, i) *= reg;
                for (int j = i + 1; j < n; j++)
                    TDLS_COOP_LUPP_DYN_A(pos, j) -= TDLS_COOP_LUPP_DYN_A(pos, i) * prow[j];
                if constexpr (fuse_rhs) rB[K] -= TDLS_COOP_LUPP_DYN_A(pos, i) * prow[n];
            }
        }
    }

    /// \brief Factorization of the rows of the group, in place.
    /// \tparam fuse_rhs apply the elimination to the right-hand side as well
    ///         (the forward pass of solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       right-hand-side entries (fuse_rhs only)
    /// \param[in,out] rpiv     pivot entry of each slot of the thread,
    ///                initialized by init_piv
    /// \param[in]     work     workspace of the group
    /// \param[in]     sync     barrier of the group
    /// \return 0, or 1 + the first column whose best pivot fell below
    ///         singular_floor (MAGMA's linfo)
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    eliminate(const int n, const int tx, const int threads, T* TDLS_RESTRICT A, const int A_stride,
              T (&rB)[rows_per_thread], int (&rpiv)[rows_per_thread], T* work,
              Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        int linfo = 0;
        if constexpr (Config.row_interchange == RowInterchange::Logical) {
            T* const sB  = work;
            T* const sx  = work + n;
            T* const dsx = work + 2 * n;
            for (int i = 0; i < n; i++)
                logical_elimination_step<fuse_rhs>(n, i, tx, threads, A, A_stride, rB, rpiv, sB, sx,
                                                   dsx, linfo, sync);
        } else {
            for (int i = 0; i < n; i++)
                physical_elimination_step<fuse_rhs>(n, i, tx, threads, A, A_stride, rB, rpiv, work,
                                                    linfo, sync);
            sync(); // no barrier closes the steps: this one frees the workspace
        }
        return linfo;
    }

    /* =====================================================================
       SUBSTITUTION, as in the compile-time solver: the forward pass
       replays the elimination of the right-hand side from the factored
       rows, the backward pass publishes column i of U, every thread
       computes x_i, the owner of entry i keeps it and every thread updates
       its entries above. Under physical row interchanges, the right-hand
       side is first gathered in the pivoted order, and both passes run on
       the positions of the rows.
       ===================================================================== */

    /// \brief Forward pass: L y = P b on the given row positions, y
    /// overwriting the right-hand-side entries of the thread.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in]     A        factored matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       right-hand-side entries of the thread
    /// \param[in]     rowid    position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work     workspace of the group
    /// \param[in]     sync     barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    forward_pass(const int n, const int tx, const int threads, const T* TDLS_RESTRICT A,
                 const int A_stride, T (&rB)[rows_per_thread], const int (&rowid)[rows_per_thread],
                 T* work, Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        T* const sB = work;
        for (int i = 0; i < n; i++) {
            for (int K = 0; K < rows_per_thread; ++K) {
                if (tx + K * threads >= n) break;
                if (rowid[K] == i) sB[0] = rB[K];
            }
            sync();

            for (int K = 0; K < rows_per_thread; ++K) {
                const int r = tx + K * threads;
                if (r >= n) break;
                if (rowid[K] > i) rB[K] -= TDLS_COOP_LUPP_DYN_A(r, i) * sB[0];
            }
            sync();
        }
    }

    /// \brief Backward pass: U x = y on the given row positions, the
    /// forward-reduced right-hand side y in the entries of the thread on
    /// entry, its solution entries on exit.
    ///
    /// Every thread computes the same x_i, and the owner of entry i keeps
    /// it in its own entries instead of writing it to the workspace that
    /// every thread reads in the same barrier interval.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in]     A        factored matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       y entries of the rows of the thread on
    ///                entry, solution entries tx + K * threads on exit
    /// \param[in]     rowid    position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work     workspace of the group
    /// \param[in]     sync     barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    backward_pass(const int n, const int tx, const int threads, const T* TDLS_RESTRICT A,
                  const int A_stride, T (&rB)[rows_per_thread], const int (&rowid)[rows_per_thread],
                  T* work, Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        T* const sB = work;
        T* const sx = work + n;
        for (int K = 0; K < rows_per_thread; ++K) {
            if (tx + K * threads >= n) break;
            sB[rowid[K]] = rB[K];
        }
        sync();

        for (int i = n - 1; i >= 0; i--) {
            for (int K = 0; K < rows_per_thread; ++K) {
                const int r = tx + K * threads;
                if (r >= n) break;
                sx[rowid[K]] = TDLS_COOP_LUPP_DYN_A(r, i);
            }
            sync();

            const T reg = sB[i] / sx[i];

            for (int K = 0; K < rows_per_thread; ++K) {
                const int idx = tx + K * threads;
                if (idx >= n) break;
                if (idx < i) sB[idx] = sB[idx] - reg * sx[idx];
                rB[K] = (idx == i) ? reg : rB[K];
            }
            sync();
        }
    }

    /// \brief Forward substitution L y = P b, y overwriting the
    /// right-hand-side entries of the thread.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in]     A        factored matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       right-hand-side entries of the thread
    /// \param[in]     rpiv     pivot entry of each slot of the thread
    /// \param[in]     work     workspace of the group
    /// \param[in]     sync     barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    forward_substitution(const int n, const int tx, const int threads, const T* TDLS_RESTRICT A,
                         const int A_stride, T (&rB)[rows_per_thread],
                         const int (&rpiv)[rows_per_thread], T* work,
                         Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        if constexpr (Config.row_interchange == RowInterchange::Logical) {
            forward_pass(n, tx, threads, A, A_stride, rB, rpiv, work, sync);
        } else {
            // Gather in the pivoted order: each slot receives the entry of
            // the original row it holds. The entries are staged apart from
            // the broadcast slot of the pass, which a faster thread may
            // already write.
            T* const staged = work + n;
            for (int K = 0; K < rows_per_thread; ++K) {
                const int r = tx + K * threads;
                if (r >= n) break;
                staged[r] = rB[K];
            }
            sync();
            for (int K = 0; K < rows_per_thread; ++K) {
                if (tx + K * threads >= n) break;
                rB[K] = staged[rpiv[K]];
            }
            int position[rows_per_thread];
            init_piv(tx, threads, position);
            forward_pass(n, tx, threads, A, A_stride, rB, position, work, sync);
        }
    }

    /// \brief Backward substitution U x = y: the forward-reduced
    /// right-hand side y in the entries of the thread on entry, its
    /// solution entries on exit.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n        system dimension
    /// \param[in]     tx       rank of the calling thread in its group
    /// \param[in]     threads  threads of the group, threads_per_system(n)
    /// \param[in]     A        factored matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in,out] rB       y entries of the rows of the thread on
    ///                entry, solution entries tx + K * threads on exit
    /// \param[in]     rpiv     pivot entry of each slot of the thread
    /// \param[in]     work     workspace of the group
    /// \param[in]     sync     barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    backward_substitution(const int n, const int tx, const int threads, const T* TDLS_RESTRICT A,
                          const int A_stride, T (&rB)[rows_per_thread],
                          const int (&rpiv)[rows_per_thread], T* work,
                          Sync& sync) noexcept(detail::nothrow_sync<Sync>) {
        if constexpr (Config.row_interchange == RowInterchange::Logical) {
            backward_pass(n, tx, threads, A, A_stride, rB, rpiv, work, sync);
        } else {
            int position[rows_per_thread];
            init_piv(tx, threads, position);
            backward_pass(n, tx, threads, A, A_stride, rB, position, work, sync);
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
    /// \param[in] n    system dimension
    /// \param[in] work workspace of the group
    /// \return the deduced barrier, which does nothing for one thread per
    ///         system
    TDLS_EXEC_CHECK_DISABLE
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr auto make_sync(const int n,
                                                                      T* work) noexcept {
        AutoSync deduce;
        return detail::resolve_sync<0>(deduce, work, threads_per_system(n));
    }

    /// \brief In-place LU factorization with partial pivoting.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n          system dimension
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          matrix, pre-offset by the caller; its
    ///                factorization on exit
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        pivot entries (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (factorization unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::AutoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    factorize(const int n, const int tx, T* TDLS_RESTRICT A, const int A_stride,
              int* TDLS_RESTRICT piv, const int piv_stride, T* work,
              Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        const int threads = threads_per_system(n);
        auto&& group      = detail::resolve_sync<0>(sync, work, threads);
        T* const exchange = work + detail::barrier_elements;
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        init_piv(tx, threads, rpiv);
        const int linfo = eliminate<false>(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_piv(n, tx, threads, piv, piv_stride, rpiv);
        group(); // every output visible to the whole group on return
        return linfo == 0;
    }

    /// \brief Solve from a factorization produced by factorize (b and x
    /// distinct).
    /// \tparam Sync callable type of the barrier
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  A          factored matrix
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        pivot entries produced by factorize
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  b          right-hand side, in original order
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of b and x
    /// \param[in]  work       workspace of workspace_size(n) elements,
    ///             shared by the group
    /// \param[in]  sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::AutoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute(const int n, const int tx, const T* TDLS_RESTRICT A, const int A_stride,
               const int* TDLS_RESTRICT piv, const int piv_stride, const T* TDLS_RESTRICT b,
               T* TDLS_RESTRICT x, const int rhs_stride, T* work,
               Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        const int threads = threads_per_system(n);
        auto&& group      = detail::resolve_sync<0>(sync, work, threads);
        T* const exchange = work + detail::barrier_elements;
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_piv(n, tx, threads, piv, piv_stride, rpiv);
        load_rhs(n, tx, threads, b, rhs_stride, rB);
        forward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        backward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
    }

    /// \brief Solve with b = e_col generated on the fly: the
    /// consistent-tangent-operator path.
    /// \tparam Sync callable type of the barrier
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  A          factored matrix
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        pivot entries produced by factorize
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  col        index of the canonical column e_col
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of x
    /// \param[in]  work       workspace of workspace_size(n) elements,
    ///             shared by the group
    /// \param[in]  sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::AutoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_canonical(const int n, const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                         const int* TDLS_RESTRICT piv, const int piv_stride, const int col,
                         T* TDLS_RESTRICT x, const int rhs_stride, T* work,
                         Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        const int threads = threads_per_system(n);
        auto&& group      = detail::resolve_sync<0>(sync, work, threads);
        T* const exchange = work + detail::barrier_elements;
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_piv(n, tx, threads, piv, piv_stride, rpiv);
        for (int K = 0; K < rows_per_thread; ++K)
            rB[K] = (tx + K * threads == col) ? T(1) : T(0);
        forward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        backward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
    }

    /// \brief Solve in place from a factorization produced by factorize:
    /// x holds the right-hand side on entry and the solution on exit.
    ///
    /// Each thread reads and writes only the entries of its own rows, the
    /// same set before and after, so the in-place form needs no copy.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n          system dimension
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in]     A          factored matrix
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        pivot entries produced by factorize
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] x          right-hand side on entry, solution on
    ///                exit
    /// \param[in]     rhs_stride element stride of x
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::AutoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_inplace(const int n, const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                       const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                       const int rhs_stride, T* work,
                       Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        const int threads = threads_per_system(n);
        auto&& group      = detail::resolve_sync<0>(sync, work, threads);
        T* const exchange = work + detail::barrier_elements;
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        load_piv(n, tx, threads, piv, piv_stride, rpiv);
        load_rhs(n, tx, threads, x, rhs_stride, rB);
        forward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        backward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
    }

    /// \brief factorize + substitute in one call, the pivot entries staying
    /// with the threads in between.
    ///
    /// The barrier sequence does not depend on the verdict: a singular
    /// matrix is substituted as well, and x is then unspecified.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n          system dimension
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          on entry the matrix, pre-offset by the
    ///                caller; on exit its factorization (usable for
    ///                further substitute* calls)
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        pivot entries (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (x unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::AutoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve(const int n, const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
          const int piv_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          T* work, Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        const int threads = threads_per_system(n);
        auto&& group      = detail::resolve_sync<0>(sync, work, threads);
        T* const exchange = work + detail::barrier_elements;
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        init_piv(tx, threads, rpiv);
        const int linfo = eliminate<false>(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_piv(n, tx, threads, piv, piv_stride, rpiv);
        load_rhs(n, tx, threads, b, rhs_stride, rB);
        forward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        backward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        group(); // every output visible to the whole group on return
        return linfo == 0;
    }

    /// \brief Factorization with the forward substitution folded in: the
    /// original MAGMA kernel.
    ///
    /// y holds the unpermuted right-hand side on entry and the solution on
    /// exit. The result is bitwise-identical to factorize +
    /// substitute_inplace. The barrier sequence does not depend on the
    /// verdict: a singular matrix is substituted as well, and y is then
    /// unspecified.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n          system dimension
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          on entry the matrix, pre-offset by the
    ///                caller; on exit its factorization
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        pivot entries (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (y unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::AutoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve_inplace(const int n, const int tx, T* TDLS_RESTRICT A, const int A_stride,
                  int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT y,
                  const int rhs_stride, T* work,
                  Sync&& sync = Sync{}) noexcept(detail::nothrow_sync<Sync>) {
        const int threads = threads_per_system(n);
        auto&& group      = detail::resolve_sync<0>(sync, work, threads);
        T* const exchange = work + detail::barrier_elements;
        T rB[rows_per_thread];
        int rpiv[rows_per_thread];
        init_piv(tx, threads, rpiv);
        load_rhs(n, tx, threads, y, rhs_stride, rB);
        const int linfo = eliminate<true>(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        store_piv(n, tx, threads, piv, piv_stride, rpiv);
        backward_substitution(n, tx, threads, A, A_stride, rB, rpiv, exchange, group);
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            TDLS_COOP_LUPP_DYN_Y(r) = rB[K];
        }
        group(); // every output visible to the whole group on return
        return linfo == 0;
    }
};



#undef TDLS_COOP_LUPP_DYN_A
#undef TDLS_COOP_LUPP_DYN_PIV
#undef TDLS_COOP_LUPP_DYN_B
#undef TDLS_COOP_LUPP_DYN_X
#undef TDLS_COOP_LUPP_DYN_Y



} // namespace tdls

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif



#endif // TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_DYNAMIC_HPP
