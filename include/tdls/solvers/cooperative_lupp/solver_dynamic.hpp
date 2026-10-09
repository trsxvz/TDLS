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
/// Same algorithm as CooperativeLUppSolverStatic (solver_static.hpp): the
/// LU with logical partial pivoting of MAGMA's register-blocked
/// small-system kernel, the rows of a system shared by a group of
/// threads, a workspace and a caller-provided barrier, the row positions
/// as pivot output and the pivots on the factored diagonal. The
/// difference is that the system dimension n is a runtime function
/// parameter instead of a template parameter. Only the number of rows per
/// thread (Config.rows_per_thread) stays compile-time: it sizes the row
/// positions and the right-hand-side entries kept by each thread, while
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
///   - The barrier contract cannot be checked at compile time, the number
///     of threads depending on n: with the default tdls::NoSync, the
///     caller must keep n <= Config.rows_per_thread (one thread per
///     system).
///
/// For equal shapes (n = N, same configuration), results are bitwise
/// identical to the compile-time solver: factored rows, row positions and
/// solution. The arithmetic sequence is the same, only the storage of the
/// rows and the loop mechanics differ.
///
/// The compile-time solver is the performance path; this variant is the
/// flexibility path (dimensions unknown at compile time, fast builds).
///
/// The calling convention, the factored format and the pivot convention
/// are those of the compile-time solver, described in solver_static.hpp.
/// So are the guarantees of the group: a fixed sequence of barriers, no
/// data race, inputs read by their owners on entry, results visible to
/// the whole group on return. The workspace holds workspace_size(n) =
/// 3 * n elements, without the restrict qualifier, for the reason given
/// there. The matrix layout follows Config.layout: row-major by
/// default, the flat index remapped for column-major storage, resolved
/// at compile time.
///
/// Preconditions: n >= 1 and 0 <= tx < threads_per_system(n). Offsets are
/// computed in 32-bit arithmetic: the flat element index (for the matrix,
/// n*n) must stay below 2^31 and the largest element offset of every
/// array (for the matrix, (n*n-1)*A_stride) below 2^32.



#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/solvers/cooperative_lupp/barrier.hpp>
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



/// \brief Runtime-size dense LU factorization with logical partial
/// pivoting shared by a group of threads, solving one n x n system per
/// group.
///
/// All entry points are static, host- and device-callable, called by
/// every thread of the group with the dimension n and its rank tx, and
/// take raw pointers pre-offset by the caller with one runtime stride per
/// array. Entry points:
///   - factorize:             A := P*L*U in place, row positions out
///   - substitute:            x := A^-1 b from a factorization (b and x
///     distinct)
///   - substitute_canonical:  idem with b = e_col (tangent-operator columns)
///   - substitute_inplace:    idem with b == x
///   - solve:                 factorize + substitute
///   - solve_inplace:         factorize with the forward pass folded in,
///     the original MAGMA kernel
///
/// The factored diagonal holds the pivots and the pivot array the
/// position of each row, exactly as in CooperativeLUppSolverStatic: a
/// factorization produced here must be consumed by the substitution
/// routines of this family.
///
/// \tparam T      scalar type (float, double or long double)
/// \tparam Config compile-time knobs, passed as a constexpr value: rows per
///         thread, singularity floor, matrix layout; see
///         CooperativeLUppConfig (unroll_loops is ignored by this
///         variant)
template<typename T, CooperativeLUppConfig<T> Config = CooperativeLUppConfig<T>{}>
struct CooperativeLUppSolverDynamic {

    static_assert(Config.rows_per_thread >= 1,
                  "CooperativeLUppSolverDynamic: rows_per_thread must be >= 1");
    static_assert(Config.singular_floor.is_finite(),
                  "CooperativeLUppSolverDynamic: singular_floor must be finite (and fit a 63-bit "
                  "mantissa)");

    /// \brief Row slots of each thread: Config.rows_per_thread. When n is
    /// smaller, one thread solves the system and its slots beyond n hold
    /// phantom rows. A value below 1, rejected above, is clamped to 1 so
    /// that the rejection stays the only diagnostic.
    static constexpr int rows_per_thread = Config.rows_per_thread < 1 ? 1 : Config.rows_per_thread;
    /// \brief Singularity floor, read once from the configuration (see
    /// CooperativeLUppConfig::singular_floor).
    static constexpr T singular_floor = Config.singular_floor;

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

    /// \brief Elements of the workspace shared by the threads of a group.
    /// \param[in] n system dimension
    /// \return the number of elements of the workspace
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int workspace_size(const int n) noexcept {
        return 3 * n;
    }

    /* =====================================================================
       Row positions and right-hand-side entries.
       Slot K of thread tx holds row tx + K * threads, a phantom row when
       it reaches n. The rows themselves stay in the matrix; only their
       positions in the pivoted order and their right-hand-side entries
       are kept by the thread. Each thread reads and writes only the
       entries of its own rows, so the operands never carry data between
       threads: the workspace does.
       ===================================================================== */

    /// \brief Initial row positions: every row at its own index, phantom
    /// rows included (theirs stays at or above n).
    /// \param[in]  tx      rank of the calling thread in its group
    /// \param[in]  threads threads of the group, threads_per_system(n)
    /// \param[out] rowid   position of each row of the thread in the
    ///             pivoted order
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    init_rowid(const int tx, const int threads, int (&rowid)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K)
            rowid[K] = tx + K * threads;
    }

    /// \brief Load the row positions recorded by factorize.
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  threads    threads of the group, threads_per_system(n)
    /// \param[in]  piv        row positions (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv
    /// \param[out] rowid      position of each row of the thread in the
    ///             pivoted order
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_rowid(const int n, const int tx, const int threads, const int* TDLS_RESTRICT piv,
               const int piv_stride, int (&rowid)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            rowid[K]    = r < n ? TDLS_COOP_LUPP_DYN_PIV(r) : r;
        }
    }

    /// \brief Record the row positions of the calling thread.
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  threads    threads of the group, threads_per_system(n)
    /// \param[out] piv        row positions (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  rowid      position of each row of the thread in the
    ///             pivoted order
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_rowid(const int n, const int tx, const int threads, int* TDLS_RESTRICT piv,
                const int piv_stride, const int (&rowid)[rows_per_thread]) noexcept {
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            TDLS_COOP_LUPP_DYN_PIV(r) = rowid[K];
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
       compile-time solver: the magnitudes of the column are published at
       the positions of their rows, every thread finds the same pivot, the
       owner of the pivot row publishes it, and every thread eliminates its
       own rows below the pivot, in place in the matrix.
       ===================================================================== */

    /// \brief One column of the factorization.
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
    elimination_step(const int n, const int i, const int tx, const int threads, T* TDLS_RESTRICT A,
                     const int A_stride, [[maybe_unused]] T (&rB)[rows_per_thread],
                     int (&rowid)[rows_per_thread], T* sB, T* sx,
                     T* dsx, int& linfo,
                     Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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
    /// \param[in,out] rowid    position of each row of the thread in the
    ///                pivoted order, initialized by init_rowid
    /// \param[in]     work     workspace of the group
    /// \param[in]     sync     barrier of the group
    /// \return 0, or 1 + the first column whose best pivot fell below
    ///         singular_floor (MAGMA's linfo)
    template<bool fuse_rhs, typename Sync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    eliminate(const int n, const int tx, const int threads, T* TDLS_RESTRICT A, const int A_stride,
              T (&rB)[rows_per_thread], int (&rowid)[rows_per_thread], T* work,
              Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        T* const sB  = work;
        T* const sx  = work + n;
        T* const dsx = work + 2 * n;
        int linfo    = 0;
        for (int i = 0; i < n; i++)
            elimination_step<fuse_rhs>(n, i, tx, threads, A, A_stride, rB, rowid, sB, sx, dsx,
                                       linfo, sync);
        return linfo;
    }

    /* =====================================================================
       SUBSTITUTION, as in the compile-time solver: the forward pass
       replays the elimination of the right-hand side from the factored
       rows, the backward pass publishes column i of U, every thread
       computes x_i, the owner of entry i keeps it and every thread updates
       its entries above.
       ===================================================================== */

    /// \brief Forward substitution L y = P b, y overwriting the
    /// right-hand-side entries of the thread.
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
    forward_substitution(const int n, const int tx, const int threads, const T* TDLS_RESTRICT A,
                         const int A_stride, T (&rB)[rows_per_thread],
                         const int (&rowid)[rows_per_thread], T* work,
                         Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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

    /// \brief Backward substitution U x = y: the forward-reduced
    /// right-hand side y in the entries of the thread on entry, its
    /// solution entries on exit.
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
    backward_substitution(const int n, const int tx, const int threads, const T* TDLS_RESTRICT A,
                          const int A_stride, T (&rB)[rows_per_thread],
                          const int (&rowid)[rows_per_thread], T* work,
                          Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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

    /* =====================================================================
       Entry points.
       ===================================================================== */

    /// \brief In-place LU factorization with logical partial pivoting.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     n          system dimension
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          matrix, pre-offset by the caller; its
    ///                factorization on exit
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        row positions (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (factorization unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::NoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    factorize(const int n, const int tx, T* TDLS_RESTRICT A, const int A_stride,
              int* TDLS_RESTRICT piv, const int piv_stride, T* work,
              Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        const int threads = threads_per_system(n);
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        init_rowid(tx, threads, rowid);
        const int linfo = eliminate<false>(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_rowid(n, tx, threads, piv, piv_stride, rowid);
        sync(); // every output visible to the whole group on return
        return linfo == 0;
    }

    /// \brief Solve from a factorization produced by factorize (b and x
    /// distinct).
    /// \tparam Sync callable type of the barrier
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  A          factored matrix
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        row positions produced by factorize
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  b          right-hand side, in original order
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of b and x
    /// \param[in]  work       workspace of workspace_size(n) elements,
    ///             shared by the group
    /// \param[in]  sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::NoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute(const int n, const int tx, const T* TDLS_RESTRICT A, const int A_stride,
               const int* TDLS_RESTRICT piv, const int piv_stride, const T* TDLS_RESTRICT b,
               T* TDLS_RESTRICT x, const int rhs_stride, T* work,
               Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        const int threads = threads_per_system(n);
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rowid(n, tx, threads, piv, piv_stride, rowid);
        load_rhs(n, tx, threads, b, rhs_stride, rB);
        forward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        backward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
    }

    /// \brief Solve with b = e_col generated on the fly: the
    /// consistent-tangent-operator path.
    /// \tparam Sync callable type of the barrier
    /// \param[in]  n          system dimension
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  A          factored matrix
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        row positions produced by factorize
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  col        index of the canonical column e_col
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of x
    /// \param[in]  work       workspace of workspace_size(n) elements,
    ///             shared by the group
    /// \param[in]  sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::NoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_canonical(const int n, const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                         const int* TDLS_RESTRICT piv, const int piv_stride, const int col,
                         T* TDLS_RESTRICT x, const int rhs_stride, T* work,
                         Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        const int threads = threads_per_system(n);
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rowid(n, tx, threads, piv, piv_stride, rowid);
        for (int K = 0; K < rows_per_thread; ++K)
            rB[K] = (tx + K * threads == col) ? T(1) : T(0);
        forward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        backward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
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
    /// \param[in]     piv        row positions produced by factorize
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] x          right-hand side on entry, solution on
    ///                exit
    /// \param[in]     rhs_stride element stride of x
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::NoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_inplace(const int n, const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                       const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                       const int rhs_stride, T* work,
                       Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        const int threads = threads_per_system(n);
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rowid(n, tx, threads, piv, piv_stride, rowid);
        load_rhs(n, tx, threads, x, rhs_stride, rB);
        forward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        backward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
    }

    /// \brief factorize + substitute in one call, the row positions staying
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
    /// \param[out]    piv        row positions (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (x unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::NoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve(const int n, const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
          const int piv_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          T* work,
          Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        const int threads = threads_per_system(n);
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        init_rowid(tx, threads, rowid);
        const int linfo = eliminate<false>(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_rowid(n, tx, threads, piv, piv_stride, rowid);
        load_rhs(n, tx, threads, b, rhs_stride, rB);
        forward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        backward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_solution(n, tx, threads, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
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
    /// \param[out]    piv        row positions (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y
    /// \param[in]     work       workspace of workspace_size(n) elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (y unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<typename Sync = tdls::NoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve_inplace(const int n, const int tx, T* TDLS_RESTRICT A, const int A_stride,
                  int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT y,
                  const int rhs_stride, T* work,
                  Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        const int threads = threads_per_system(n);
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        init_rowid(tx, threads, rowid);
        load_rhs(n, tx, threads, y, rhs_stride, rB);
        const int linfo = eliminate<true>(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        store_rowid(n, tx, threads, piv, piv_stride, rowid);
        backward_substitution(n, tx, threads, A, A_stride, rB, rowid, work, sync);
        for (int K = 0; K < rows_per_thread; ++K) {
            const int r = tx + K * threads;
            if (r >= n) break;
            TDLS_COOP_LUPP_DYN_Y(r) = rB[K];
        }
        sync(); // every output visible to the whole group on return
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
