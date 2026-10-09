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
/// One system is solved by a group of threads_per_system threads that
/// share its rows: thread tx holds rows tx, tx + threads_per_system,
/// tx + 2 * threads_per_system and so on, rows_per_thread of them, in
/// registers. The threads cooperate through a small workspace (shared
/// memory on GPU) and a barrier provided by the caller. The numerical
/// algorithm is the right-looking LU with partial pivoting of MAGMA's
/// register-blocked small-system kernel, unchanged: same operations, same
/// order, same pivot choice. Pivoting is logical: rows never move, each
/// thread tracks the position of its rows in the pivoted order (MAGMA's
/// rowid) and the pivot array records it. The mapping of rows to threads
/// changes nothing in the arithmetic, so every value of rows_per_thread
/// produces bitwise-identical results on identical inputs. The runtime
/// dimension variant is CooperativeLUppSolverDynamic (solver_dynamic.hpp).
///
/// **Calling convention.** Every thread of the group calls the same entry
/// point, with the same template arguments, the same operands, the same
/// workspace and its own rank tx in [0, threads_per_system). The solver
/// never sees a global thread index: callers pre-offset every remote
/// pointer with the system index and pass a runtime stride, as for the
/// TiledLUpp solvers. The barrier `sync` is a callable taking no
/// argument, invoked by every thread of the group; it must make the
/// memory writes of each thread visible to all of them, in the workspace
/// and in the operands. Under CUDA,
/// `[mask] { __syncwarp(mask); }` with the lanes of the group in mask is
/// the natural choice; a SYCL group barrier, a Kokkos team barrier or
/// a CPU thread barrier serve the same purpose. With one thread per system
/// (rows_per_thread >= N) the barrier may be omitted, and the default
/// tdls::NoSync applies; with several threads it is mandatory, a
/// contract checked at compile time.
///
/// **Fixed barrier sequence.** Every entry point executes a sequence of
/// barriers set by N, rows_per_thread and the entry point alone: it never
/// depends on the data, not even on a singular verdict, which is returned
/// at the end instead of by an early exit. A barrier wider than the group
/// (a whole warp, a work-group, a team) is therefore valid as well,
/// provided every thread of its scope makes the same sequence of calls.
/// Between two barriers, no workspace element is written by one thread
/// while another thread accesses it, so the solver is free of data races
/// even when the threads of a group do not run in lockstep.
///
/// **Inputs on entry.** Each thread reads only the entries of its own
/// rows. A caller that builds the rows of each thread in that thread
/// needs no barrier before the call.
///
/// **Results on return.** Every entry point ends with a barrier. When a
/// thread returns, every thread of the group has written its results
/// (factored rows, row positions, solution entries), and they are
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
///   - matrix: external element (r, c) lives at
///     `A[((r)*N+(c))*A_stride]` under the default row-major layout,
///     `A[((c)*N+(r))*A_stride]` under the column-major one. The internal
///     slice holds the rows_per_thread rows of the thread: element (K, c),
///     column c of row tx + K * threads_per_system, lives at
///     `A[K*N+c]` row-major and `A[c*rows_per_thread+K]` column-major.
///   - pivot: external entry r holds the position of physical row r in the
///     pivoted order; the internal slice holds that position for the rows
///     of the thread, entry K for row tx + K * threads_per_system. This
///     maps physical to logical rows, the inverse of the TiledLUpp
///     convention.
///   - right-hand side and solution: external entry r is row r; the
///     internal slice holds entry tx + K * threads_per_system at index K.
///
/// Offsets are computed in 32-bit arithmetic, with the bounds of the
/// TiledLUpp solvers. The single form `base[e*stride]` covers every
/// batched layout: AoS (stride 1), SoA (stride = batch size), AoSoA
/// (stride = W).
///
/// **Factored format.** The factorization stays in the physical rows:
/// physical row r holds, before the column of its pivot step, the
/// multipliers of L, and from that column on, its row of U. The diagonal
/// holds the pivots themselves, not their reciprocals. A factorization
/// produced here must be consumed by the substitution routines of this
/// solver.
///
/// **Workspace.** workspace_size = 3 * N elements per group, MAGMA's sB,
/// sx and dsx, in this order. The workspace pointer carries no restrict
/// qualifier: the other threads of the group write the workspace between
/// two barriers, and a restrict pointer would let the compiler reuse a
/// value read before the barrier. Every other operand keeps the
/// qualifier, since each thread only accesses its own rows of it.
///
/// **Departures from the MAGMA kernel.** The arithmetic is MAGMA's; the
/// following changes are structural.
///   - rows_per_thread rows per thread instead of one, phantom rows
///     skipped at compile time, so that several systems share a warp.
///   - Device-callable entry points with a caller-provided barrier,
///     instead of a kernel with one system per block.
///   - The back substitution keeps the solution entry x_i in a register of
///     the thread owning entry i. MAGMA writes it to sB[i] in the very
///     barrier interval where every thread reads sB[i]: correct in
///     lockstep execution only, a data race otherwise.
///   - The pivot output is the logical position of each row (rowid), not
///     the LAPACK swap sequence, and the matrix keeps its physical row
///     order: a separate substitution can then rebuild its state from the
///     factored rows alone. The swap sequence (sipiv) is not computed.
///   - A pivot below Config.singular_floor is singular, instead of an
///     exactly zero one. The default floor only adds the subnormal pivots,
///     the criterion of every TDLS solver.
///   - Real scalar types: the magnitude of the pivot search is the
///     absolute value, as in the real-precision variants of MAGMA.



#include <type_traits>
#include <utility>

#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/solvers/cooperative_lupp/barrier.hpp>
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
   rhs_stride), the residency template booleans, the Config value and the
   class constants are in scope. K is a row slot of the calling thread,
   its row is tx + K * threads_per_system. #undef'd at the end of this
   header. */

/// \def TDLS_COOP_LUPP_A
/// \brief Element (row of slot K, c) of the matrix: slice of the calling
/// thread under internal residency, whole strided matrix otherwise, flat
/// index remapped by Config.layout.
#define TDLS_COOP_LUPP_A(K, c)                                                                     \
    A[internal_matrix                                                                              \
          ? (Config.layout == tdls::MatrixLayout::RowMajor                                         \
                 ? unsigned((K) * N + (c))                                                         \
                 : unsigned((c) * rows_per_thread + (K)))                                          \
          : TDLS_LAYOUT_INDEX(tx + (K) * threads_per_system, c, N) * unsigned(A_stride)]
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



/// \brief Dense LU factorization with logical partial pivoting held in
/// the registers of a group of threads, solving one NxN system per group.
///
/// All entry points are static, host- and device-callable, called by
/// every thread of the group with its rank tx, and take raw pointers
/// pre-offset by the caller plus one runtime stride per array (see the
/// file documentation for the calling convention). Entry points:
///   - factorize:             A := P*L*U in place, row positions out
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
///         thread, singularity floor, unroll policy, matrix layout; see
///         CooperativeLUppConfig
template<typename T, int N, CooperativeLUppConfig<T> Config = CooperativeLUppConfig<T>{}>
struct CooperativeLUppSolverStatic {

    static_assert(N >= 1, "CooperativeLUppSolverStatic: N must be >= 1");
    static_assert(Config.rows_per_thread >= 1,
                  "CooperativeLUppSolverStatic: rows_per_thread must be >= 1");
    static_assert(Config.singular_floor.is_finite(),
                  "CooperativeLUppSolverStatic: singular_floor must be finite (and fit a 63-bit "
                  "mantissa)");

    /// \brief Rows held by each thread: Config.rows_per_thread, or N when
    /// it is larger. A value below 1, rejected above, is clamped to 1 so
    /// that the rejection stays the only diagnostic.
    static constexpr int rows_per_thread = Config.rows_per_thread < 1   ? 1
                                           : Config.rows_per_thread < N ? Config.rows_per_thread
                                                                        : N;
    /// \brief Threads solving one system: ceil(N / rows_per_thread).
    static constexpr int threads_per_system = (N + rows_per_thread - 1) / rows_per_thread;
    /// \brief Elements of the workspace shared by the threads of a group.
    static constexpr int workspace_size = 3 * N;
    /// \brief Singularity floor, read once from the configuration (see
    /// CooperativeLUppConfig::singular_floor).
    static constexpr T singular_floor = Config.singular_floor;

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

    /// \brief Compile-time contract of the barrier: a group of several
    /// threads cannot run without one.
    /// \tparam Sync callable type of the barrier
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void require_barrier() noexcept {
        static_assert(threads_per_system == 1 ||
                          !std::is_same_v<std::remove_cvref_t<Sync>, tdls::NoSync>,
                      "CooperativeLUppSolverStatic: a group of several threads needs a barrier "
                      "(sync argument)");
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

    /// \brief Initial row positions: every row at its own index, phantom
    /// rows included (theirs stays at or above N).
    /// \param[in]  tx    rank of the calling thread in its group
    /// \param[out] rowid position of each row of the thread in the pivoted
    ///             order
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    init_rowid(const int tx, int (&rowid)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            rowid[K]        = tx + K * threads_per_system;
        });
    }

    /// \brief Load the row positions recorded by factorize.
    /// \tparam internal_piv residency of piv
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[in]  piv        row positions, or slice of the thread
    ///             (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[out] rowid      position of each row of the thread in the
    ///             pivoted order
    template<bool internal_piv>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    load_rowid(const int tx, const int* TDLS_RESTRICT piv, const int piv_stride,
               int (&rowid)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if (slot_is_full(K) || tx + K * threads_per_system < N)
                rowid[K] = TDLS_COOP_LUPP_PIV(K);
            else
                rowid[K] = tx + K * threads_per_system;
        });
    }

    /// \brief Record the row positions of the calling thread.
    /// \tparam internal_piv residency of piv
    /// \param[in]  tx         rank of the calling thread in its group
    /// \param[out] piv        row positions, or slice of the thread
    ///             (caller-pre-offset)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  rowid      position of each row of the thread in the
    ///             pivoted order
    template<bool internal_piv>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    store_rowid(const int tx, int* TDLS_RESTRICT piv, const int piv_stride,
                const int (&rowid)[rows_per_thread]) noexcept {
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N)
                    TDLS_COOP_LUPP_PIV(K) = rowid[K];
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
       Three barrier intervals per column: the magnitudes of the column are
       published at the positions of their rows, every thread finds the
       same pivot, the owner of the pivot row publishes it, and every
       thread eliminates its own rows below the pivot.
       ===================================================================== */

    /// \brief One column of the factorization.
    /// \tparam fuse_rhs apply the elimination to the right-hand side as well
    ///         (the forward pass of solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in]     i     column of this step
    /// \param[in,out] rA    register rows of the thread
    /// \param[in,out] rB    right-hand-side registers (fuse_rhs only)
    /// \param[in,out] rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     sB    workspace: broadcast slot of the pivot
    ///                right-hand side (fuse_rhs only)
    /// \param[in]     sx    workspace: pivot row
    /// \param[in]     dsx   workspace: magnitudes of the column
    /// \param[in,out] linfo 0, or 1 + the first column whose best pivot
    ///                fell below singular_floor
    /// \param[in]     sync  barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool fuse_rhs, typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    elimination_step(const int i, T (&rA)[rows_per_thread][N],
                     [[maybe_unused]] T (&rB)[rows_per_thread], int (&rowid)[rows_per_thread],
                     T* sB, T* sx, T* dsx, int& linfo,
                     Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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
                    if constexpr (fuse_rhs) sB[0] = rB[K];
                } else if (rowid[K] == i) {
                    rowid[K] = max_id;
                }
            }
        });
        sync();

        const T reg = zero_pivot ? T(1) : T(1) / sx[i];

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
                    if constexpr (fuse_rhs) rB[K] -= rA[K][i] * sB[0];
                }
            }
        });
        sync();
    }

    /// \brief Factorization of the register rows of the group.
    /// \tparam fuse_rhs apply the elimination to the right-hand side as well
    ///         (the forward pass of solve_inplace)
    /// \tparam Sync     callable type of the barrier
    /// \param[in,out] rA    register rows of the thread
    /// \param[in,out] rB    right-hand-side registers (fuse_rhs only)
    /// \param[in,out] rowid position of each row of the thread in the
    ///                pivoted order, initialized by init_rowid
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    /// \return 0, or 1 + the first column whose best pivot fell below
    ///         singular_floor (MAGMA's linfo)
    template<bool fuse_rhs, typename Sync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    eliminate(T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread], int (&rowid)[rows_per_thread],
              T* work, Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        T* const sB  = work;
        T* const sx  = work + N;
        T* const dsx = work + 2 * N;
        int linfo    = 0;
        if constexpr (Config.unroll_loops) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < N; i++)
                elimination_step<fuse_rhs>(i, rA, rB, rowid, sB, sx, dsx, linfo, sync);
        } else {
            for (int i = 0; i < N; i++)
                elimination_step<fuse_rhs>(i, rA, rB, rowid, sB, sx, dsx, linfo, sync);
        }
        return linfo;
    }

    /* =====================================================================
       SUBSTITUTION.
       Forward: the owner of the row at position i publishes its
       right-hand-side entry, every thread updates its rows below. This is
       the elimination of the right-hand side that solve_inplace folds into
       the factorization, replayed from the factored rows. Backward: the
       right-hand side is staged at the positions of its rows, then column
       i of U is published, every thread computes x_i, the owner of entry i
       keeps it and every thread updates its entries above.
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
                 Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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

    /// \brief Forward substitution L y = P b, y overwriting the
    /// right-hand-side registers.
    /// \tparam Sync callable type of the barrier
    /// \param[in]     rA    factored register rows of the thread
    /// \param[in,out] rB    right-hand-side registers of the thread
    /// \param[in]     rowid position of each row of the thread in the
    ///                pivoted order
    /// \param[in]     work  workspace of the group
    /// \param[in]     sync  barrier of the group
    template<typename Sync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    forward_substitution(const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                         const int (&rowid)[rows_per_thread], T* work,
                         Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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
                  T (&rB)[rows_per_thread], const int (&rowid)[rows_per_thread],
                  T* sB, T* sx,
                  Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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

    /// \brief Backward substitution U x = y: the forward-reduced
    /// right-hand side y in the registers on entry, the solution entries
    /// of the thread on exit.
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
    backward_substitution(const int tx, const T (&rA)[rows_per_thread][N], T (&rB)[rows_per_thread],
                          const int (&rowid)[rows_per_thread], T* work,
                          Sync& sync) noexcept(std::is_nothrow_invocable_v<Sync&>) {
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

    /* =====================================================================
       Entry points.
       ===================================================================== */

    /// \brief In-place LU factorization with logical partial pivoting.
    /// \tparam internal_piv    residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam Sync            callable type of the barrier
    /// \param[in]     tx         rank of the calling thread in its group
    /// \param[in,out] A          matrix, or slice of the thread, pre-offset
    ///                by the caller; its factorization on exit
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        row positions, or slice of the thread
    ///                (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (factorization unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_piv, bool internal_matrix, typename Sync = tdls::NoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    factorize(const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
              const int piv_stride, T* work,
              Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        require_barrier<Sync>();
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        init_rowid(tx, rowid);
        const int linfo = eliminate<false>(rA, rB, rowid, work, sync);
        store_rows<internal_matrix>(tx, A, A_stride, rA);
        store_rowid<internal_piv>(tx, piv, piv_stride, rowid);
        sync(); // every output visible to the whole group on return
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
    /// \param[in]  piv        row positions produced by factorize
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  b          right-hand side, in original order
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of b and x (external mode)
    /// \param[in]  work       workspace of workspace_size elements, shared
    ///             by the group
    /// \param[in]  sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::NoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
               const int* TDLS_RESTRICT piv, const int piv_stride, const T* TDLS_RESTRICT b,
               T* TDLS_RESTRICT x, const int rhs_stride, T* work,
               Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        require_barrier<Sync>();
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        load_rowid<internal_piv>(tx, piv, piv_stride, rowid);
        load_rhs<internal_rhs>(tx, b, rhs_stride, rB);
        forward_substitution(rA, rB, rowid, work, sync);
        backward_substitution(tx, rA, rB, rowid, work, sync);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
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
    /// \param[in]  piv        row positions produced by factorize
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  col        index of the canonical column e_col
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of x (external mode)
    /// \param[in]  work       workspace of workspace_size elements, shared
    ///             by the group
    /// \param[in]  sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::NoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_canonical(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                         const int* TDLS_RESTRICT piv, const int piv_stride, const int col,
                         T* TDLS_RESTRICT x, const int rhs_stride, T* work,
                         Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        require_barrier<Sync>();
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        load_rowid<internal_piv>(tx, piv, piv_stride, rowid);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            rB[K]           = (tx + K * threads_per_system == col) ? T(1) : T(0);
        });
        forward_substitution(rA, rB, rowid, work, sync);
        backward_substitution(tx, rA, rB, rowid, work, sync);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
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
    /// \param[in]     piv        row positions produced by factorize
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] x          right-hand side on entry, solution on
    ///                exit
    /// \param[in]     rhs_stride element stride of x (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::NoSync>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    substitute_inplace(const int tx, const T* TDLS_RESTRICT A, const int A_stride,
                       const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                       const int rhs_stride, T* work,
                       Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        require_barrier<Sync>();
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        load_rowid<internal_piv>(tx, piv, piv_stride, rowid);
        load_rhs<internal_rhs>(tx, x, rhs_stride, rB);
        forward_substitution(rA, rB, rowid, work, sync);
        backward_substitution(tx, rA, rB, rowid, work, sync);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
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
    /// \param[out]    piv        row positions, or slice of the thread
    ///                (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (x unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::NoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve(const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
          const int piv_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          T* work,
          Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        require_barrier<Sync>();
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        init_rowid(tx, rowid);
        const int linfo = eliminate<false>(rA, rB, rowid, work, sync);
        store_rows<internal_matrix>(tx, A, A_stride, rA);
        store_rowid<internal_piv>(tx, piv, piv_stride, rowid);
        load_rhs<internal_rhs>(tx, b, rhs_stride, rB);
        forward_substitution(rA, rB, rowid, work, sync);
        backward_substitution(tx, rA, rB, rowid, work, sync);
        store_solution<internal_rhs>(tx, x, rhs_stride, rB);
        sync(); // every output visible to the whole group on return
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
    /// \param[out]    piv        row positions, or slice of the thread
    ///                (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \param[in]     work       workspace of workspace_size elements,
    ///                shared by the group
    /// \param[in]     sync       barrier of the group
    /// \return false on a singular matrix (y unspecified).
    TDLS_EXEC_CHECK_DISABLE
    template<bool internal_rhs, bool internal_piv, bool internal_matrix,
             typename Sync = tdls::NoSync>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr bool
    solve_inplace(const int tx, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                  const int piv_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                  T* work,
                  Sync&& sync = Sync{}) noexcept(std::is_nothrow_invocable_v<Sync&>) {
        require_barrier<Sync>();
        T rA[rows_per_thread][N];
        T rB[rows_per_thread];
        int rowid[rows_per_thread];
        load_rows<internal_matrix>(tx, A, A_stride, rA);
        init_rowid(tx, rowid);
        load_rhs<internal_rhs>(tx, y, rhs_stride, rB);
        const int linfo = eliminate<true>(rA, rB, rowid, work, sync);
        store_rows<internal_matrix>(tx, A, A_stride, rA);
        store_rowid<internal_piv>(tx, piv, piv_stride, rowid);
        backward_substitution(tx, rA, rB, rowid, work, sync);
        detail::unroll_K<rows_per_thread>([&](auto K_tag) {
            constexpr int K = decltype(K_tag)::value;
            if constexpr (slot_has_rows(K)) {
                if (slot_is_full(K) || tx + K * threads_per_system < N) TDLS_COOP_LUPP_Y(K) = rB[K];
            }
        });
        sync(); // every output visible to the whole group on return
        return linfo == 0;
    }
};



#undef TDLS_COOP_LUPP_A
#undef TDLS_COOP_LUPP_PIV
#undef TDLS_COOP_LUPP_B
#undef TDLS_COOP_LUPP_X
#undef TDLS_COOP_LUPP_Y



} // namespace tdls

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_COOPERATIVE_LUPP_SOLVER_STATIC_HPP
