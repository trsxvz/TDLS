#ifndef TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_STATIC_HPP
#define TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_STATIC_HPP



/// \file
/// \brief Compile-time-shape solver of arrowhead (bordered block-diagonal)
/// systems by static condensation, built on TiledLUpp.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The system has B border unknowns first, then NB blocks of S unknowns:
///
///     [ Abb  Ab0  Ab1 ... ] [xb]   [yb]
///     [ A0b  D0           ] [x0]   [y0]
///     [ A1b       D1      ] [x1] = [y1]
///     [ ...           ... ] [..]   [..]
///
/// Two blocks never couple. The solver reads only the entries of this
/// pattern: the zeros are never read, so they need not be stored (see
/// storage_map.hpp).
///
/// Algorithm. First pass, per block k:
///   1. D_k is copied into a local array and factored by TiledLUpp, a
///      TiledLUppSolverStatic of dimension S.
///   2. The pivots are checked on the factors: every multiplier of the
///      elimination of the block stays below 1 / condensation_pivot_threshold,
///      for its own rows (L) and for the border rows that share its
///      columns (M = Cb U^-1). A block that fails, singular included, is
///      refused and left untouched for the second pass.
///   3. Otherwise the block is condensed: its contribution goes to the
///      Schur complement of the border, its recovery data in place of A_kb
///      and y_k (see below).
/// Second pass, only when blocks were refused, in the workspace: each
/// refused block is eliminated together with the border rows, by partial
/// pivoting over its S rows and the B border rows. A border row may serve
/// as pivot; a row of the block then takes its place in the border. The
/// border keeps B rows whatever the number of refused blocks. Its rows
/// carry their couplings to the refused blocks still to come through a
/// mixing matrix T (B x S under the diagonal coupling, B x B otherwise).
/// Each refused block leaves x_k = c_k - G_k xb - H_k w, where w gathers
/// the couplings A_bj x_j of the refused blocks after it.
/// Then the border system (B x B) is solved by TiledLUpp, a
/// TiledLUppSolverStatic of dimension B, and the blocks are recovered,
/// the last one first.
///
/// Condensation of an accepted block, per layout:
///   - column-major, in one pass: the forward stage U^-T [A_kb | y_k] of
///     the solve (TiledLUpp's forward_inplace_multirhs) holds the border
///     multipliers of the check; the backward stage completes
///     W_k = D_k^-1 [A_kb | y_k], stored in place of A_kb and y_k; the
///     Schur complement receives -A_bk W_k; x_k = W_k(:, B) - W_k(:, 0:B) xb;
///   - row-major, diagonal coupling (and column-major in several passes):
///     the same with the whole solve, the check computing M = Cb U^-1
///     apart;
///   - row-major, dense coupling: V_k = L^-1 P [A_kb | y_k], the forward
///     stage alone, is stored in place of A_kb and y_k, U in the upper
///     triangle of D_k; the Schur complement receives -M V_k (M from the
///     check: A_bk D_k^-1 = M L^-1 P); x_k = U^-1 (V_k(:, B) - V_k(:, 0:B) xb).
///
/// A refused block costs the second pass, never a failure: the solver
/// succeeds whenever the full matrix is invertible, rounding aside. Like
/// TiledLUpp, the solver does not detect non-finite entries: a block with
/// one fails the pivot check and goes to the second pass, where they
/// spread to the solution. A true return is no proof of a finite
/// solution.
///
/// On exit y holds the solution. A holds the recovery data in place of
/// A_kb, and in place of D_k for the refused blocks and, under the
/// row-major layout with the dense coupling, for the accepted ones too;
/// with schur_in_place, the coupled border rows of A and y hold the Schur
/// complement. The other entries are only read.
///
/// The caller pre-offsets every remote pointer with the system index and
/// passes a runtime stride, as for the other families; the
/// `internal_rhs` / `internal_work` / `internal_matrix` template booleans
/// declare a caller-local array instead (stride ignored). The workspace
/// holds the front of the second pass: it is touched only when blocks are
/// refused.



#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/solvers/arrowhead_lupp/config.hpp>
#include <tdls/solvers/arrowhead_lupp/storage_map.hpp>
#include <tdls/solvers/tiled_lupp/solver_static.hpp>



// The unrolling requested by TDLS_UNROLL_FORCE is a performance hint: a
// failed or misplaced one does not affect correctness. clang reports a
// forced unrolling that the optimizer could not perform through
// -Wpass-failed (clang 12 does, on the multiplier pass). nvcc 12.0 places
// the _Pragma of a loop nested in the body of another TDLS_ARROW_LUPP_FOR,
// a macro argument, before the enclosing statement, and reports #20058
// ("the #pragma unroll must precede a loop"); nvcc 12.9 and 13.0 place it
// on the loop. Both suppressions are scoped to this header and to those
// warnings only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
#if defined(__NVCC__)
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20058
#endif

namespace tdls {



/* Addressing macros. They expand inside member functions where the
   parameter names (A, A_stride, y, rhs_stride, work, work_stride), the
   residency template booleans and the class constants are in scope.
   #undef'd at the end of this header. */

/// \def TDLS_ARROW_LUPP_A
/// \brief Stored entry at index idx of the matrix: contiguous under
/// internal residency, strided otherwise.
#define TDLS_ARROW_LUPP_A(idx) A[internal_matrix ? unsigned(idx) : TDLS_STRIDED(idx, A_stride)]
/// \def TDLS_ARROW_LUPP_Y
/// \brief Entry i of the right-hand side and solution: contiguous under
/// internal residency, strided otherwise.
#define TDLS_ARROW_LUPP_Y(i) y[internal_rhs ? unsigned(i) : TDLS_STRIDED(i, rhs_stride)]
/// \def TDLS_ARROW_LUPP_YW
/// \brief Entry i of column m of the right-hand sides and solutions:
/// columns of dimension contiguous entries under internal residency,
/// xcol_stride-strided columns otherwise. The function declares ycols,
/// its number of columns: with one, m is ignored and the macro is
/// TDLS_ARROW_LUPP_Y exactly.
#define TDLS_ARROW_LUPP_YW(m, i)                                                                   \
    y[internal_rhs ? unsigned((ycols == 1 ? 0 : (m)) * dimension + (i))                            \
                   : TDLS_STRIDED(i, rhs_stride) +                                                 \
                         (ycols == 1 ? std::size_t(0) : TDLS_STRIDED(m, xcol_stride))]
/// \def TDLS_ARROW_LUPP_W
/// \brief Entry e of the workspace: contiguous under internal residency,
/// strided otherwise.
#define TDLS_ARROW_LUPP_W(e) work[internal_work ? unsigned(e) : TDLS_STRIDED(e, work_stride)]
/// \def TDLS_ARROW_LUPP_F
/// \brief Entry (r, c) of the front of the second pass, row-major at the
/// start of the workspace, fcols columns per row.
#define TDLS_ARROW_LUPP_F(r, c) TDLS_ARROW_LUPP_W((r) * fcols + (c))
/// \def TDLS_ARROW_LUPP_FOR
/// \brief `for (int i = first; i < last; ++i) { body }` under the unroll
/// policy of the configuration: the two-branch `if constexpr` of the
/// other families, the forced-unroll pragma lexically present in the
/// first branch only. The body is the variadic argument.
#define TDLS_ARROW_LUPP_FOR(i, first, last, ...)                                                   \
    if constexpr (Config.unroll_loops) {                                                           \
        TDLS_UNROLL_FORCE                                                                          \
        for (int i = (first); i < (last); ++i) {                                                   \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } else {                                                                                       \
        for (int i = (first); i < (last); ++i) {                                                   \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    }



/// \brief Arrowhead solver with a compile-time shape.
/// \tparam T      scalar type (float, double or long double)
/// \tparam Shape  shape of the matrix (valid for the storage and the
///                coupling of Config)
/// \tparam Config configuration (see ArrowheadLUppConfig)
template<typename T, ArrowheadShape Shape, ArrowheadLUppConfig<T> Config>
struct ArrowheadLUppSolverStatic {
    /// Index of every entry of the matrix in its storage.
    using storage_map = ArrowheadStorageMap<Config.storage, Config.coupling, Config.layout>;
    /// Shape of the matrix.
    static constexpr ArrowheadShape shape = Shape;
    /// Dimension of the matrix, border + block * blocks.
    static constexpr int dimension = Shape.n();
    /// Number of stored entries of the matrix.
    static constexpr int storage_size = storage_map::size(Shape);

  private:
    static constexpr int B     = Shape.border;
    static constexpr int S     = Shape.block;
    static constexpr int NB    = Shape.blocks;
    static constexpr bool col  = Config.layout == MatrixLayout::ColMajor;
    static constexpr bool diag = Config.coupling == ArrowheadCoupling::Diagonal;
    /// Border rows the blocks update: block rows from coupling_row on
    /// under the diagonal coupling, every row otherwise. Only these rows of
    /// the Schur complement are accumulated during the sweep over the
    /// blocks; w and T of the second pass have R entries and B x R.
    static constexpr int R  = diag ? S : B;
    static constexpr int R0 = diag ? Shape.coupling_row : 0;

    /// \brief The shape as a value, for the storage map, built from the
    /// constants above (coupling_row is R0: 0 under the dense coupling).
    /// The map takes the shape by reference, and nvcc 12 rejects a
    /// reference to the template parameter object Shape in device code.
    /// \return the shape
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr ArrowheadShape
    shape_value() noexcept {
        return {B, S, NB, R0};
    }
    /// Rows of the front of a refused block: its S rows over the B border
    /// rows.
    static constexpr int front_rows = S + B;
    /// Columns of the front with nc right-hand sides: x_k (S), xb (B), the
    /// carried couplings T (R), the nc right-hand sides.
    template<int nc>
    static constexpr int front_cols_for = S + B + R + nc;
    /// Columns of H_k beyond the S that fit in place of D_k (dense
    /// coupling with B > S), per block, stored after the front.
    static constexpr int extra_cols = R > S ? R - S : 0;

  public:
    /// Entries of the workspace of the entry points with nrhs right-hand
    /// sides, used only when blocks are refused: the front grows by one
    /// column per right-hand side.
    template<int nrhs>
    static constexpr int multirhs_workspace_size =
        front_rows * front_cols_for<nrhs> + NB * S * extra_cols;
    /// Entries of the workspace of solve_inplace and solve, used only when
    /// blocks are refused.
    static constexpr int workspace_size = multirhs_workspace_size<1>;
    /// Columns [A_kb | y_k] solved per pass, one right-hand side.
    static constexpr int pass_width = Config.rhs_pass_width == 0 ? B + 1 : Config.rhs_pass_width;

  private:
    /// Columns [A_kb | Y_k] solved per pass with ny right-hand sides in
    /// the blocks (pass_width for ny = 1).
    template<int ny>
    static constexpr int pass_width_for =
        Config.rhs_pass_width == 0 || Config.rhs_pass_width > B + ny ? B + ny
                                                                     : Config.rhs_pass_width;
    /// Column-major and a single pass: [A_kb | Y_k] is loaded once, and the
    /// pivot check reads its border part from the forward stage.
    template<int ny>
    static constexpr bool forward_check_for = col && pass_width_for<ny> == B + ny;
    /// Row-major with the dense coupling: the Schur complement is updated
    /// by M V, M = Cb U^-1 from the pivot check and V = L^-1 P [A_kb | y_k]
    /// the forward stage of the solve; the backward stage, on one column,
    /// moves to the recovery. Under the diagonal coupling M V costs more
    /// than the sparse A_bk W, and storing U costs traffic: the whole solve
    /// is kept.
    static constexpr bool multiplier_update = !col && !diag;
    /// Schur complement accumulated in the border entries of A and y
    /// (ArrowheadLUppConfig::schur_in_place).
    static constexpr bool in_place = Config.schur_in_place;

    static_assert(Shape.border >= 1 && Shape.block >= 1 && Shape.blocks >= 1,
                  "ArrowheadLUppSolverStatic: border, block and blocks must be >= 1");
    static_assert(diag || Shape.coupling_row == 0,
                  "ArrowheadLUppSolverStatic: coupling_row applies to the diagonal coupling only");
    static_assert(!diag || (Shape.coupling_row >= 0 && Shape.coupling_row + S <= B),
                  "ArrowheadLUppSolverStatic: the diagonal coupling rows exceed the border");
    static_assert(Config.storage != ArrowheadStorage::Bsr ||
                      (B % S == 0 && (!diag || Shape.coupling_row % S == 0)),
                  "ArrowheadLUppSolverStatic: the Bsr storage needs a border (and a coupling_row) "
                  "multiple of the block");
    static_assert(storage_map::wide_size(Shape) < (1LL << 31),
                  "ArrowheadLUppSolverStatic: the storage exceeds 2^31 entries");
    static_assert(Config.condensation_pivot_threshold.is_finite() &&
                      T(Config.condensation_pivot_threshold) >= T(0) &&
                      T(Config.condensation_pivot_threshold) <= T(1),
                  "ArrowheadLUppSolverStatic: condensation_pivot_threshold must be in [0, 1]");

    /// \brief Threshold of the pivot check and of the second pass, read
    /// once from the configuration at compile time (see
    /// ArrowheadLUppConfig::condensation_pivot_threshold).
    static constexpr T condensation_pivot_threshold = Config.condensation_pivot_threshold;
    static_assert(Config.rhs_pass_width >= 0 && Config.rhs_pass_width <= Shape.border + 1,
                  "ArrowheadLUppSolverStatic: rhs_pass_width must be in [0, border + 1]");

    using BlockSolver  = TiledLUppSolverStatic<T, S, detail::arrowhead_block_config(Config)>;
    using BorderSolver = TiledLUppSolverStatic<T, B, detail::arrowhead_border_config(Config)>;

  public:
    /// \brief Solves A x = y in place, counting the refused blocks.
    /// \tparam internal_rhs    y is a caller-local array (stride ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A             matrix in its storage (pattern entries
    ///                              only); recovery data on exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of workspace_size entries,
    ///                              touched only when blocks are refused
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride    element stride of y
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check, eliminated by the second pass
    /// \return false when the matrix is singular
    template<bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                  const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                  int& refused_count) noexcept {
        return solve_core<1, false, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, y, rhs_stride, 0, 0, refused_count);
    }

    /// \brief Solves A x = y in place.
    /// \tparam internal_rhs    y is a caller-local array (stride ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    work        workspace of workspace_size entries
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \return false when the matrix is singular
    template<bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                  const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride) noexcept {
        int refused_count = 0;
        return solve_inplace<internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, y, rhs_stride, refused_count);
    }

    /// \brief Solves A x = b, b left untouched, counting the refused
    /// blocks: b is copied into x, then solved in place.
    /// \tparam internal_rhs    b and x are caller-local arrays (stride
    ///                         ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A             matrix in its storage; recovery data on
    ///                              exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of workspace_size entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     b             right-hand side
    /// \param[out]    x             solution
    /// \param[in]     rhs_stride    element stride of b and x
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check
    /// \return false when the matrix is singular
    template<bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work, const int work_stride,
          const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          int& refused_count) noexcept {
        for (int i = 0; i < dimension; ++i) {
            if constexpr (internal_rhs)
                x[i] = b[i];
            else
                x[TDLS_STRIDED(i, rhs_stride)] = b[TDLS_STRIDED(i, rhs_stride)];
        }
        return solve_inplace<internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, x, rhs_stride, refused_count);
    }

    /// \brief Solves A x = b, b left untouched.
    /// \tparam internal_rhs    b and x are caller-local arrays (stride
    ///                         ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    work        workspace of workspace_size entries
    /// \param[in]     work_stride element stride of work
    /// \param[in]     b           right-hand side
    /// \param[out]    x           solution
    /// \param[in]     rhs_stride  element stride of b and x
    /// \return false when the matrix is singular
    template<bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work, const int work_stride,
          const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride) noexcept {
        int refused_count = 0;
        return solve<internal_rhs, internal_work, internal_matrix>(A, A_stride, work, work_stride,
                                                                   b, x, rhs_stride, refused_count);
    }

    /// \brief Solves A X = Y in place for nrhs right-hand sides, counting
    /// the refused blocks. The right-hand sides ride along the
    /// condensation, which runs once: each block solves [A_kb | Y_k] in
    /// passes of rhs_pass_width columns, the border system nrhs columns.
    /// Column m comes out bitwise identical to solve_inplace on it alone.
    /// \tparam nrhs            number of right-hand sides, >= 1
    /// \tparam internal_rhs    y is a caller-local array, column m from
    ///                         m * dimension (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A             matrix in its storage (pattern entries
    ///                              only); recovery data on exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of multirhs_workspace_size<nrhs>
    ///                              entries, touched only when blocks are
    ///                              refused
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             nrhs right-hand-side columns on entry,
    ///                              the solutions on exit
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     xcol_stride   element stride between columns of y
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check, eliminated by the second pass
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace_multirhs(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                           const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                           const int xcol_stride, int& refused_count) noexcept {
        static_assert(nrhs >= 1, "ArrowheadLUppSolverStatic: nrhs must be >= 1");
        return solve_core<nrhs, false, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, 0, refused_count);
    }

    /// \brief Solves A X = Y in place for nrhs right-hand sides.
    /// \tparam nrhs            number of right-hand sides, >= 1
    /// \tparam internal_rhs    y is a caller-local array, column m from
    ///                         m * dimension (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    work        workspace of multirhs_workspace_size<nrhs>
    ///                            entries
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           nrhs right-hand-side columns on entry, the
    ///                            solutions on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace_multirhs(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                           const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                           const int xcol_stride) noexcept {
        int refused_count = 0;
        return solve_inplace_multirhs<nrhs, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Solves A X = B for nrhs right-hand sides, B left untouched,
    /// counting the refused blocks: B is copied into X, then solved in
    /// place.
    /// \tparam nrhs            number of right-hand sides, >= 1
    /// \tparam internal_rhs    b and x are caller-local arrays, column m
    ///                         from m * dimension (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A             matrix in its storage; recovery data on
    ///                              exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of multirhs_workspace_size<nrhs>
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     b             nrhs right-hand-side columns
    /// \param[out]    x             the nrhs solutions
    /// \param[in]     rhs_stride    element stride of b and x
    /// \param[in]     xcol_stride   element stride between columns of b and x
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_multirhs(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                   const int work_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x,
                   const int rhs_stride, const int xcol_stride, int& refused_count) noexcept {
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < dimension; ++i) {
                if constexpr (internal_rhs)
                    x[m * dimension + i] = b[m * dimension + i];
                else
                    x[TDLS_STRIDED(i, rhs_stride) + TDLS_STRIDED(m, xcol_stride)] =
                        b[TDLS_STRIDED(i, rhs_stride) + TDLS_STRIDED(m, xcol_stride)];
            }
        return solve_inplace_multirhs<nrhs, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, x, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Solves A X = B for nrhs right-hand sides, B left untouched.
    /// \tparam nrhs            number of right-hand sides, >= 1
    /// \tparam internal_rhs    b and x are caller-local arrays, column m
    ///                         from m * dimension (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    work        workspace of multirhs_workspace_size<nrhs>
    ///                            entries
    /// \param[in]     work_stride element stride of work
    /// \param[in]     b           nrhs right-hand-side columns
    /// \param[out]    x           the nrhs solutions
    /// \param[in]     rhs_stride  element stride of b and x
    /// \param[in]     xcol_stride element stride between columns of b and x
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_multirhs(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                   const int work_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x,
                   const int rhs_stride, const int xcol_stride) noexcept {
        int refused_count = 0;
        return solve_multirhs<nrhs, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, b, x, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Columns col0 .. col0 + nrhs - 1 of the border block of A^-1,
    /// the inverse of the Schur complement of the border (col0 + nrhs <=
    /// border), counting the refused blocks: rows 0 .. border - 1 of the
    /// solutions of A X = [e_col0 .. e_col0+nrhs-1]. These right-hand sides
    /// vanish on the rows of the blocks: the blocks are condensed without
    /// right-hand side, and nothing is recovered. The cost is the
    /// condensation of the matrix and nrhs substitutions of the border
    /// system: the tangent operator of a behaviour whose loading enters the
    /// border equations only. Each column equals the border rows of
    /// solve_inplace on the unit column; a zero may differ in sign, the
    /// blocks contributing exact zeros where solve_inplace subtracts zero
    /// products.
    /// \tparam nrhs            number of unit columns, in [1, border]
    /// \tparam internal_rhs    x is a caller-local array, column m from
    ///                         m * border (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A             matrix in its storage; overwritten
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of multirhs_workspace_size<nrhs>
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     col0          first unit column, col0 + nrhs <= border
    /// \param[out]    x             the border rows, border entries per
    ///                              column
    /// \param[in]     rhs_stride    element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_border_canonical_multirhs(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                                    const int work_stride, const int col0, T* TDLS_RESTRICT x,
                                    const int rhs_stride, const int xcol_stride,
                                    int& refused_count) noexcept {
        static_assert(nrhs >= 1 && nrhs <= Shape.border,
                      "ArrowheadLUppSolverStatic: nrhs must be in [1, border]");
        return solve_core<nrhs, true, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, x, rhs_stride, xcol_stride, col0, refused_count);
    }

    /// \brief Columns col0 .. col0 + nrhs - 1 of the border block of A^-1
    /// (col0 + nrhs <= border).
    /// \tparam nrhs            number of unit columns, in [1, border]
    /// \tparam internal_rhs    x is a caller-local array, column m from
    ///                         m * border (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A           matrix in its storage; overwritten
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    work        workspace of multirhs_workspace_size<nrhs>
    ///                            entries
    /// \param[in]     work_stride element stride of work
    /// \param[in]     col0        first unit column, col0 + nrhs <= border
    /// \param[out]    x           the border rows, border entries per column
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_border_canonical_multirhs(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                                    const int work_stride, const int col0, T* TDLS_RESTRICT x,
                                    const int rhs_stride, const int xcol_stride) noexcept {
        int refused_count = 0;
        return solve_border_canonical_multirhs<nrhs, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, col0, x, rhs_stride, xcol_stride, refused_count);
    }

  private:
    /// \brief The solve shared by every entry point: nc right-hand sides
    /// solved during the condensation (columns of Y), or, canonical, the
    /// border rows of nc unit columns of the border (no right-hand side in
    /// the blocks, no recovery).
    /// \tparam nc              number of right-hand sides
    /// \tparam canonical       unit columns col0 .. col0 + nc - 1, border
    ///                         rows written to y (B rows per column)
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_work   residency of work
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A             matrix in its storage
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of multirhs_workspace_size<nc>
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             right-hand sides on entry, solutions on
    ///                              exit (canonical: the border rows, out)
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     xcol_stride   element stride between columns of y
    /// \param[in]     col0          first unit column (canonical)
    /// \param[out]    refused_count number of refused blocks
    /// \return false when the matrix is singular
    template<int nc, bool canonical, bool internal_rhs, bool internal_work, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL static constexpr bool
    solve_core(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work, const int work_stride,
               T* TDLS_RESTRICT y, const int rhs_stride, [[maybe_unused]] const int xcol_stride,
               [[maybe_unused]] const int col0, int& refused_count) noexcept {
        // Right-hand sides carried through the blocks.
        constexpr int ny = canonical ? 0 : nc;
        // Columns of y, for TDLS_ARROW_LUPP_YW.
        [[maybe_unused]] constexpr int ycols = nc;
        // Rows R0 .. R0 + R of the Schur complement and of its right-hand
        // sides (column m at m * R), unless accumulated in place.
        T Sacc[in_place ? 1 : R * B];
        T racc[in_place || ny == 0 ? 1 : R * ny];
        if constexpr (in_place) {
            // Placeholders, never read: initialized for the flow analysis
            // of the compilers.
            Sacc[0] = T(0);
            racc[0] = T(0);
        } else {
            if constexpr (ny == 0) racc[0] = T(0);
            TDLS_ARROW_LUPP_FOR(
                r, 0, R,
                TDLS_ARROW_LUPP_FOR(c, 0, B,
                                    Sacc[r * B + c] = TDLS_ARROW_LUPP_A(
                                        storage_map::abb(shape_value(), R0 + r, c));)
                    TDLS_ARROW_LUPP_FOR(m, 0, ny, racc[m * R + r] = TDLS_ARROW_LUPP_YW(m, R0 + r);))
        }
        constexpr int words = (NB + 31) / 32;
        unsigned refused[words];
        TDLS_ARROW_LUPP_FOR(e, 0, words, refused[e] = 0u;)
        refused_count = 0;
        for (int k = 0; k < NB; ++k) {
            if (!condense<ny, internal_rhs, internal_matrix>(A, A_stride, y, rhs_stride,
                                                             xcol_stride, k, Sacc, racc)) {
                refused[k >> 5] |= 1u << (k & 31);
                ++refused_count;
            }
        }
        // The border system: the accumulated rows, the others as stored;
        // column m of its right-hand sides at m * B.
        T Sbb[B * B];
        T rb[B * nc];
        TDLS_ARROW_LUPP_FOR(
            r, 0, B, const bool acc = !in_place && r >= R0 && r < R0 + R; TDLS_ARROW_LUPP_FOR(
                c, 0, B,
                Sbb[r * B + c] = acc ? Sacc[(r - R0) * B + c]
                                     : TDLS_ARROW_LUPP_A(storage_map::abb(shape_value(), r, c));)
                TDLS_ARROW_LUPP_FOR(
                    m, 0, nc, if constexpr (canonical) rb[m * B + r] = r == col0 + m ? T(1) : T(0);
                    else rb[m * B + r] = acc ? racc[m * R + r - R0] : TDLS_ARROW_LUPP_YW(m, r);))
        if (refused_count > 0 &&
            !second_pass<nc, canonical, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, refused, Sbb, rb))
            return false;
        int pb[B];
        if constexpr (nc == 1) {
            if (!BorderSolver::template solve_inplace<true, true, true>(Sbb, 1, pb, 1, rb, 1))
                return false;
        } else {
            if (!BorderSolver::template solve_inplace_multirhs<nc, true, true, true>(Sbb, 1, pb, 1,
                                                                                     rb, 1, B))
                return false;
        }
        if constexpr (canonical) {
            TDLS_ARROW_LUPP_FOR(
                m, 0, nc,
                TDLS_ARROW_LUPP_FOR(
                    r, 0, B,
                    y[internal_rhs ? unsigned(m * B + r)
                                   : TDLS_STRIDED(r, rhs_stride) + TDLS_STRIDED(m, xcol_stride)] =
                        rb[m * B + r];))
        } else {
            TDLS_ARROW_LUPP_FOR(
                m, 0, nc, TDLS_ARROW_LUPP_FOR(r, 0, B, TDLS_ARROW_LUPP_YW(m, r) = rb[m * B + r];))
            recover<nc, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, refused, rb);
        }
        return true;
    }

    /// \brief Condenses block k into the accumulated rows of the border
    /// system; false when the block fails the pivot check, then left
    /// untouched.
    ///
    /// The pivot check runs on the factors P F = L U of TiledLUpp (F = D_k,
    /// or D_k^T under the column-major layout). The candidates of a pivot
    /// are the rows of F not yet eliminated and the border rows that share
    /// its column. Each one holds, once eliminated, its multiplier times
    /// the pivot: L for the rows of F, M = Cb U^-1 for the border rows, Cb
    /// being their entries in the columns of F (A_bk under the row-major
    /// layout, its coupled rows under the diagonal coupling; A_kb^T under
    /// the column-major one). The block passes when no multiplier exceeds
    /// 1 / condensation_pivot_threshold: the candidates are checked as
    /// eliminated, not as stored.
    /// \tparam ny              right-hand sides carried through the blocks
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in,out] Sacc        accumulated rows of the Schur complement
    /// \param[in,out] racc        accumulated rows of its right-hand sides
    /// \return true when the block is condensed
    template<int ny, bool internal_rhs, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    condense(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y, const int rhs_stride,
             [[maybe_unused]] const int xcol_stride, const int k, T* TDLS_RESTRICT Sacc,
             T* TDLS_RESTRICT racc) noexcept {
        constexpr bool forward_check         = forward_check_for<ny>;
        [[maybe_unused]] constexpr int ycols = ny;
        // The block is copied in the order of its storage into a local
        // array, and TiledLUpp reads it in that layout. Measured on the RTX
        // 5080, a copy in the workspace instead (global memory, SoA) costs
        // 3 to 56 % more time although it shrinks the stack: the repeated
        // accesses of the factorization stay in the L1 cache here.
        T Dl[S * S];
        TDLS_ARROW_LUPP_FOR(i, 0, S,
                            TDLS_ARROW_LUPP_FOR(j, 0, S,
                                                Dl[col ? j * S + i : i * S + j] = TDLS_ARROW_LUPP_A(
                                                    storage_map::d(shape_value(), k, i, j));))
        int pl[S];
        if (!BlockSolver::template factorize<true, true>(Dl, 1, pl, 1)) return false;
        // The couplings of the block under the diagonal coupling; a
        // placeholder, never read, otherwise (initialized for the flow
        // analysis of the compilers, as the other placeholders below).
        T acoef[diag ? S : 1];
        if constexpr (diag) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S, acoef[i] = TDLS_ARROW_LUPP_A(storage_map::abk_diag(shape_value(), k, i));)
        } else {
            acoef[0] = T(0);
        }
        // U in logical row order, unless the forward stage of the solve
        // gives the border multipliers.
        T U[forward_check ? 1 : S * S];
        if constexpr (forward_check) U[0] = T(0);
        bool ok = lower_pass<!forward_check>(Dl, pl, U);
        if constexpr (forward_check) {
            T W[(B + ny) * S];
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(
                    c, 0, B,
                    W[c * S + i] = TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c));)
                    TDLS_ARROW_LUPP_FOR(m, 0, ny,
                                        W[(B + m) * S + i] = TDLS_ARROW_LUPP_YW(m, B + k * S + i);))
            BlockSolver::template forward_inplace_multirhs<B + ny, true, true, true>(Dl, 1, pl, 1,
                                                                                     W, 1, S);
            // Column c of U^-T A_kb is row c of the border multipliers.
            const T u = condensation_pivot_threshold;
            TDLS_ARROW_LUPP_FOR(e, 0, B * S, ok = ok && u * detail::abs(W[e]) <= T(1);)
            if (!ok) return false;
            BlockSolver::template backward_inplace_multirhs<B + ny, true, true, true>(Dl, 1, pl, 1,
                                                                                      W, 1, S);
            condense_pass<0, true, ny, internal_rhs, internal_matrix>(
                A, A_stride, y, rhs_stride, xcol_stride, k, Dl, pl, acoef, nullptr, Sacc, racc, W);
        } else {
            T Mb[multiplier_update ? R * S : 1];
            if constexpr (!multiplier_update) Mb[0] = T(0);
            ok = border_multipliers<internal_matrix>(A, A_stride, k, U, acoef, Mb) && ok;
            if (!ok) return false;
            if constexpr (multiplier_update) {
                TDLS_ARROW_LUPP_FOR(
                    l, 0, S,
                    TDLS_ARROW_LUPP_FOR(j, l, S,
                                        TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, l, j)) =
                                            U[l * S + j];))
            }
            condense_pass<0, false, ny, internal_rhs, internal_matrix>(
                A, A_stride, y, rhs_stride, xcol_stride, k, Dl, pl, acoef, Mb, Sacc, racc, nullptr);
        }
        return true;
    }

    /// \brief Checks the multipliers L of the factors against
    /// 1 / condensation_pivot_threshold, and copies U in logical row order
    /// when keep_u (its diagonal holds 1 / pivot).
    /// \tparam keep_u whether U is copied
    /// \param[in]  Dl factored block, as TiledLUpp leaves it
    /// \param[in]  pl its pivots
    /// \param[out] U  S x S, upper triangle filled when keep_u
    /// \return true when no multiplier of L exceeds the bound
    template<bool keep_u>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    lower_pass(const T* TDLS_RESTRICT Dl, const int* TDLS_RESTRICT pl,
               T* TDLS_RESTRICT U) noexcept {
        const T u = condensation_pivot_threshold;
        bool ok   = true;
        TDLS_ARROW_LUPP_FOR(l, 0, S, const int p = pl[l]; TDLS_ARROW_LUPP_FOR(
                                j, 0, S,
                                if (j < l) {
                                    ok = ok && u * detail::abs(Dl[p * S + j]) <= T(1);
                                } else if constexpr (keep_u) { U[l * S + j] = Dl[p * S + j]; }))
        return ok;
    }

    /// \brief Border multipliers M = Cb U^-1 of block k, checked against
    /// 1 / condensation_pivot_threshold; kept in Mb (R x S) under the
    /// row-major layout with the dense coupling, where the Schur update
    /// uses them (a 36-entry array kept for nothing costs 10 % on CPU
    /// under clang).
    /// \tparam internal_matrix residency of A
    /// \param[in]  A        matrix in its storage
    /// \param[in]  A_stride element stride of A
    /// \param[in]  k        block
    /// \param[in]  U        U of the block in logical row order
    /// \param[in]  acoef    couplings of the block (diagonal coupling)
    /// \param[out] Mb       the multipliers (row-major, dense coupling)
    /// \return true when no multiplier exceeds the bound
    template<bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    border_multipliers(const T* TDLS_RESTRICT A, const int A_stride, const int k,
                       const T* TDLS_RESTRICT U, const T* TDLS_RESTRICT acoef,
                       T* TDLS_RESTRICT Mb) noexcept {
        const T u = condensation_pivot_threshold;
        bool ok   = true;
        if constexpr (col) {
            TDLS_ARROW_LUPP_FOR(
                c, 0, B, T m[S];
                TDLS_ARROW_LUPP_FOR(
                    i, 0, S, m[i] = TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c));)
                    ok = multipliers_pass(m, 0, U, u) && ok;)
        } else if constexpr (diag) {
            // Border row coupling_row + i holds one entry, in column i.
            TDLS_ARROW_LUPP_FOR(i, 0, S, T m[S];
                                TDLS_ARROW_LUPP_FOR(j, 0, S, m[j] = j == i ? acoef[i] : T(0);) ok =
                                    multipliers_pass(m, i, U, u) && ok;)
        } else {
            TDLS_ARROW_LUPP_FOR(r, 0, B,
                                TDLS_ARROW_LUPP_FOR(j, 0, S,
                                                    Mb[r * S + j] = TDLS_ARROW_LUPP_A(
                                                        storage_map::abk(shape_value(), k, r, j));)
                                    ok = multipliers_pass(Mb + r * S, 0, U, u) && ok;)
        }
        return ok;
    }

    /// \brief Turns one border row c into its multipliers m = c U^-1, in
    /// place, from column `first` on (c is zero before it).
    /// \param[in,out] m     the row on entry, its multipliers on exit
    /// \param[in]     first first nonzero column of the row
    /// \param[in]     U     U of the block in logical row order
    /// \param[in]     u     condensation_pivot_threshold
    /// \return true when no multiplier exceeds 1 / u in magnitude
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    multipliers_pass(T* TDLS_RESTRICT m, const int first, const T* TDLS_RESTRICT U,
                     const T u) noexcept {
        bool ok = true;
        TDLS_ARROW_LUPP_FOR(l, first, S, m[l] *= U[l * S + l];
                            ok = ok && u * detail::abs(m[l]) <= T(1);
                            TDLS_ARROW_LUPP_FOR(j, l + 1, S, m[j] -= m[l] * U[l * S + j];))
        return ok;
    }

    /// \brief Subtracts v from entry (R0 + r, c) of the Schur complement
    /// (c = B: from its right-hand side), accumulated in place or in Sacc
    /// and racc.
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in,out] Sacc        accumulated rows of the Schur complement
    /// \param[in,out] racc        accumulated rows of its right-hand sides
    /// \param[in]     r           accumulated row
    /// \param[in]     c           border column, or B + m for right-hand side m
    /// \param[in]     v           value subtracted
    template<int ny, bool internal_rhs, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    schur_sub(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y, const int rhs_stride,
              [[maybe_unused]] const int xcol_stride, T* TDLS_RESTRICT Sacc, T* TDLS_RESTRICT racc,
              const int r, const int c, const T v) noexcept {
        [[maybe_unused]] constexpr int ycols = ny;
        if constexpr (in_place) {
            if (c < B)
                TDLS_ARROW_LUPP_A(storage_map::abb(shape_value(), R0 + r, c)) -= v;
            else
                TDLS_ARROW_LUPP_YW(c - B, R0 + r) -= v;
        } else {
            if (c < B)
                Sacc[r * B + c] -= v;
            else
                racc[(ny == 1 ? 0 : c - B) * R + r] -= v;
        }
    }

    /// \brief Columns c0 .. c0 + w of [A_kb | Y_k] (column B + m is column m
    /// of the right-hand sides): solved (V_k under the row-major layout with
    /// the dense coupling, W_k otherwise), stored in place, folded into the
    /// accumulated rows. The columns go in passes of pass_width_for<ny> so
    /// that only one slice is alive at a time. When solved, Wsolved already
    /// holds all of W_k (one pass).
    /// \tparam c0              first column of the pass
    /// \tparam solved          whether Wsolved holds the solved columns
    /// \tparam ny              right-hand sides carried through the blocks
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in]     Dl          factored block
    /// \param[in]     pl          its pivots
    /// \param[in]     acoef       couplings of the block (diagonal coupling)
    /// \param[in]     Mb          border multipliers (row-major, dense
    ///                            coupling)
    /// \param[in,out] Sacc        accumulated rows of the Schur complement
    /// \param[in,out] racc        accumulated rows of its right-hand sides
    /// \param[in]     Wsolved     the solved columns when solved
    template<int c0, bool solved, int ny, bool internal_rhs, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    condense_pass(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                  [[maybe_unused]] const int xcol_stride, const int k, const T* TDLS_RESTRICT Dl,
                  const int* TDLS_RESTRICT pl, const T* TDLS_RESTRICT acoef,
                  const T* TDLS_RESTRICT Mb, T* TDLS_RESTRICT Sacc, T* TDLS_RESTRICT racc,
                  T* TDLS_RESTRICT Wsolved) noexcept {
        constexpr int cols = B + ny;
        constexpr int w    = cols - c0 < pass_width_for<ny> ? cols - c0 : pass_width_for<ny>;
        [[maybe_unused]] constexpr int ycols = ny;
        T Wl[solved ? 1 : w * S];
        T* W = solved ? Wsolved : Wl;
        if constexpr (!solved) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(
                    j, 0, w,
                    W[j * S + i] = c0 + j < B ? TDLS_ARROW_LUPP_A(
                                                    storage_map::akb(shape_value(), k, i, c0 + j))
                                              : TDLS_ARROW_LUPP_YW(c0 + j - B, B + k * S + i);))
            if constexpr (multiplier_update) {
                BlockSolver::template forward_inplace_multirhs<w, true, true, true>(Dl, 1, pl, 1, W,
                                                                                    1, S);
            } else if constexpr (Config.gather_rhs) {
                T X[w * S];
                BlockSolver::template substitute_multirhs<w, true, true, true>(Dl, 1, pl, 1, W, X,
                                                                               1, S);
                TDLS_ARROW_LUPP_FOR(e, 0, w * S, W[e] = X[e];)
            } else {
                BlockSolver::template substitute_inplace_multirhs<w, true, true, true>(Dl, 1, pl, 1,
                                                                                       W, 1, S);
            }
        }
        TDLS_ARROW_LUPP_FOR(
            i, 0, S,
            TDLS_ARROW_LUPP_FOR(j, 0, w,
                                if (c0 + j < B) TDLS_ARROW_LUPP_A(
                                    storage_map::akb(shape_value(), k, i, c0 + j)) = W[j * S + i];
                                else TDLS_ARROW_LUPP_YW(c0 + j - B, B + k * S + i) = W[j * S + i];))
        if constexpr (multiplier_update) {
            TDLS_ARROW_LUPP_FOR(
                r, 0, R,
                TDLS_ARROW_LUPP_FOR(
                    j, 0, w, T sum = T(0);
                    TDLS_ARROW_LUPP_FOR(l, 0, S, sum += Mb[r * S + l] * W[j * S + l];)
                        schur_sub<ny, internal_rhs, internal_matrix>(
                            A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc, r, c0 + j, sum);))
        } else if constexpr (diag) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(j, 0, w,
                                    schur_sub<ny, internal_rhs, internal_matrix>(
                                        A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc, i,
                                        c0 + j, acoef[i] * W[j * S + i]);))
        } else {
            TDLS_ARROW_LUPP_FOR(
                r, 0, B, T a[S]; TDLS_ARROW_LUPP_FOR(
                    i, 0, S, a[i] = TDLS_ARROW_LUPP_A(storage_map::abk(shape_value(), k, r, i));)
                    TDLS_ARROW_LUPP_FOR(j, 0, w, T sum = T(0);
                                        TDLS_ARROW_LUPP_FOR(i, 0, S, sum += a[i] * W[j * S + i];)
                                            schur_sub<ny, internal_rhs, internal_matrix>(
                                                A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc,
                                                r, c0 + j, sum);))
        }
        if constexpr (c0 + w < cols)
            condense_pass<c0 + w, false, ny, internal_rhs, internal_matrix>(
                A, A_stride, y, rhs_stride, xcol_stride, k, Dl, pl, acoef, Mb, Sacc, racc, nullptr);
    }

    /// \brief Recovers the blocks from the border solution, the last block
    /// first: x_k from the recovery data of a condensed block, or from
    /// x_k = c_k - G_k xb - H_k w for a refused one, w gathering the
    /// couplings A_bj x_j (its R coupled rows) of the refused blocks
    /// recovered so far. Each stored coefficient is read once per block and
    /// applied to the nc columns.
    /// \tparam nc              number of right-hand sides
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_work   residency of work
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           matrix holding the recovery data
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     work        workspace, H_k beyond its S columns
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           the recovery right-hand sides on entry,
    ///                            the solutions on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     refused     bit k set for a refused block k
    /// \param[in]     rb          border solutions, column m at m * B
    template<int nc, bool internal_rhs, bool internal_work, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    recover(const T* TDLS_RESTRICT A, const int A_stride, const T* TDLS_RESTRICT work,
            const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
            [[maybe_unused]] const int xcol_stride, const unsigned* TDLS_RESTRICT refused,
            const T* TDLS_RESTRICT rb) noexcept {
        constexpr int fcols                  = front_cols_for<nc>;
        [[maybe_unused]] constexpr int ycols = nc;
        // w and s: column m at m * R and m * S.
        T w[R * nc];
        TDLS_ARROW_LUPP_FOR(e, 0, R * nc, w[e] = T(0);)
        for (int k = NB - 1; k >= 0; --k) {
            T s[S * nc];
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(m, 0, nc, s[m * S + i] = TDLS_ARROW_LUPP_YW(m, B + k * S + i);)
                    TDLS_ARROW_LUPP_FOR(
                        c, 0, B,
                        const T a = TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c));
                        TDLS_ARROW_LUPP_FOR(m, 0, nc, s[m * S + i] -= a * rb[m * B + c];)))
            if ((refused[k >> 5] >> (k & 31)) & 1u) {
                // x_k = c_k - G_k xb - H_k w, then w += A_bk x_k.
                TDLS_ARROW_LUPP_FOR(
                    i, 0, S,
                    TDLS_ARROW_LUPP_FOR(
                        a, 0, R,
                        const T h = a < S
                                        ? TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, i, a))
                                        : TDLS_ARROW_LUPP_W(front_rows * fcols +
                                                            (k * S + i) * extra_cols + a - S);
                        TDLS_ARROW_LUPP_FOR(m, 0, nc, s[m * S + i] -= h * w[m * R + a];)))
                if constexpr (diag) {
                    TDLS_ARROW_LUPP_FOR(
                        i, 0, S,
                        const T coef =
                            TDLS_ARROW_LUPP_A(storage_map::abk_diag(shape_value(), k, i));
                        TDLS_ARROW_LUPP_FOR(m, 0, nc, w[m * R + i] += coef * s[m * S + i];))
                } else {
                    TDLS_ARROW_LUPP_FOR(
                        r, 0, B,
                        TDLS_ARROW_LUPP_FOR(
                            i, 0, S,
                            const T coef =
                                TDLS_ARROW_LUPP_A(storage_map::abk(shape_value(), k, r, i));
                            TDLS_ARROW_LUPP_FOR(m, 0, nc, w[m * R + r] += coef * s[m * S + i];)))
                }
            } else if constexpr (multiplier_update) {
                // x_k = U^-1 s, U in the upper triangle of D_k, 1 / pivot on
                // its diagonal. Columns j = l + 1 + u, u < t: gcc -O2 takes
                // the empty range j in [l + 1, S) of S = 1 for a nonempty
                // one and warns of an out-of-bounds s.
                TDLS_ARROW_LUPP_FOR(
                    t, 0, S, const int l = S - 1 - t;
                    TDLS_ARROW_LUPP_FOR(
                        u, 0, t, const int j = l + 1 + u;
                        const T ulj = TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, l, j));
                        TDLS_ARROW_LUPP_FOR(m, 0, nc, s[m * S + l] -= ulj * s[m * S + j];))
                        const T ull = TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, l, l));
                    TDLS_ARROW_LUPP_FOR(m, 0, nc, s[m * S + l] *= ull;))
            }
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(m, 0, nc, TDLS_ARROW_LUPP_YW(m, B + k * S + i) = s[m * S + i];))
        }
    }

    /// \brief Second pass: eliminates every refused block together with the
    /// border rows, in the front held by the workspace (see the file
    /// documentation), the nc right-hand sides in its last columns. Out of
    /// line: its registers stay apart from those of the first pass.
    /// \tparam nc              number of right-hand sides
    /// \tparam canonical       right-hand sides zero on the rows of the
    ///                         blocks, and no recovery: the recovery data
    ///                         are neither computed nor stored
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_work   residency of work
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A           matrix in its storage; the refused blocks
    ///                            receive their recovery data
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    work        workspace: the front, then H_k beyond its
    ///                            S columns
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     refused     bit k set for a refused block k
    /// \param[in,out] Sbb         the border matrix, the final one on exit
    /// \param[in,out] rb          its right-hand sides (column m at m * B),
    ///                            the final ones on exit
    /// \return false when a column of a refused block has no nonzero
    ///         candidate: the matrix is singular
    template<int nc, bool canonical, bool internal_rhs, bool internal_work, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_NO_INLINE static constexpr bool
    second_pass(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                [[maybe_unused]] const int xcol_stride, const unsigned* TDLS_RESTRICT refused,
                T* TDLS_RESTRICT Sbb, T* TDLS_RESTRICT rb) noexcept {
        constexpr int fcols                  = front_cols_for<nc>;
        [[maybe_unused]] constexpr int ycols = nc;
        // First column of the right-hand sides in the front.
        constexpr int rhs0 = S + B + R;
        const T u          = condensation_pivot_threshold;
        // The border rows [T A_bk | Sbb | T | rb], T starting as the
        // injection of the R coupled rows.
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_F(S + r, S + c) = Sbb[r * B + c];
            for (int a = 0; a < R; ++a)
                TDLS_ARROW_LUPP_F(S + r, S + B + a) = r == R0 + a ? T(1) : T(0);
            for (int m = 0; m < nc; ++m)
                TDLS_ARROW_LUPP_F(S + r, rhs0 + m) = rb[m * B + r];
        }
        for (int k = 0; k < NB; ++k) {
            if (!((refused[k >> 5] >> (k & 31)) & 1u)) continue;
            // The rows of the block: [D_k | A_kb | 0 | Y_k].
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < S; ++j)
                    TDLS_ARROW_LUPP_F(i, j) =
                        TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, i, j));
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_F(i, S + c) =
                        TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c));
                for (int a = 0; a < R; ++a)
                    TDLS_ARROW_LUPP_F(i, S + B + a) = T(0);
                for (int m = 0; m < nc; ++m) {
                    if constexpr (canonical)
                        TDLS_ARROW_LUPP_F(i, rhs0 + m) = T(0);
                    else
                        TDLS_ARROW_LUPP_F(i, rhs0 + m) = TDLS_ARROW_LUPP_YW(m, B + k * S + i);
                }
            }
            // The border rows meet the columns of the block through T A_bk.
            for (int r = 0; r < B; ++r) {
                for (int j = 0; j < S; ++j) {
                    T v = T(0);
                    if constexpr (diag) {
                        v = TDLS_ARROW_LUPP_F(S + r, S + B + j) *
                            TDLS_ARROW_LUPP_A(storage_map::abk_diag(shape_value(), k, j));
                    } else {
                        for (int a = 0; a < B; ++a)
                            v += TDLS_ARROW_LUPP_F(S + r, S + B + a) *
                                 TDLS_ARROW_LUPP_A(storage_map::abk(shape_value(), k, a, j));
                    }
                    TDLS_ARROW_LUPP_F(S + r, j) = v;
                }
            }
            // S pivots over the S + B rows: a row of the block when it
            // reaches u times the largest candidate, the largest otherwise.
            bool block_row[front_rows];
            for (int e = 0; e < front_rows; ++e)
                block_row[e] = e < S;
            for (int j = 0; j < S; ++j) {
                int pmax = j, pblk = -1;
                T vmax = T(0), vblk = T(0);
                for (int e = j; e < front_rows; ++e) {
                    const T v = detail::abs(TDLS_ARROW_LUPP_F(e, j));
                    if (v > vmax) {
                        vmax = v;
                        pmax = e;
                    }
                    if (block_row[e] && v > vblk) {
                        vblk = v;
                        pblk = e;
                    }
                }
                if (!(vmax > T(0))) return false;
                const int p = pblk >= 0 && vblk >= u * vmax ? pblk : pmax;
                if (p != j) {
                    for (int c = j; c < fcols; ++c) {
                        const T t               = TDLS_ARROW_LUPP_F(p, c);
                        TDLS_ARROW_LUPP_F(p, c) = TDLS_ARROW_LUPP_F(j, c);
                        TDLS_ARROW_LUPP_F(j, c) = t;
                    }
                    const bool t = block_row[p];
                    block_row[p] = block_row[j];
                    block_row[j] = t;
                }
                const T inv             = T(1) / TDLS_ARROW_LUPP_F(j, j);
                TDLS_ARROW_LUPP_F(j, j) = inv;
                for (int e = j + 1; e < front_rows; ++e) {
                    const T l = TDLS_ARROW_LUPP_F(e, j) * inv;
                    if (l == T(0)) continue;
                    for (int c = j + 1; c < fcols; ++c)
                        TDLS_ARROW_LUPP_F(e, c) -= l * TDLS_ARROW_LUPP_F(j, c);
                }
            }
            if constexpr (!canonical) {
                // [G_k | H_k | c_k] = U^-1 [V | Z | h] on the S pivot rows,
                // then stored in place of A_kb, D_k (and after the front)
                // and Y_k.
                for (int c = S; c < fcols; ++c) {
                    for (int i = S - 1; i >= 0; --i) {
                        T t = TDLS_ARROW_LUPP_F(i, c);
                        for (int m = i + 1; m < S; ++m)
                            t -= TDLS_ARROW_LUPP_F(i, m) * TDLS_ARROW_LUPP_F(m, c);
                        TDLS_ARROW_LUPP_F(i, c) = t * TDLS_ARROW_LUPP_F(i, i);
                    }
                }
                for (int i = 0; i < S; ++i) {
                    for (int c = 0; c < B; ++c)
                        TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c)) =
                            TDLS_ARROW_LUPP_F(i, S + c);
                    for (int a = 0; a < R; ++a) {
                        if (a < S)
                            TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, i, a)) =
                                TDLS_ARROW_LUPP_F(i, S + B + a);
                        else
                            TDLS_ARROW_LUPP_W(front_rows * fcols + (k * S + i) * extra_cols + a -
                                              S) = TDLS_ARROW_LUPP_F(i, S + B + a);
                    }
                    for (int m = 0; m < nc; ++m)
                        TDLS_ARROW_LUPP_YW(m, B + k * S + i) = TDLS_ARROW_LUPP_F(i, rhs0 + m);
                }
            }
        }
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                Sbb[r * B + c] = TDLS_ARROW_LUPP_F(S + r, S + c);
            for (int m = 0; m < nc; ++m)
                rb[m * B + r] = TDLS_ARROW_LUPP_F(S + r, rhs0 + m);
        }
        return true;
    }
};



#undef TDLS_ARROW_LUPP_A
#undef TDLS_ARROW_LUPP_Y
#undef TDLS_ARROW_LUPP_W
#undef TDLS_ARROW_LUPP_F
#undef TDLS_ARROW_LUPP_FOR



} // namespace tdls

#if defined(__NVCC__)
#pragma nv_diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_STATIC_HPP
