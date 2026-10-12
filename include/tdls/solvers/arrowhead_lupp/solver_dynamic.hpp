#ifndef TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_DYNAMIC_HPP
#define TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_DYNAMIC_HPP



/// \file
/// \brief Runtime-shape variant of the ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Same algorithm as ArrowheadLUppSolverStatic (solver_static.hpp): static
/// condensation of the blocks with the pivot check, second pass of the
/// refused blocks, border solve, recovery. The difference is that the
/// shape is a runtime function argument instead of a template argument.
/// The storage, the coupling, the layout and the knobs stay compile-time.
/// The blocks and the border are solved by TiledLUppSolverDynamic.
///
/// Deliberate differences with the compile-time solver:
///   - No local array: a device has no array of runtime size. Every
///     temporary of the solver (the copy of a block, its border
///     multipliers, the solved right-hand sides, the accumulated Schur
///     complement, the border system...) lives in the workspace, of
///     workspace_size(shape) entries, and the pivots of the blocks and of
///     the border in a caller array of pivot_size(shape) ints.
///   - No residency booleans and no unroll pragma: every array is a
///     pointer plus a stride, and the unroll_loops knob is ignored.
///
/// For equal shapes and configurations, the results are bitwise identical
/// to the compile-time solver: the arithmetic sequence is the same, only
/// the addressing and the loop mechanics differ.
///
/// The compile-time solver is the performance path; this variant is the
/// flexibility path (shapes unknown at compile time, faster builds).
///
/// Preconditions: storage_map::valid(shape), and a workspace and a pivot
/// array of the sizes above. An rhs_pass_width above border + 1 counts as
/// border + 1 (the compile-time solver rejects it, its border being
/// known). Element offsets, index times stride, are computed in 64-bit
/// arithmetic.



#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/solvers/arrowhead_lupp/config.hpp>
#include <tdls/solvers/arrowhead_lupp/storage_map.hpp>
#include <tdls/solvers/tiled_lupp/solver_dynamic.hpp>



namespace tdls {



/* Addressing macros. They expand inside member functions where the
   parameter names (A, A_stride, y, rhs_stride, work, work_stride, piv,
   piv_stride) are in scope. #undef'd at the end of this header. */

/// \def TDLS_ARROW_LUPP_DYN_A
/// \brief Strided stored entry at index idx of the matrix.
#define TDLS_ARROW_LUPP_DYN_A(idx) A[TDLS_STRIDED(idx, A_stride)]
/// \def TDLS_ARROW_LUPP_DYN_Y
/// \brief Strided entry i of the right-hand side and solution.
#define TDLS_ARROW_LUPP_DYN_Y(i) y[TDLS_STRIDED(i, rhs_stride)]
/// \def TDLS_ARROW_LUPP_DYN_W
/// \brief Strided entry e of the workspace.
#define TDLS_ARROW_LUPP_DYN_W(e) work[TDLS_STRIDED(e, work_stride)]
/// \def TDLS_ARROW_LUPP_DYN_F
/// \brief Entry (r, c) of the front of the second pass, row-major at the
/// start of the workspace.
#define TDLS_ARROW_LUPP_DYN_F(r, c) TDLS_ARROW_LUPP_DYN_W((r) * front_cols + (c))



/// \brief Arrowhead solver with a runtime shape.
/// \tparam T      scalar type (float, double or long double)
/// \tparam Config configuration (see ArrowheadLUppConfig)
template<typename T, ArrowheadLUppConfig<T> Config>
struct ArrowheadLUppSolverDynamic {
    /// Index of every entry of the matrix in its storage.
    using storage_map = ArrowheadStorageMap<Config.storage, Config.coupling, Config.layout>;

  private:
    static constexpr bool col  = Config.layout == MatrixLayout::ColMajor;
    static constexpr bool diag = Config.coupling == ArrowheadCoupling::Diagonal;
    /// Same paths as the compile-time solver, see solver_static.hpp.
    static constexpr bool multiplier_update = !col && !diag;
    static constexpr bool in_place          = Config.schur_in_place;

    static_assert(Config.condensation_pivot_threshold.is_finite() &&
                      T(Config.condensation_pivot_threshold) >= T(0) &&
                      T(Config.condensation_pivot_threshold) <= T(1),
                  "ArrowheadLUppSolverDynamic: condensation_pivot_threshold must be in [0, 1]");

    /// \brief Threshold of the pivot check and of the second pass, read
    /// once from the configuration at compile time (see
    /// ArrowheadLUppConfig::condensation_pivot_threshold).
    static constexpr T condensation_pivot_threshold = Config.condensation_pivot_threshold;
    static_assert(Config.rhs_pass_width >= 0,
                  "ArrowheadLUppSolverDynamic: rhs_pass_width must be >= 0");

    using BlockSolver  = TiledLUppSolverDynamic<T, detail::arrowhead_block_config(Config)>;
    using BorderSolver = TiledLUppSolverDynamic<T, detail::arrowhead_border_config(Config)>;

    /// \brief Sizes of a shape, the names of the compile-time solver.
    struct Sizes {
        int B, S, NB, R, R0, front_rows, front_cols, extra_cols, pass_width;
    };
    /// \brief Offsets of the temporaries in the workspace.
    struct Offsets {
        int extra, Dl, U, W, X, Mb, m, a, acoef, Sacc, racc, Sbb, rb, w, s, refused, block_row,
            total;
    };

    /// \brief The sizes of a shape.
    /// \param[in] shape shape of the matrix
    /// \return the border, block and coupling sizes, the front and the
    ///         pass width
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr Sizes
    sizes(const ArrowheadShape& shape) noexcept {
        Sizes z{};
        z.B          = shape.border;
        z.S          = shape.block;
        z.NB         = shape.blocks;
        z.R          = diag ? z.S : z.B;
        z.R0         = diag ? shape.coupling_row : 0;
        z.front_rows = z.S + z.B;
        z.front_cols = z.S + z.B + z.R + 1;
        z.extra_cols = z.R > z.S ? z.R - z.S : 0;
        z.pass_width = Config.rhs_pass_width == 0 || Config.rhs_pass_width > z.B + 1
                           ? z.B + 1
                           : Config.rhs_pass_width;
        return z;
    }

    /// \brief The offsets of the temporaries: the front and the columns of
    /// H_k beyond S first, as in the compile-time solver, then the arrays
    /// that solver keeps local.
    /// \param[in] shape shape of the matrix
    /// \return the offsets and the total
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr Offsets
    offsets(const ArrowheadShape& shape) noexcept {
        const Sizes z = sizes(shape);
        Offsets o{};
        int e   = z.front_rows * z.front_cols;
        o.extra = e;
        e += z.NB * z.S * z.extra_cols;
        o.Dl = e;
        e += z.S * z.S;
        o.U = e;
        e += z.S * z.S;
        o.W = e;
        e += (z.B + 1) * z.S;
        o.X = e;
        e += (z.B + 1) * z.S;
        o.Mb = e;
        e += z.R * z.S;
        o.m = e;
        e += z.S;
        o.a = e;
        e += z.S;
        o.acoef = e;
        e += z.S;
        o.Sacc = e;
        e += z.R * z.B;
        o.racc = e;
        e += z.R;
        o.Sbb = e;
        e += z.B * z.B;
        o.rb = e;
        e += z.B;
        o.w = e;
        e += z.R;
        o.s = e;
        e += z.S;
        o.refused = e;
        e += z.NB;
        o.block_row = e;
        e += z.front_rows;
        o.total = e;
        return o;
    }

  public:
    /// \brief Entries of the workspace.
    /// \param[in] shape shape of the matrix
    /// \return the number of workspace entries a call needs
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    workspace_size(const ArrowheadShape& shape) noexcept {
        return offsets(shape).total;
    }

    /// \brief Entries of the pivot array: the pivots of a block, then of
    /// the border system, reuse it.
    /// \param[in] shape shape of the matrix
    /// \return max(block, border)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    pivot_size(const ArrowheadShape& shape) noexcept {
        return shape.block > shape.border ? shape.block : shape.border;
    }

    /// \brief Solves A x = y in place, counting the refused blocks.
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in,out] A             matrix in its storage (pattern entries
    ///                              only); recovery data on exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of workspace_size(shape)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride    element stride of y
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check, eliminated by the second pass
    /// \return false when the matrix is singular
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
                  int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
                  const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                  int& refused_count) noexcept {
        const Sizes z   = sizes(shape);
        const Offsets o = offsets(shape);
        const int B = z.B, R = z.R, R0 = z.R0, NB = z.NB;
        if constexpr (!in_place) {
            for (int r = 0; r < R; ++r) {
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_DYN_W(o.Sacc + r * B + c) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, R0 + r, c));
                TDLS_ARROW_LUPP_DYN_W(o.racc + r) = TDLS_ARROW_LUPP_DYN_Y(R0 + r);
            }
        }
        refused_count = 0;
        for (int k = 0; k < NB; ++k) {
            const bool ok = condense(shape, z, o, A, A_stride, piv, piv_stride, work, work_stride,
                                     y, rhs_stride, k);
            TDLS_ARROW_LUPP_DYN_W(o.refused + k) = ok ? T(0) : T(1);
            if (!ok) ++refused_count;
        }
        // The border system: the accumulated rows, the others as stored.
        for (int r = 0; r < B; ++r) {
            const bool acc = !in_place && r >= R0 && r < R0 + R;
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    acc ? TDLS_ARROW_LUPP_DYN_W(o.Sacc + (r - R0) * B + c)
                        : TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, r, c));
            TDLS_ARROW_LUPP_DYN_W(o.rb + r) =
                acc ? TDLS_ARROW_LUPP_DYN_W(o.racc + r - R0) : TDLS_ARROW_LUPP_DYN_Y(r);
        }
        if (refused_count > 0 &&
            !second_pass(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride))
            return false;
        if (!BorderSolver::solve_inplace(B, work + TDLS_STRIDED(o.Sbb, work_stride), work_stride,
                                         piv, piv_stride, work + TDLS_STRIDED(o.rb, work_stride),
                                         work_stride))
            return false;
        for (int r = 0; r < B; ++r)
            TDLS_ARROW_LUPP_DYN_Y(r) = TDLS_ARROW_LUPP_DYN_W(o.rb + r);
        recover(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride);
        return true;
    }

    /// \brief Solves A x = y in place.
    /// \param[in]     shape       shape of the matrix (valid)
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride  element stride of piv
    /// \param[out]    work        workspace of workspace_size(shape) entries
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \return false when the matrix is singular
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
                  int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
                  const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride) noexcept {
        int refused_count = 0;
        return solve_inplace(shape, A, A_stride, piv, piv_stride, work, work_stride, y, rhs_stride,
                             refused_count);
    }

    /// \brief Solves A x = b, b left untouched, counting the refused
    /// blocks: b is copied into x, then solved in place.
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in,out] A             matrix in its storage; recovery data on
    ///                              exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of workspace_size(shape)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     b             right-hand side
    /// \param[out]    x             solution
    /// \param[in]     rhs_stride    element stride of b and x
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check
    /// \return false when the matrix is singular
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
          int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
          const int work_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          int& refused_count) noexcept {
        const int n = shape.n();
        for (int i = 0; i < n; ++i)
            x[TDLS_STRIDED(i, rhs_stride)] = b[TDLS_STRIDED(i, rhs_stride)];
        return solve_inplace(shape, A, A_stride, piv, piv_stride, work, work_stride, x, rhs_stride,
                             refused_count);
    }

    /// \brief Solves A x = b, b left untouched.
    /// \param[in]     shape       shape of the matrix (valid)
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride  element stride of piv
    /// \param[out]    work        workspace of workspace_size(shape) entries
    /// \param[in]     work_stride element stride of work
    /// \param[in]     b           right-hand side
    /// \param[out]    x           solution
    /// \param[in]     rhs_stride  element stride of b and x
    /// \return false when the matrix is singular
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
          int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
          const int work_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x,
          const int rhs_stride) noexcept {
        int refused_count = 0;
        return solve(shape, A, A_stride, piv, piv_stride, work, work_stride, b, x, rhs_stride,
                     refused_count);
    }

  private:
    /// \brief Condenses block k into the accumulated rows of the border
    /// system; false when the block fails the pivot check, then left
    /// untouched. The arithmetic of ArrowheadLUppSolverStatic::condense,
    /// on the workspace.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand side
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     k           block
    /// \return true when the block is condensed
    TDLS_HOST_DEVICE static constexpr bool condense(const ArrowheadShape& shape, const Sizes& z,
                                                    const Offsets& o, T* TDLS_RESTRICT A,
                                                    const int A_stride, int* TDLS_RESTRICT piv,
                                                    const int piv_stride, T* TDLS_RESTRICT work,
                                                    const int work_stride, T* TDLS_RESTRICT y,
                                                    const int rhs_stride, const int k) noexcept {
        const int B = z.B, S = z.S;
        const T u   = condensation_pivot_threshold;
        T* const Dl = work + TDLS_STRIDED(o.Dl, work_stride);
        T* const W  = work + TDLS_STRIDED(o.W, work_stride);
        // The block, in the order of its storage.
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                TDLS_ARROW_LUPP_DYN_W(o.Dl + (col ? j * S + i : i * S + j)) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, j));
        if (!BlockSolver::factorize(S, Dl, work_stride, piv, piv_stride)) return false;
        if constexpr (diag) {
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_W(o.acoef + i) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abk_diag(shape, k, i));
        }
        const bool forward_check = col && z.pass_width == B + 1;
        // The multipliers L, and U in logical row order unless the forward
        // stage of the solve gives the border multipliers.
        bool ok = true;
        for (int l = 0; l < S; ++l) {
            const int p = piv[TDLS_STRIDED(l, piv_stride)];
            for (int j = 0; j < S; ++j) {
                if (j < l) {
                    ok = ok && u * detail::abs(TDLS_ARROW_LUPP_DYN_W(o.Dl + p * S + j)) <= T(1);
                } else if (!forward_check) {
                    TDLS_ARROW_LUPP_DYN_W(o.U + l * S + j) =
                        TDLS_ARROW_LUPP_DYN_W(o.Dl + p * S + j);
                }
            }
        }
        if (forward_check) {
            for (int i = 0; i < S; ++i) {
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_DYN_W(o.W + c * S + i) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c));
                TDLS_ARROW_LUPP_DYN_W(o.W + B * S + i) = TDLS_ARROW_LUPP_DYN_Y(B + k * S + i);
            }
            BlockSolver::forward_inplace_multirhs(S, B + 1, Dl, work_stride, piv, piv_stride, W,
                                                  work_stride, S * work_stride);
            // Column c of U^-T A_kb is row c of the border multipliers.
            for (int e = 0; e < B * S; ++e)
                ok = ok && u * detail::abs(TDLS_ARROW_LUPP_DYN_W(o.W + e)) <= T(1);
            if (!ok) return false;
            BlockSolver::backward_inplace_multirhs(S, B + 1, Dl, work_stride, piv, piv_stride, W,
                                                   work_stride, S * work_stride);
            condense_pass(shape, z, o, A, A_stride, piv, piv_stride, work, work_stride, y,
                          rhs_stride, k, 0, B + 1, true);
        } else {
            ok = border_multipliers(shape, z, o, A, A_stride, work, work_stride, k) && ok;
            if (!ok) return false;
            if constexpr (multiplier_update) {
                for (int l = 0; l < S; ++l)
                    for (int j = l; j < S; ++j)
                        TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, l, j)) =
                            TDLS_ARROW_LUPP_DYN_W(o.U + l * S + j);
            }
            for (int c0 = 0; c0 < B + 1; c0 += z.pass_width) {
                const int w = B + 1 - c0 < z.pass_width ? B + 1 - c0 : z.pass_width;
                condense_pass(shape, z, o, A, A_stride, piv, piv_stride, work, work_stride, y,
                              rhs_stride, k, c0, w, false);
            }
        }
        return true;
    }

    /// \brief Border multipliers M = Cb U^-1 of block k, checked against
    /// 1 / condensation_pivot_threshold; kept in the workspace (R x S)
    /// under the row-major layout with the dense coupling.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace (U, acoef in; Mb out)
    /// \param[in]     work_stride element stride of work
    /// \param[in]     k           block
    /// \return true when no multiplier exceeds the bound
    TDLS_HOST_DEVICE static constexpr bool
    border_multipliers(const ArrowheadShape& shape, const Sizes& z, const Offsets& o,
                       const T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                       const int work_stride, const int k) noexcept {
        const int B = z.B, S = z.S;
        bool ok = true;
        if constexpr (col) {
            for (int c = 0; c < B; ++c) {
                for (int i = 0; i < S; ++i)
                    TDLS_ARROW_LUPP_DYN_W(o.m + i) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c));
                ok = multipliers_pass(z, o, work, work_stride, o.m, 0) && ok;
            }
        } else if constexpr (diag) {
            // Border row coupling_row + i holds one entry, in column i.
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < S; ++j)
                    TDLS_ARROW_LUPP_DYN_W(o.m + j) =
                        j == i ? TDLS_ARROW_LUPP_DYN_W(o.acoef + i) : T(0);
                ok = multipliers_pass(z, o, work, work_stride, o.m, i) && ok;
            }
        } else {
            for (int r = 0; r < B; ++r) {
                for (int j = 0; j < S; ++j)
                    TDLS_ARROW_LUPP_DYN_W(o.Mb + r * S + j) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, j));
                ok = multipliers_pass(z, o, work, work_stride, o.Mb + r * S, 0) && ok;
            }
        }
        return ok;
    }

    /// \brief Turns one border row, at offset m of the workspace, into its
    /// multipliers c U^-1, in place, from column `first` on.
    /// \param[in]     z           sizes of the shape
    /// \param[in]     o           the offsets of the workspace (U)
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in]     m           offset of the row in the workspace
    /// \param[in]     first       first nonzero column of the row
    /// \return true when no multiplier exceeds 1 / condensation_pivot_threshold
    TDLS_HOST_DEVICE static constexpr bool multipliers_pass(const Sizes& z, const Offsets& o,
                                                            T* TDLS_RESTRICT work,
                                                            const int work_stride, const int m,
                                                            const int first) noexcept {
        const int S = z.S;
        const T u   = condensation_pivot_threshold;
        bool ok     = true;
        for (int l = first; l < S; ++l) {
            TDLS_ARROW_LUPP_DYN_W(m + l) *= TDLS_ARROW_LUPP_DYN_W(o.U + l * S + l);
            ok = ok && u * detail::abs(TDLS_ARROW_LUPP_DYN_W(m + l)) <= T(1);
            for (int j = l + 1; j < S; ++j)
                TDLS_ARROW_LUPP_DYN_W(m + j) -=
                    TDLS_ARROW_LUPP_DYN_W(m + l) * TDLS_ARROW_LUPP_DYN_W(o.U + l * S + j);
        }
        return ok;
    }

    /// \brief Subtracts v from entry (R0 + r, c) of the Schur complement
    /// (c = B: from its right-hand side), in place or in the workspace.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand side
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     r           accumulated row
    /// \param[in]     c           border column, or B for the right-hand side
    /// \param[in]     v           value subtracted
    TDLS_HOST_DEVICE static constexpr void
    schur_sub([[maybe_unused]] const ArrowheadShape& shape, const Sizes& z,
              [[maybe_unused]] const Offsets& o, [[maybe_unused]] T* TDLS_RESTRICT A,
              [[maybe_unused]] const int A_stride, [[maybe_unused]] T* TDLS_RESTRICT work,
              [[maybe_unused]] const int work_stride, [[maybe_unused]] T* TDLS_RESTRICT y,
              [[maybe_unused]] const int rhs_stride, const int r, const int c, const T v) noexcept {
        if constexpr (in_place) {
            if (c < z.B)
                TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, z.R0 + r, c)) -= v;
            else
                TDLS_ARROW_LUPP_DYN_Y(z.R0 + r) -= v;
        } else {
            if (c < z.B)
                TDLS_ARROW_LUPP_DYN_W(o.Sacc + r * z.B + c) -= v;
            else
                TDLS_ARROW_LUPP_DYN_W(o.racc + r) -= v;
        }
    }

    /// \brief Columns c0 .. c0 + w of [A_kb | y_k]: solved (unless solved
    /// holds), stored in place, folded into the accumulated rows. The
    /// arithmetic of ArrowheadLUppSolverStatic::condense_pass.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         pivots of the block
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand side
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     k           block
    /// \param[in]     c0          first column of the pass
    /// \param[in]     w           columns of the pass
    /// \param[in]     solved      whether the workspace holds the solved
    ///                            columns already
    TDLS_HOST_DEVICE static constexpr void
    condense_pass(const ArrowheadShape& shape, const Sizes& z, const Offsets& o, T* TDLS_RESTRICT A,
                  const int A_stride, const int* TDLS_RESTRICT piv, const int piv_stride,
                  T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                  const int rhs_stride, const int k, const int c0, const int w,
                  const bool solved) noexcept {
        const int B = z.B, S = z.S, R = z.R;
        const T* const Dl = work + TDLS_STRIDED(o.Dl, work_stride);
        T* const W        = work + TDLS_STRIDED(o.W, work_stride);
        if (!solved) {
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < w; ++j)
                    TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i) =
                        c0 + j < B ? TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c0 + j))
                                   : TDLS_ARROW_LUPP_DYN_Y(B + k * S + i);
            if constexpr (multiplier_update) {
                BlockSolver::forward_inplace_multirhs(S, w, Dl, work_stride, piv, piv_stride, W,
                                                      work_stride, S * work_stride);
            } else if constexpr (Config.gather_rhs) {
                T* const X = work + TDLS_STRIDED(o.X, work_stride);
                BlockSolver::substitute_multirhs(S, w, Dl, work_stride, piv, piv_stride, W, X,
                                                 work_stride, S * work_stride);
                for (int e = 0; e < w * S; ++e)
                    TDLS_ARROW_LUPP_DYN_W(o.W + e) = TDLS_ARROW_LUPP_DYN_W(o.X + e);
            } else {
                BlockSolver::substitute_inplace_multirhs(S, w, Dl, work_stride, piv, piv_stride, W,
                                                         work_stride, S * work_stride);
            }
        }
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < w; ++j) {
                if (c0 + j < B)
                    TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c0 + j)) =
                        TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i);
                else
                    TDLS_ARROW_LUPP_DYN_Y(B + k * S + i) = TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i);
            }
        if constexpr (multiplier_update) {
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < w; ++j) {
                    T sum = T(0);
                    for (int l = 0; l < S; ++l)
                        sum += TDLS_ARROW_LUPP_DYN_W(o.Mb + r * S + l) *
                               TDLS_ARROW_LUPP_DYN_W(o.W + j * S + l);
                    schur_sub(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride, r, c0 + j,
                              sum);
                }
        } else if constexpr (diag) {
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < w; ++j)
                    schur_sub(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride, i, c0 + j,
                              TDLS_ARROW_LUPP_DYN_W(o.acoef + i) *
                                  TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i));
        } else {
            for (int r = 0; r < B; ++r) {
                for (int i = 0; i < S; ++i)
                    TDLS_ARROW_LUPP_DYN_W(o.a + i) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, i));
                for (int j = 0; j < w; ++j) {
                    T sum = T(0);
                    for (int i = 0; i < S; ++i)
                        sum +=
                            TDLS_ARROW_LUPP_DYN_W(o.a + i) * TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i);
                    schur_sub(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride, r, c0 + j,
                              sum);
                }
            }
        }
    }

    /// \brief Recovers the blocks from the border solution, the last block
    /// first. The arithmetic of ArrowheadLUppSolverStatic::recover.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     A           matrix holding the recovery data
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace (rb, refused flags, H_k)
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           the recovery right-hand sides on entry,
    ///                            the solution on exit
    /// \param[in]     rhs_stride  element stride of y
    TDLS_HOST_DEVICE static constexpr void recover(const ArrowheadShape& shape, const Sizes& z,
                                                   const Offsets& o, const T* TDLS_RESTRICT A,
                                                   const int A_stride, T* TDLS_RESTRICT work,
                                                   const int work_stride, T* TDLS_RESTRICT y,
                                                   const int rhs_stride) noexcept {
        const int B = z.B, S = z.S, NB = z.NB, R = z.R;
        for (int a = 0; a < R; ++a)
            TDLS_ARROW_LUPP_DYN_W(o.w + a) = T(0);
        for (int k = NB - 1; k >= 0; --k) {
            for (int i = 0; i < S; ++i) {
                TDLS_ARROW_LUPP_DYN_W(o.s + i) = TDLS_ARROW_LUPP_DYN_Y(B + k * S + i);
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_DYN_W(o.s + i) -=
                        TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c)) *
                        TDLS_ARROW_LUPP_DYN_W(o.rb + c);
            }
            if (TDLS_ARROW_LUPP_DYN_W(o.refused + k) != T(0)) {
                // x_k = c_k - G_k xb - H_k w, then w += A_bk x_k.
                for (int i = 0; i < S; ++i)
                    for (int a = 0; a < R; ++a)
                        TDLS_ARROW_LUPP_DYN_W(o.s + i) -=
                            (a < S ? TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, a))
                                   : TDLS_ARROW_LUPP_DYN_W(o.extra + (k * S + i) * z.extra_cols +
                                                           a - S)) *
                            TDLS_ARROW_LUPP_DYN_W(o.w + a);
                if constexpr (diag) {
                    for (int i = 0; i < S; ++i)
                        TDLS_ARROW_LUPP_DYN_W(o.w + i) +=
                            TDLS_ARROW_LUPP_DYN_A(storage_map::abk_diag(shape, k, i)) *
                            TDLS_ARROW_LUPP_DYN_W(o.s + i);
                } else {
                    for (int r = 0; r < B; ++r)
                        for (int i = 0; i < S; ++i)
                            TDLS_ARROW_LUPP_DYN_W(o.w + r) +=
                                TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, i)) *
                                TDLS_ARROW_LUPP_DYN_W(o.s + i);
                }
            } else if constexpr (multiplier_update) {
                // x_k = U^-1 s, U in the upper triangle of D_k.
                for (int l = S - 1; l >= 0; --l) {
                    for (int j = l + 1; j < S; ++j)
                        TDLS_ARROW_LUPP_DYN_W(o.s + l) -=
                            TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, l, j)) *
                            TDLS_ARROW_LUPP_DYN_W(o.s + j);
                    TDLS_ARROW_LUPP_DYN_W(o.s + l) *=
                        TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, l, l));
                }
            }
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_Y(B + k * S + i) = TDLS_ARROW_LUPP_DYN_W(o.s + i);
        }
    }

    /// \brief Second pass: eliminates every refused block together with the
    /// border rows, in the front at the start of the workspace. The
    /// arithmetic of ArrowheadLUppSolverStatic::second_pass.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace (front, H_k, Sbb, rb, flags)
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand side
    /// \param[in]     rhs_stride  element stride of y
    /// \return false when a column of a refused block has no nonzero
    ///         candidate: the matrix is singular
    TDLS_HOST_DEVICE TDLS_NO_INLINE static constexpr bool
    second_pass(const ArrowheadShape& shape, const Sizes& z, const Offsets& o, T* TDLS_RESTRICT A,
                const int A_stride, T* TDLS_RESTRICT work, const int work_stride,
                T* TDLS_RESTRICT y, const int rhs_stride) noexcept {
        const int B = z.B, S = z.S, NB = z.NB, R = z.R, R0 = z.R0;
        const int front_rows = z.front_rows, front_cols = z.front_cols;
        const T u = condensation_pivot_threshold;
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_F(S + r, S + c) = TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c);
            for (int a = 0; a < R; ++a)
                TDLS_ARROW_LUPP_DYN_F(S + r, S + B + a) = r == R0 + a ? T(1) : T(0);
            TDLS_ARROW_LUPP_DYN_F(S + r, front_cols - 1) = TDLS_ARROW_LUPP_DYN_W(o.rb + r);
        }
        for (int k = 0; k < NB; ++k) {
            if (TDLS_ARROW_LUPP_DYN_W(o.refused + k) == T(0)) continue;
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < S; ++j)
                    TDLS_ARROW_LUPP_DYN_F(i, j) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, j));
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_DYN_F(i, S + c) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c));
                for (int a = 0; a < R; ++a)
                    TDLS_ARROW_LUPP_DYN_F(i, S + B + a) = T(0);
                TDLS_ARROW_LUPP_DYN_F(i, front_cols - 1) = TDLS_ARROW_LUPP_DYN_Y(B + k * S + i);
            }
            for (int r = 0; r < B; ++r) {
                for (int j = 0; j < S; ++j) {
                    T v = T(0);
                    if constexpr (diag) {
                        v = TDLS_ARROW_LUPP_DYN_F(S + r, S + B + j) *
                            TDLS_ARROW_LUPP_DYN_A(storage_map::abk_diag(shape, k, j));
                    } else {
                        for (int a = 0; a < B; ++a)
                            v += TDLS_ARROW_LUPP_DYN_F(S + r, S + B + a) *
                                 TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, a, j));
                    }
                    TDLS_ARROW_LUPP_DYN_F(S + r, j) = v;
                }
            }
            for (int e = 0; e < front_rows; ++e)
                TDLS_ARROW_LUPP_DYN_W(o.block_row + e) = e < S ? T(1) : T(0);
            for (int j = 0; j < S; ++j) {
                int pmax = j, pblk = -1;
                T vmax = T(0), vblk = T(0);
                for (int e = j; e < front_rows; ++e) {
                    const T v = detail::abs(TDLS_ARROW_LUPP_DYN_F(e, j));
                    if (v > vmax) {
                        vmax = v;
                        pmax = e;
                    }
                    if (TDLS_ARROW_LUPP_DYN_W(o.block_row + e) != T(0) && v > vblk) {
                        vblk = v;
                        pblk = e;
                    }
                }
                if (!(vmax > T(0))) return false;
                const int p = pblk >= 0 && vblk >= u * vmax ? pblk : pmax;
                if (p != j) {
                    for (int c = j; c < front_cols; ++c) {
                        const T t                   = TDLS_ARROW_LUPP_DYN_F(p, c);
                        TDLS_ARROW_LUPP_DYN_F(p, c) = TDLS_ARROW_LUPP_DYN_F(j, c);
                        TDLS_ARROW_LUPP_DYN_F(j, c) = t;
                    }
                    const T t                              = TDLS_ARROW_LUPP_DYN_W(o.block_row + p);
                    TDLS_ARROW_LUPP_DYN_W(o.block_row + p) = TDLS_ARROW_LUPP_DYN_W(o.block_row + j);
                    TDLS_ARROW_LUPP_DYN_W(o.block_row + j) = t;
                }
                const T inv                 = T(1) / TDLS_ARROW_LUPP_DYN_F(j, j);
                TDLS_ARROW_LUPP_DYN_F(j, j) = inv;
                for (int e = j + 1; e < front_rows; ++e) {
                    const T l = TDLS_ARROW_LUPP_DYN_F(e, j) * inv;
                    if (l == T(0)) continue;
                    for (int c = j + 1; c < front_cols; ++c)
                        TDLS_ARROW_LUPP_DYN_F(e, c) -= l * TDLS_ARROW_LUPP_DYN_F(j, c);
                }
            }
            for (int c = S; c < front_cols; ++c) {
                for (int i = S - 1; i >= 0; --i) {
                    T t = TDLS_ARROW_LUPP_DYN_F(i, c);
                    for (int m = i + 1; m < S; ++m)
                        t -= TDLS_ARROW_LUPP_DYN_F(i, m) * TDLS_ARROW_LUPP_DYN_F(m, c);
                    TDLS_ARROW_LUPP_DYN_F(i, c) = t * TDLS_ARROW_LUPP_DYN_F(i, i);
                }
            }
            for (int i = 0; i < S; ++i) {
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c)) =
                        TDLS_ARROW_LUPP_DYN_F(i, S + c);
                for (int a = 0; a < R; ++a) {
                    if (a < S)
                        TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, a)) =
                            TDLS_ARROW_LUPP_DYN_F(i, S + B + a);
                    else
                        TDLS_ARROW_LUPP_DYN_W(o.extra + (k * S + i) * z.extra_cols + a - S) =
                            TDLS_ARROW_LUPP_DYN_F(i, S + B + a);
                }
                TDLS_ARROW_LUPP_DYN_Y(B + k * S + i) = TDLS_ARROW_LUPP_DYN_F(i, front_cols - 1);
            }
        }
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) = TDLS_ARROW_LUPP_DYN_F(S + r, S + c);
            TDLS_ARROW_LUPP_DYN_W(o.rb + r) = TDLS_ARROW_LUPP_DYN_F(S + r, front_cols - 1);
        }
        return true;
    }
};



#undef TDLS_ARROW_LUPP_DYN_A
#undef TDLS_ARROW_LUPP_DYN_Y
#undef TDLS_ARROW_LUPP_DYN_W
#undef TDLS_ARROW_LUPP_DYN_F



} // namespace tdls



#endif // TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_DYNAMIC_HPP
