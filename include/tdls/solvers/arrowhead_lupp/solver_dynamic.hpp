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
///   - A streamed block (StreamedBlock) is a view of the workspace, where
///     the provider writes every entry of the block, zeros included, and
///     the solver factors and solves it;
///     the streamed entry points need streamed_workspace_size(shape)
///     entries: the block is written where the stored path copies it, and
///     A_bk under the dense coupling and the copies that restore a refused
///     block come after the areas of the stored path.
///     The provider must not throw: the entry points are noexcept.
///
/// For equal shapes and configurations, the results are bitwise identical
/// to the compile-time solver: the arithmetic sequence is the same, only
/// the addressing and the loop mechanics differ. A compiler that fuses
/// multiplications and additions differently in the two (GCC with
/// -march=native does) changes the last bits. -ffp-contract=off
/// (--fmad=false for nvcc) keeps them identical.
///
/// The compile-time solver is the performance path; this variant is the
/// flexibility path (shapes unknown at compile time, faster builds).
///
/// Preconditions: a workspace and a pivot array of the sizes above. The
/// entry points check the shape first, valid(shape, nrhs), and return
/// false on a shape they reject. An rhs_pass_width above border + nrhs
/// counts as border + nrhs, one pass (the compile-time solver rejects a
/// width above border + 1, its border being known). Element offsets,
/// index times stride, are computed in 64-bit arithmetic.



#include <limits>

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
/// \def TDLS_ARROW_LUPP_DYN_YW
/// \brief Strided entry i of column m of the right-hand sides and
/// solutions. The function declares ycols, its number of columns when
/// fixed at compile time (0 otherwise): with one, m is ignored and the
/// macro is TDLS_ARROW_LUPP_DYN_Y exactly.
#define TDLS_ARROW_LUPP_DYN_YW(m, i)                                                               \
    y[TDLS_STRIDED(i, rhs_stride) +                                                                \
      (ycols == 1 ? std::size_t(0) : TDLS_STRIDED_WIDE(m, xcol_stride))]
/// \def TDLS_ARROW_LUPP_DYN_W
/// \brief Strided entry e of the workspace.
#define TDLS_ARROW_LUPP_DYN_W(e) work[TDLS_STRIDED(e, work_stride)]
/// \def TDLS_ARROW_LUPP_DYN_F
/// \brief Entry (r, c) of the front of the second pass, row-major at the
/// start of the workspace.
#define TDLS_ARROW_LUPP_DYN_F(r, c) TDLS_ARROW_LUPP_DYN_W((r) * front_cols + (c))
/// \def TDLS_ARROW_LUPP_DYN_LU
/// \brief Entry (i, j) of the factors of the block in the workspace, by
/// rows (physical_solve).
#define TDLS_ARROW_LUPP_DYN_LU(i, j) TDLS_ARROW_LUPP_DYN_W(o.Dl + (i) * S + (j))
/// \def TDLS_ARROW_LUPP_DYN_RC
/// \brief Entry i of column c of the right-hand sides of the block in the
/// workspace (physical_solve).
#define TDLS_ARROW_LUPP_DYN_RC(c, i) TDLS_ARROW_LUPP_DYN_W(o.W + (c) * S + (i))



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
    /// Same paths as the compile-time solver, see solver_static.hpp: the
    /// update by the border multipliers under the row-major layout with
    /// the dense coupling and ArrowheadCondensation::MinFlops.
    static constexpr bool multiplier_update =
        !col && !diag && Config.condensation == ArrowheadCondensation::MinFlops;
    static constexpr bool in_place = Config.schur_in_place;

    static_assert(Config.condensation_pivot_threshold.is_finite() &&
                      T(Config.condensation_pivot_threshold) >= T(0) &&
                      T(Config.condensation_pivot_threshold) <= T(1),
                  "ArrowheadLUppSolverDynamic: condensation_pivot_threshold must be in [0, 1]");

    /// \brief Threshold of the pivot check and of the second pass, read
    /// once from the configuration at compile time (see
    /// ArrowheadLUppConfig::condensation_pivot_threshold).
    static constexpr T condensation_pivot_threshold = Config.condensation_pivot_threshold;
    static_assert(Config.rhs_pass_width >= -1,
                  "ArrowheadLUppSolverDynamic: rhs_pass_width must be >= -1");

    using BlockSolver  = TiledLUppSolverDynamic<T, detail::arrowhead_block_config(Config)>;
    using BorderSolver = TiledLUppSolverDynamic<T, detail::arrowhead_border_config(Config)>;
    /// Solver of the diagonal blocks in registers: TiledLUpp with
    /// physical row interchanges.
    using PhysicalBlockSolver =
        TiledLUppSolverDynamic<T, detail::arrowhead_physical_block_config(Config)>;

    /// \brief Sizes of a shape, the names of the compile-time solver, for a
    /// number of right-hand sides (front_cols, pass_width).
    struct Sizes {
        int B, S, NB, R, R0, front_rows, front_cols, extra_cols, pass_width;
        /// Blocks eliminated in registers (ArrowheadLUppConfig::block_row_interchange,
        /// Auto resolved for this block size).
        bool reg;
    };
    /// \brief Offsets of the temporaries in the workspace, of type I: int
    /// for the solve, long long for the size check of valid().
    template<typename I>
    struct OffsetsOf {
        I extra, Dl, U, W, X, Mb, m, a, acoef, Sacc, racc, Sbb, rb, w, s, refused, block_row, total;
    };
    /// \brief Offsets of the temporaries in the workspace.
    using Offsets = OffsetsOf<int>;
    /// \brief Offsets of the areas of a streamed block, after those of
    /// Offsets (streamed entry points only), of type I.
    template<typename I>
    struct StreamedOffsetsOf {
        I Abk, Dsave, Wsave, total;
    };
    /// \brief Offsets of the areas of a streamed block.
    using StreamedOffsets = StreamedOffsetsOf<int>;

    /// \brief The sizes of a shape.
    /// \param[in] shape shape of the matrix
    /// \param[in] nc    number of right-hand sides
    /// \param[in] ny    right-hand sides carried through the blocks
    /// \return the border, block and coupling sizes, the front and the
    ///         pass width
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr Sizes
    sizes(const ArrowheadShape& shape, const int nc, const int ny) noexcept {
        Sizes z{};
        z.B                                  = shape.border;
        z.S                                  = shape.block;
        z.NB                                 = shape.blocks;
        z.R                                  = diag ? z.S : z.B;
        z.R0                                 = diag ? shape.coupling_row : 0;
        z.front_rows                         = z.S + z.B;
        z.front_cols                         = z.S + z.B + z.R + nc;
        z.extra_cols                         = z.R > z.S ? z.R - z.S : 0;
        constexpr RowInterchange interchange = Config.block_row_interchange;
        constexpr int width                  = Config.rhs_pass_width;
        z.reg          = detail::arrowhead_register_blocks_here<T>(interchange, z.S);
        const int pass = detail::arrowhead_rhs_pass_width_here(width, z.reg);
        z.pass_width   = pass == 0 || pass > z.B + ny ? z.B + ny : pass;
        return z;
    }

    /// \brief The offsets of the temporaries: the front and the columns of
    /// H_k beyond S first, as in the compile-time solver, then the arrays
    /// that solver keeps local, sized for nc right-hand sides.
    /// \tparam I integer type of the offsets: int for the solve, long long
    ///           for the size check of valid() (nc <= 2^30 then)
    /// \param[in] shape shape of the matrix
    /// \param[in] nc    number of right-hand sides
    /// \return the offsets and the total
    template<typename I = int>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr OffsetsOf<I>
    offsets(const ArrowheadShape& shape, const int nc) noexcept {
        const Sizes z = sizes(shape, nc, nc);
        OffsetsOf<I> o{};
        I e     = I(z.front_rows) * z.front_cols;
        o.extra = e;
        e += I(z.NB) * z.S * z.extra_cols;
        o.Dl = e;
        e += I(z.S) * z.S;
        o.U = e;
        e += I(z.S) * z.S;
        o.W = e;
        e += (I(z.B) + nc) * z.S;
        o.X = e;
        e += (I(z.B) + nc) * z.S;
        o.Mb = e;
        e += I(z.R) * z.S;
        o.m = e;
        e += z.S;
        o.a = e;
        e += z.S;
        o.acoef = e;
        e += z.S;
        o.Sacc = e;
        e += I(z.R) * z.B;
        o.racc = e;
        e += I(z.R) * nc;
        o.Sbb = e;
        e += I(z.B) * z.B;
        o.rb = e;
        e += I(z.B) * nc;
        o.w = e;
        e += I(z.R) * nc;
        o.s = e;
        e += I(z.S) * nc;
        o.refused = e;
        e += z.NB;
        o.block_row = e;
        e += z.front_rows;
        o.total = e;
        return o;
    }

    /// \brief The offsets of the areas of a streamed block, after the
    /// temporaries: A_bk under the dense coupling (the diagonal couplings
    /// go to acoef), and the copies of D_k and of [A_kb | y_k] that restore
    /// a refused block (one right-hand side at most in the blocks).
    /// \tparam I integer type of the offsets (see offsets)
    /// \param[in] shape shape of the matrix
    /// \param[in] nc    number of right-hand sides
    /// \return the offsets, and the total of the workspace
    template<typename I = int>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr StreamedOffsetsOf<I>
    streamed_offsets(const ArrowheadShape& shape, const int nc) noexcept {
        const Sizes z = sizes(shape, nc, nc);
        StreamedOffsetsOf<I> so{};
        I e    = offsets<I>(shape, nc).total;
        so.Abk = e;
        e += diag ? I(0) : I(z.B) * z.S;
        so.Dsave = e;
        e += I(z.S) * z.S;
        so.Wsave = e;
        e += (I(z.B) + 1) * z.S;
        so.total = e;
        return so;
    }

  public:
    /// \brief Whether the solver accepts a shape, with nrhs right-hand
    /// sides (unit columns, tangent columns): storage_map::valid(shape),
    /// nrhs in [1, 2^30], and the workspaces of the entry points below 2^31
    /// entries, their offsets being ints. The entry points return false on
    /// a shape they reject, before any access, and the sizes are 0 for it.
    /// The checks cannot overflow, whatever the shape.
    /// \param[in] shape shape of the matrix
    /// \param[in] nrhs  number of right-hand sides
    /// \return true when the solver accepts the shape
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool valid(const ArrowheadShape& shape,
                                                               const int nrhs = 1) noexcept {
        return storage_map::valid(shape) && nrhs >= 1 && nrhs <= (1 << 30) &&
               streamed_offsets<long long>(shape, nrhs).total < (1LL << 31);
    }

    /// \brief Entries of the workspace of solve_inplace and solve.
    /// \param[in] shape shape of the matrix
    /// \return the number of workspace entries a call needs, 0 for a
    ///         shape the solver rejects
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    workspace_size(const ArrowheadShape& shape) noexcept {
        return valid(shape) ? offsets(shape, 1).total : 0;
    }

    /// \brief Entries of the workspace of the entry points with nrhs
    /// right-hand sides: the temporaries of the right-hand sides and one
    /// column of the front per right-hand side.
    /// \param[in] shape shape of the matrix
    /// \param[in] nrhs  number of right-hand sides
    /// \return the number of workspace entries a call needs, 0 for a
    ///         shape the solver rejects
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int workspace_size(const ArrowheadShape& shape,
                                                                       const int nrhs) noexcept {
        return valid(shape, nrhs) ? offsets(shape, nrhs).total : 0;
    }

    /// \brief Entries of the workspace of solve_inplace_streamed: those of
    /// solve_inplace, then A_bk of a block under the dense coupling and the
    /// copies that restore a refused block.
    /// \param[in] shape shape of the matrix
    /// \return the number of workspace entries a call needs, 0 for a
    ///         shape the solver rejects
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    streamed_workspace_size(const ArrowheadShape& shape) noexcept {
        return valid(shape) ? streamed_offsets(shape, 1).total : 0;
    }

    /// \brief Entries of the workspace of
    /// solve_border_canonical_multirhs_streamed with nrhs unit columns, and
    /// of solve_inplace_streamed_or_tangent with nrhs tangent columns.
    /// \param[in] shape shape of the matrix
    /// \param[in] nrhs  number of columns
    /// \return the number of workspace entries a call needs, 0 for a
    ///         shape the solver rejects
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    streamed_workspace_size(const ArrowheadShape& shape, const int nrhs) noexcept {
        return valid(shape, nrhs) ? streamed_offsets(shape, nrhs).total : 0;
    }

    /// \brief Entries of the pivot array: the pivots of a block, then of
    /// the border system, reuse it.
    /// \param[in] shape shape of the matrix
    /// \return max(block, border), 0 for a shape the solver rejects
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    pivot_size(const ArrowheadShape& shape) noexcept {
        if (!valid(shape)) return 0;
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
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
                  int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
                  const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                  int& refused_count) noexcept {
        return solve_core<1, false>(shape, 1, A, A_stride, piv, piv_stride, work, work_stride, y,
                                    rhs_stride, 0, 0, refused_count);
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
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
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
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
          int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
          const int work_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          int& refused_count) noexcept {
        if (!valid(shape)) {
            refused_count = 0;
            return false;
        }
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
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve(const ArrowheadShape& shape, T* TDLS_RESTRICT A, const int A_stride,
          int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
          const int work_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x,
          const int rhs_stride) noexcept {
        int refused_count = 0;
        return solve(shape, A, A_stride, piv, piv_stride, work, work_stride, b, x, rhs_stride,
                     refused_count);
    }

    /// \brief Solves A X = Y in place for nrhs right-hand sides, counting
    /// the refused blocks. The right-hand sides ride along the
    /// condensation, which runs once. Column m comes out bitwise identical to
    /// solve_inplace on it alone under the default rhs_pass_width, within
    /// rounding errors under another (see ArrowheadLUppConfig::rhs_pass_width),
    /// and bitwise identical to the compile-time solver (the same contraction
    /// of multiply-adds assumed, see the file documentation).
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in]     nrhs          number of right-hand sides, >= 1
    /// \param[in,out] A             matrix in its storage (pattern entries
    ///                              only); recovery data on exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of workspace_size(shape, nrhs)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             nrhs right-hand-side columns on entry,
    ///                              the solutions on exit
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     xcol_stride   element stride between columns of y
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check, eliminated by the second pass
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace_multirhs(const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A,
                           const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
                           T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                           const int rhs_stride, const std::size_t xcol_stride,
                           int& refused_count) noexcept {
        return solve_core<0, false>(shape, nrhs, A, A_stride, piv, piv_stride, work, work_stride, y,
                                    rhs_stride, xcol_stride, 0, refused_count);
    }

    /// \brief Solves A X = Y in place for nrhs right-hand sides.
    /// \param[in]     shape       shape of the matrix (valid)
    /// \param[in]     nrhs        number of right-hand sides, >= 1
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride  element stride of piv
    /// \param[out]    work        workspace of workspace_size(shape, nrhs)
    ///                            entries
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           nrhs right-hand-side columns on entry, the
    ///                            solutions on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace_multirhs(const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A,
                           const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
                           T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                           const int rhs_stride, const std::size_t xcol_stride) noexcept {
        int refused_count = 0;
        return solve_inplace_multirhs(shape, nrhs, A, A_stride, piv, piv_stride, work, work_stride,
                                      y, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Solves A X = B for nrhs right-hand sides, B left untouched,
    /// counting the refused blocks: B is copied into X, then solved in
    /// place.
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in]     nrhs          number of right-hand sides, >= 1
    /// \param[in,out] A             matrix in its storage; recovery data on
    ///                              exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of workspace_size(shape, nrhs)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     b             nrhs right-hand-side columns
    /// \param[out]    x             the nrhs solutions
    /// \param[in]     rhs_stride    element stride of b and x
    /// \param[in]     xcol_stride   element stride between columns of b and x
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_multirhs(const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A,
                   const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
                   T* TDLS_RESTRICT work, const int work_stride, const T* TDLS_RESTRICT b,
                   T* TDLS_RESTRICT x, const int rhs_stride, const std::size_t xcol_stride,
                   int& refused_count) noexcept {
        if (!valid(shape, nrhs)) {
            refused_count = 0;
            return false;
        }
        const int n = shape.n();
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < n; ++i)
                x[TDLS_STRIDED(i, rhs_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)] =
                    b[TDLS_STRIDED(i, rhs_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)];
        return solve_inplace_multirhs(shape, nrhs, A, A_stride, piv, piv_stride, work, work_stride,
                                      x, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Solves A X = B for nrhs right-hand sides, B left untouched.
    /// \param[in]     shape       shape of the matrix (valid)
    /// \param[in]     nrhs        number of right-hand sides, >= 1
    /// \param[in,out] A           matrix in its storage; recovery data on exit
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride  element stride of piv
    /// \param[out]    work        workspace of workspace_size(shape, nrhs)
    ///                            entries
    /// \param[in]     work_stride element stride of work
    /// \param[in]     b           nrhs right-hand-side columns
    /// \param[out]    x           the nrhs solutions
    /// \param[in]     rhs_stride  element stride of b and x
    /// \param[in]     xcol_stride element stride between columns of b and x
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_multirhs(const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A,
                   const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
                   T* TDLS_RESTRICT work, const int work_stride, const T* TDLS_RESTRICT b,
                   T* TDLS_RESTRICT x, const int rhs_stride,
                   const std::size_t xcol_stride) noexcept {
        int refused_count = 0;
        return solve_multirhs(shape, nrhs, A, A_stride, piv, piv_stride, work, work_stride, b, x,
                              rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Columns col0 .. col0 + nrhs - 1 of the border block of A^-1
    /// (col0 + nrhs <= border), counting the refused blocks; see
    /// ArrowheadLUppSolverStatic::solve_border_canonical_multirhs.
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in]     nrhs          number of unit columns, in [1, border]
    /// \param[in,out] A             matrix in its storage; unchanged on exit,
    ///                              but under schur_in_place
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of workspace_size(shape, nrhs)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     col0          first unit column, col0 + nrhs <= border
    /// \param[out]    x             the border rows, border entries per
    ///                              column
    /// \param[in]     rhs_stride    element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of blocks refused by the pivot
    ///                              check
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool solve_border_canonical_multirhs(
        const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A, const int A_stride,
        int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work, const int work_stride,
        const int col0, T* TDLS_RESTRICT x, const int rhs_stride, const std::size_t xcol_stride,
        int& refused_count) noexcept {
        return solve_core<0, true>(shape, nrhs, A, A_stride, piv, piv_stride, work, work_stride, x,
                                   rhs_stride, xcol_stride, col0, refused_count);
    }

    /// \brief Columns col0 .. col0 + nrhs - 1 of the border block of A^-1
    /// (col0 + nrhs <= border).
    /// \param[in]     shape       shape of the matrix (valid)
    /// \param[in]     nrhs        number of unit columns, in [1, border]
    /// \param[in,out] A           matrix in its storage; unchanged on exit,
    ///                            but under schur_in_place
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride  element stride of piv
    /// \param[out]    work        workspace of workspace_size(shape, nrhs)
    ///                            entries
    /// \param[in]     work_stride element stride of work
    /// \param[in]     col0        first unit column, col0 + nrhs <= border
    /// \param[out]    x           the border rows, border entries per column
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_border_canonical_multirhs(const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A,
                                    const int A_stride, int* TDLS_RESTRICT piv,
                                    const int piv_stride, T* TDLS_RESTRICT work,
                                    const int work_stride, const int col0, T* TDLS_RESTRICT x,
                                    const int rhs_stride, const std::size_t xcol_stride) noexcept {
        int refused_count = 0;
        return solve_border_canonical_multirhs(shape, nrhs, A, A_stride, piv, piv_stride, work,
                                               work_stride, col0, x, rhs_stride, xcol_stride,
                                               refused_count);
    }

    /// \brief Block k of a streamed solve: a view of the areas of the
    /// workspace where the solver factors and solves the block, which the
    /// caller fills through the accessors when the solver asks for it; the
    /// factorization and the solve then run in place; only a save, after the
    /// areas of the stored path, is copied to store a refused block as given.
    /// The accessors are those of ArrowheadLUppSolverStatic::StreamedBlock, so
    /// that one provider serves both solvers. Entries in logical order whatever
    /// the layout and the storage. The areas hold the previous block: the
    /// caller writes every entry, zeros included, by assignment.
    struct StreamedBlock {
        T* Dl;      ///< D_k in the layout of the factorization
        T* W;       ///< [A_kb | y_k], column c at c * S
        T* Abk;     ///< A_bk: coupled rows (diagonal) or B x S (dense)
        int S;      ///< block size
        int B;      ///< border size
        int stride; ///< element stride of the workspace
        /// \param[in] i row of the block
        /// \param[in] j column of the block
        /// \return entry (i, j) of D_k
        TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& d(const int i,
                                                              const int j) const noexcept {
            return Dl[TDLS_STRIDED(col ? j * S + i : i * S + j, stride)];
        }
        /// \param[in] i row of the block
        /// \param[in] c border column
        /// \return entry (i, c) of A_kb
        TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& akb(const int i,
                                                                const int c) const noexcept {
            return W[TDLS_STRIDED(c * S + i, stride)];
        }
        /// \param[in] i row of the block
        /// \return entry i of y_k, the right-hand side on the rows of the
        ///         block
        TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& y(const int i) const noexcept {
            return W[TDLS_STRIDED(B * S + i, stride)];
        }
        /// \param[in] i row and column of the block
        /// \return the coupling A_bk(coupling_row + i, i) (diagonal coupling)
        TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& coupling(const int i) const noexcept {
            return Abk[TDLS_STRIDED(i, stride)];
        }
        /// \param[in] r border row
        /// \param[in] i column of the block
        /// \return entry (r, i) of A_bk (dense coupling)
        TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& abk(const int r,
                                                                const int i) const noexcept {
            return Abk[TDLS_STRIDED(r * S + i, stride)];
        }
    };

    /// \brief Block k of a recomputed solve (solve_inplace_recomputed,
    /// solve_inplace_recomputed_or_tangent): a StreamedBlock, the same view
    /// and accessors, and whether the solver asks for the block again, in its
    /// recovery. A provider taking a StreamedBlock& serves too; one taking a
    /// RecomputedBlock&, or auto&, reads the flag. The streamed entry points
    /// keep the StreamedBlock alone, so that their code is that of a solver
    /// without the recomputed ones.
    struct RecomputedBlock : StreamedBlock {
        /// Set when the solver asks for the block again, in its recovery: the
        /// provider may then skip what it accumulates once per pass, a
        /// residual for instance. False while it condenses the block.
        bool recovery = false;
    };

    /// \brief Outcome of solve_inplace_streamed_or_tangent.
    enum class StreamedOutcome {
        solved,  ///< y holds the solution (converged() returned false)
        tangent, ///< converged: x holds the canonical border columns, y
                 ///< intermediate values
        singular ///< the matrix is singular, or the shape rejected
    };

    /// \brief Solves A x = y in place, the blocks streamed: provider(k, blk)
    /// fills block k (StreamedBlock) when the solver needs it, k = 0, 1, ...
    /// in order, exactly once, and the solver condenses it at once. The
    /// caller writes beforehand only the border: A_bb in A and the border rows
    /// of y. An accepted block leaves its recovery data in its storage; a
    /// refused one is stored as given for the second pass. Bit for bit the
    /// results of solve_inplace, and of
    /// ArrowheadLUppSolverStatic::solve_inplace_streamed (the same contraction
    /// of multiply-adds assumed, see the file documentation).
    /// \tparam Provider callable (int k, StreamedBlock& blk)
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in]     provider      the source of the blocks
    /// \param[in,out] A             storage: A_bb on entry; recovery data on
    ///                              exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of streamed_workspace_size(shape)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the solution on exit
    /// \param[in]     rhs_stride    element stride of y
    /// \param[out]    refused_count number of refused blocks
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    template<typename Provider>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace_streamed(const ArrowheadShape& shape, Provider&& provider, T* TDLS_RESTRICT A,
                           const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
                           T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                           const int rhs_stride, int& refused_count) noexcept {
        return solve_core<1, false>(shape, 1, A, A_stride, piv, piv_stride, work, work_stride, y,
                                    rhs_stride, 0, 0, refused_count, &provider);
    }

    /// \brief solve_border_canonical_multirhs with the blocks streamed (see
    /// solve_inplace_streamed; blk.y is not read).
    /// \tparam Provider callable (int k, StreamedBlock& blk)
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in]     nrhs          number of unit columns, in [1, border]
    /// \param[in]     provider      the source of the blocks
    /// \param[in,out] A             storage: A_bb on entry; on exit the
    ///                              refused blocks as given, the accepted
    ///                              ones not written
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of
    ///                              streamed_workspace_size(shape, nrhs)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     col0          first unit column, col0 + nrhs <= border
    /// \param[out]    x             the border rows, border entries per
    ///                              column
    /// \param[in]     rhs_stride    element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of refused blocks
    /// \return false when the matrix is singular, or the shape rejected
    ///         (see valid)
    template<typename Provider>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool solve_border_canonical_multirhs_streamed(
        const ArrowheadShape& shape, const int nrhs, Provider&& provider, T* TDLS_RESTRICT A,
        const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
        const int work_stride, const int col0, T* TDLS_RESTRICT x, const int rhs_stride,
        const std::size_t xcol_stride, int& refused_count) noexcept {
        return solve_core<0, true>(shape, nrhs, A, A_stride, piv, piv_stride, work, work_stride, x,
                                   rhs_stride, xcol_stride, col0, refused_count, &provider);
    }

    /// \brief One streamed pass of a Newton iteration: the blocks condensed
    /// as solve_inplace_streamed does, then converged() is called (the
    /// provider may compute the residual meanwhile). When it returns false,
    /// the pass ends as solve_inplace_streamed (y holds the correction).
    /// When it returns true, the correction is not computed: the border
    /// system condensed by the pass gives instead columns
    /// col0 .. col0 + ntan - 1 of the border block of A^-1 in x (the consistent
    /// tangent of a behaviour loaded through the border), refused blocks
    /// included. The provider is called exactly once per block, in order, and
    /// converged() once, after the last block. Bit for bit the results of
    /// solve_inplace_streamed and of
    /// ArrowheadLUppSolverStatic::solve_inplace_streamed_or_tangent, and those
    /// of solve_border_canonical_multirhs_streamed under the default
    /// rhs_pass_width (within rounding errors under another, see
    /// ArrowheadLUppConfig::rhs_pass_width), the same contraction of
    /// multiply-adds assumed (see the file documentation).
    /// \tparam Provider  callable (int k, StreamedBlock& blk)
    /// \tparam Converged callable () returning bool
    /// \param[in]     shape         shape of the matrix (valid)
    /// \param[in]     ntan          number of tangent columns, in [1, border]
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     converged     called after the last block: whether the
    ///                              tangent replaces the correction
    /// \param[in,out] A             storage: A_bb on entry; overwritten
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of
    ///                              streamed_workspace_size(shape, ntan)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the correction on exit (solved),
    ///                              intermediate values otherwise
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     col0          first tangent column, col0 + ntan <= border
    /// \param[out]    x             the tangent columns (tangent), border
    ///                              entries per column
    /// \param[in]     x_stride      element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of refused blocks
    /// \return solved, tangent, or singular when the matrix is singular or
    ///         the shape rejected (see valid)
    template<typename Provider, typename Converged>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr StreamedOutcome
    solve_inplace_streamed_or_tangent(const ArrowheadShape& shape, const int ntan,
                                      Provider&& provider, Converged&& converged,
                                      T* TDLS_RESTRICT A, const int A_stride,
                                      int* TDLS_RESTRICT piv, const int piv_stride,
                                      T* TDLS_RESTRICT work, const int work_stride,
                                      T* TDLS_RESTRICT y, const int rhs_stride, const int col0,
                                      T* TDLS_RESTRICT x, const int x_stride,
                                      const std::size_t xcol_stride, int& refused_count) noexcept {
        if (!valid(shape, ntan)) {
            refused_count = 0;
            return StreamedOutcome::singular;
        }
        // The pass carries one right-hand side through the blocks (y, with
        // one column); the workspace holds both phases, ntan columns in the
        // front and in the border right-hand sides.
        [[maybe_unused]] constexpr int ycols = 1;
        constexpr int y_xcol_stride          = 0;
        const Sizes z                        = sizes(shape, 1, 1);
        const Offsets o                      = offsets(shape, ntan);
        const StreamedOffsets so             = streamed_offsets(shape, ntan);
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
            StreamedBlock blk = streamed_block(z, o, so, work, work_stride);
            provider(k, blk);
            const bool ok = condense_or_store_streamed<1, false>(shape, z, o, so, A, A_stride, piv,
                                                                 piv_stride, work, work_stride, y,
                                                                 rhs_stride, y_xcol_stride, k, 1);
            TDLS_ARROW_LUPP_DYN_W(o.refused + k) = ok ? T(0) : T(1);
            if (!ok) ++refused_count;
        }
        // The rows as stored and the accumulated rows in separate loops, as
        // in solve_core.
        const int acc0 = in_place ? B : R0;
        const int acc1 = in_place ? B : R0 + R;
        for (int r = 0; r < acc0; ++r)
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, r, c));
        for (int r = acc0; r < acc1; ++r)
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    TDLS_ARROW_LUPP_DYN_W(o.Sacc + (r - R0) * B + c);
        for (int r = acc1; r < B; ++r)
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, r, c));
        T* const Sbb = work + TDLS_STRIDED(o.Sbb, work_stride);
        T* const rb  = work + TDLS_STRIDED(o.rb, work_stride);
        if (converged()) {
            const Sizes zt = sizes(shape, ntan, 0);
            for (int m = 0; m < ntan; ++m)
                for (int r = 0; r < B; ++r)
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = r == col0 + m ? T(1) : T(0);
            // The refused blocks, stored as given, by the second pass of the
            // canonical solve, which writes neither x nor y.
            if (refused_count > 0 &&
                !second_pass<0, true>(shape, zt, o, A, A_stride, work, work_stride, y, rhs_stride,
                                      y_xcol_stride, ntan))
                return StreamedOutcome::singular;
            const bool ok = ntan == 1 ? BorderSolver::solve_inplace(B, Sbb, work_stride, piv,
                                                                    piv_stride, rb, work_stride)
                                      : BorderSolver::solve_inplace_multirhs(
                                            B, ntan, Sbb, work_stride, piv, piv_stride, rb,
                                            work_stride, TDLS_STRIDED(B, work_stride));
            if (!ok) return StreamedOutcome::singular;
            for (int m = 0; m < ntan; ++m)
                for (int r = 0; r < B; ++r)
                    x[TDLS_STRIDED(r, x_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)] =
                        TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r);
            return StreamedOutcome::tangent;
        }
        for (int r = 0; r < acc0; ++r)
            TDLS_ARROW_LUPP_DYN_W(o.rb + r) = TDLS_ARROW_LUPP_DYN_Y(r);
        for (int r = acc0; r < acc1; ++r)
            TDLS_ARROW_LUPP_DYN_W(o.rb + r) = TDLS_ARROW_LUPP_DYN_W(o.racc + r - R0);
        for (int r = acc1; r < B; ++r)
            TDLS_ARROW_LUPP_DYN_W(o.rb + r) = TDLS_ARROW_LUPP_DYN_Y(r);
        if (refused_count > 0 && !second_pass<1, false>(shape, z, o, A, A_stride, work, work_stride,
                                                        y, rhs_stride, y_xcol_stride, 1))
            return StreamedOutcome::singular;
        if (!BorderSolver::solve_inplace(B, Sbb, work_stride, piv, piv_stride, rb, work_stride))
            return StreamedOutcome::singular;
        for (int r = 0; r < B; ++r)
            TDLS_ARROW_LUPP_DYN_Y(r) = TDLS_ARROW_LUPP_DYN_W(o.rb + r);
        recover<1>(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride, y_xcol_stride, 1);
        return StreamedOutcome::solved;
    }

    /// \brief Outcome of solve_inplace_recomputed and
    /// solve_inplace_recomputed_or_tangent.
    enum class RecomputedOutcome {
        solved,   ///< y holds the solution
        tangent,  ///< converged (solve_inplace_recomputed_or_tangent): x holds
                  ///< the tangent columns, y is untouched
        singular, ///< the matrix is singular, or the shape or the capacity
                  ///< rejected; y is untouched
        deferred  ///< more blocks refused than the capacity of the store; y is
                  ///< untouched, for a solve that stores them
                  ///< (solve_inplace_streamed)
    };

    /// \brief Whether the recomputed entry points accept a shape, a
    /// capacity and nrhs tangent columns: valid(shape, nrhs), capacity in
    /// [0, blocks], their workspace below 2^31 entries, and every block
    /// index exact in T (2^24 blocks at most in float), the store keeping
    /// the index of the block of each of its places in the workspace.
    /// \param[in] shape    shape of the matrix
    /// \param[in] capacity number of refused blocks the workspace holds
    /// \param[in] nrhs     number of tangent columns, 1 without tangent
    /// \return true when the recomputed entry points accept them
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    valid_recomputed(const ArrowheadShape& shape, const int capacity, const int nrhs = 1) noexcept {
        constexpr int digits =
            std::numeric_limits<T>::digits < 62 ? std::numeric_limits<T>::digits : 62;
        return valid(shape, nrhs) && capacity >= 0 && capacity <= shape.blocks &&
               static_cast<long long>(shape.blocks) <= (1LL << digits) &&
               recomputed_offsets<long long>(shape, capacity, nrhs).total < (1LL << 31);
    }

    /// \brief Entries of the workspace of solve_inplace_recomputed with a
    /// store of capacity refused blocks, and of
    /// solve_inplace_recomputed_or_tangent with nrhs tangent columns: the
    /// areas of a streamed block and of its condensation, the front of the
    /// second pass, and the refused blocks in the storage of a matrix of
    /// capacity blocks with their right-hand sides. It does not grow with
    /// the number of blocks of the matrix.
    /// \param[in] shape    shape of the matrix
    /// \param[in] capacity number of refused blocks the workspace holds
    /// \param[in] nrhs     number of tangent columns, 1 without tangent
    /// \return the number of workspace entries a call needs, 0 for a shape
    ///         or a capacity the solver rejects
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int
    recomputed_workspace_size(const ArrowheadShape& shape, const int capacity,
                              const int nrhs = 1) noexcept {
        return valid_recomputed(shape, capacity, nrhs)
                   ? recomputed_offsets(shape, capacity, nrhs).total
                   : 0;
    }

    /// \brief Solves A x = y in place, the blocks recomputed: the runtime
    /// counterpart of ArrowheadLUppSolverStatic::solve_inplace_recomputed,
    /// bit for bit (the same contraction of multiply-adds assumed).
    /// provider(k, blk) fills block k once while the solver condenses it, k
    /// = 0, 1, ... in order, and once more, blk.recovery set, while it
    /// recovers the unknowns of the block, in decreasing k. Nothing of an
    /// accepted block is stored, and the matrix is never stored: the caller
    /// gives the border block A_bb, entry (r, c) at (r * border + c) *
    /// Abb_stride, or (c * border + r) * Abb_stride under the column-major
    /// layout, and the border rows of y. The workspace holds the refused
    /// blocks, at most capacity of them; beyond, the solve stops, deferred.
    /// Nothing is written to y until the solution: y is untouched when the
    /// solve is deferred or the matrix singular.
    /// \tparam Provider callable (int k, StreamedBlock& blk)
    /// \param[in]     shape         shape of the matrix
    /// \param[in]     capacity      number of refused blocks the workspace
    ///                              holds, in [0, blocks]
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     Abb           the border block A_bb, B x B in the layout
    ///                              of the configuration
    /// \param[in]     Abb_stride    element stride of Abb
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of
    ///                              recomputed_workspace_size(shape, capacity)
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the solution on exit when solved
    /// \param[in]     rhs_stride    element stride of y
    /// \param[out]    refused_count number of refused blocks
    /// \return solved, singular (or a shape or a capacity rejected, see
    ///         valid_recomputed), or deferred when more than capacity blocks
    ///         are refused
    template<typename Provider>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr RecomputedOutcome
    solve_inplace_recomputed(const ArrowheadShape& shape, const int capacity, Provider&& provider,
                             const T* TDLS_RESTRICT Abb, const int Abb_stride,
                             int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
                             const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                             int& refused_count) noexcept {
        return recomputed_core(shape, capacity, 0, provider, Abb, Abb_stride, piv, piv_stride, work,
                               work_stride, y, rhs_stride, 0, nullptr, 1, 0, refused_count);
    }

    /// \brief One recomputed pass of a Newton iteration: the runtime
    /// counterpart of
    /// ArrowheadLUppSolverStatic::solve_inplace_recomputed_or_tangent, bit
    /// for bit (the same contraction of multiply-adds assumed). The blocks
    /// are condensed as solve_inplace_recomputed condenses them, then
    /// converged() is called: false, the pass ends as
    /// solve_inplace_recomputed; true, x receives columns col0 .. col0 +
    /// ntan - 1 of the border block of A^-1, refused blocks included, y is
    /// untouched, and the provider is called once per block. When more than
    /// capacity blocks are refused, the pass stops there, deferred, before
    /// converged() is called.
    /// \tparam Provider  callable (int k, StreamedBlock& blk)
    /// \tparam Converged callable () returning bool
    /// \param[in]     shape         shape of the matrix
    /// \param[in]     ntan          number of tangent columns, in [1, border]
    /// \param[in]     capacity      number of refused blocks the workspace
    ///                              holds, in [0, blocks]
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     converged     called after the last block: whether the
    ///                              tangent replaces the correction
    /// \param[in]     Abb           the border block A_bb, B x B in the layout
    ///                              of the configuration
    /// \param[in]     Abb_stride    element stride of Abb
    /// \param[out]    piv           pivot array of pivot_size(shape) ints
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace of
    ///                              recomputed_workspace_size(shape,
    ///                              capacity, ntan) entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the correction on exit when solved
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     col0          first tangent column, col0 + ntan <= border
    /// \param[out]    x             the tangent columns (tangent), border
    ///                              entries per column
    /// \param[in]     x_stride      element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of refused blocks
    /// \return solved, tangent, singular (or a shape or a capacity
    ///         rejected), or deferred when more than capacity blocks are
    ///         refused
    template<typename Provider, typename Converged>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr RecomputedOutcome
    solve_inplace_recomputed_or_tangent(
        const ArrowheadShape& shape, const int ntan, const int capacity, Provider&& provider,
        Converged&& converged, const T* TDLS_RESTRICT Abb, const int Abb_stride,
        int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work, const int work_stride,
        T* TDLS_RESTRICT y, const int rhs_stride, const int col0, T* TDLS_RESTRICT x,
        const int x_stride, const std::size_t xcol_stride, int& refused_count) noexcept {
        if (ntan < 1) {
            refused_count = 0;
            return RecomputedOutcome::singular;
        }
        return recomputed_core(shape, capacity, ntan, provider, Abb, Abb_stride, piv, piv_stride,
                               work, work_stride, y, rhs_stride, col0, x, x_stride, xcol_stride,
                               refused_count, &converged);
    }

  private:
    /// \brief Offsets of the areas of a recomputed solve after those of a
    /// streamed solve on the store (a matrix of capacity blocks), of type I:
    /// the store of the refused blocks, their right-hand sides, and the
    /// block each place of the store holds.
    template<typename I>
    struct RecomputedOffsetsOf {
        I store, ystore, slots, total;
    };
    /// \brief Offsets of the areas of a recomputed solve.
    using RecomputedOffsets = RecomputedOffsetsOf<int>;

    /// \brief The shape of the store of capacity refused blocks.
    /// \param[in] shape    shape of the matrix
    /// \param[in] capacity number of refused blocks
    /// \return {border, block, capacity, coupling_row}
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr ArrowheadShape
    store_shape(const ArrowheadShape& shape, const int capacity) noexcept {
        return {shape.border, shape.block, capacity, shape.coupling_row};
    }

    /// \brief The offsets of a recomputed solve: the temporaries and the
    /// streamed areas of the store (whose front holds nc columns), then the
    /// store, its right-hand sides and the blocks of its places.
    /// \tparam I integer type of the offsets (see offsets)
    /// \param[in] shape    shape of the matrix
    /// \param[in] capacity number of refused blocks
    /// \param[in] nc       columns of the front, one at least
    /// \return the offsets, and the total of the workspace
    template<typename I = int>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr RecomputedOffsetsOf<I>
    recomputed_offsets(const ArrowheadShape& shape, const int capacity, const int nc) noexcept {
        const ArrowheadShape st = store_shape(shape, capacity);
        RecomputedOffsetsOf<I> ro{};
        I e      = streamed_offsets<I>(st, nc).total;
        ro.store = e;
        e += capacity > 0 ? I(storage_map::size(st)) : I(0);
        ro.ystore = e;
        e += I(shape.border) + I(capacity) * shape.block;
        ro.slots = e;
        e += capacity;
        ro.total = e;
        return ro;
    }

    /// \brief Entry (r, c) of the border block given to the recomputed
    /// entry points, in the layout of the configuration.
    /// \param[in] Abb        the border block
    /// \param[in] Abb_stride element stride of Abb
    /// \param[in] B          border size
    /// \param[in] r          row
    /// \param[in] c          column
    /// \return A_bb(r, c)
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr T border_entry(const T* TDLS_RESTRICT Abb,
                                                                   const int Abb_stride,
                                                                   const int B, const int r,
                                                                   const int c) noexcept {
        return Abb[TDLS_STRIDED(col ? c * B + r : r * B + c, Abb_stride)];
    }

    /// \brief The recomputed solve shared by solve_inplace_recomputed (no
    /// converged(), ntan = 0) and solve_inplace_recomputed_or_tangent; the
    /// arithmetic of ArrowheadLUppSolverStatic::recomputed_core.
    /// \tparam Provider  callable (int k, StreamedBlock& blk)
    /// \tparam Converged none, or callable () returning bool
    /// \param[in]     shape         shape of the matrix
    /// \param[in]     capacity      number of refused blocks the workspace
    ///                              holds
    /// \param[in]     ntan          number of tangent columns, 0 without
    ///                              converged()
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     Abb           the border block
    /// \param[in]     Abb_stride    element stride of Abb
    /// \param[out]    piv           pivot array
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the solution on exit when solved
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     col0          first tangent column
    /// \param[out]    x             the tangent columns
    /// \param[in]     x_stride      element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of refused blocks
    /// \param[in]     converged     none, or whether the tangent replaces the
    ///                              correction
    /// \return the outcome
    template<typename Provider, typename... Converged>
    TDLS_HOST_DEVICE static constexpr RecomputedOutcome recomputed_core(
        const ArrowheadShape& shape, const int capacity, const int ntan, Provider& provider,
        const T* TDLS_RESTRICT Abb, const int Abb_stride, int* TDLS_RESTRICT piv,
        const int piv_stride, T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
        const int rhs_stride, [[maybe_unused]] const int col0, [[maybe_unused]] T* TDLS_RESTRICT x,
        [[maybe_unused]] const int x_stride, [[maybe_unused]] const std::size_t xcol_stride,
        int& refused_count, Converged*... converged) noexcept {
        static_assert(!in_place, "ArrowheadLUppSolverDynamic: the recomputed solve accumulates "
                                 "the Schur complement apart (schur_in_place off)");
        // Columns of the front: the tangent columns, one at least.
        const int fnc = ntan > 1 ? ntan : 1;
        refused_count = 0;
        if (!valid_recomputed(shape, capacity, fnc) || ntan > shape.border)
            return RecomputedOutcome::singular;
        [[maybe_unused]] constexpr int ycols = 1;
        const ArrowheadShape st              = store_shape(shape, capacity);
        const Sizes z                        = sizes(shape, 1, 1);
        const Offsets o                      = offsets(st, fnc);
        const StreamedOffsets so             = streamed_offsets(st, fnc);
        const RecomputedOffsets ro           = recomputed_offsets(shape, capacity, fnc);
        const int B = z.B, S = z.S, R = z.R, R0 = z.R0, NB = z.NB;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sacc + r * B + c) =
                    border_entry(Abb, Abb_stride, B, R0 + r, c);
            TDLS_ARROW_LUPP_DYN_W(o.racc + r) = TDLS_ARROW_LUPP_DYN_Y(R0 + r);
        }
        int count = 0;
        for (int k = 0; k < NB; ++k) {
            RecomputedBlock blk{streamed_block(z, o, so, work, work_stride), false};
            provider(k, blk);
            const int verdict = condense_or_keep_recomputed(
                shape, z, o, so, ro, capacity, piv, piv_stride, work, work_stride, k, count);
            refused_count = count;
            if (verdict == 2) return RecomputedOutcome::deferred;
        }
        // The flags of the places of the store, for its second pass.
        for (int j = 0; j < capacity; ++j)
            TDLS_ARROW_LUPP_DYN_W(o.refused + j) = j < count ? T(1) : T(0);
        T* const store  = work + TDLS_STRIDED(ro.store, work_stride);
        T* const ystore = work + TDLS_STRIDED(ro.ystore, work_stride);
        for (int r = 0; r < B; ++r) {
            const bool acc = r >= R0 && r < R0 + R;
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    acc ? TDLS_ARROW_LUPP_DYN_W(o.Sacc + (r - R0) * B + c)
                        : border_entry(Abb, Abb_stride, B, r, c);
        }
        T* const Sbb = work + TDLS_STRIDED(o.Sbb, work_stride);
        T* const rb  = work + TDLS_STRIDED(o.rb, work_stride);
        if constexpr (sizeof...(Converged) > 0) {
            if ((*streamed_arg(converged...))()) {
                // The tangent: unit columns, the refused blocks of the store
                // by the second pass of the canonical solve.
                const Sizes zt = sizes(st, fnc, 0);
                for (int m = 0; m < fnc; ++m)
                    for (int r = 0; r < B; ++r)
                        TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = r == col0 + m ? T(1) : T(0);
                if (count > 0 && !second_pass<0, true>(st, zt, o, store, work_stride, work,
                                                       work_stride, ystore, work_stride, 0, fnc))
                    return RecomputedOutcome::singular;
                const bool ok = fnc == 1 ? BorderSolver::solve_inplace(B, Sbb, work_stride, piv,
                                                                       piv_stride, rb, work_stride)
                                         : BorderSolver::solve_inplace_multirhs(
                                               B, fnc, Sbb, work_stride, piv, piv_stride, rb,
                                               work_stride, TDLS_STRIDED(B, work_stride));
                if (!ok) return RecomputedOutcome::singular;
                for (int m = 0; m < fnc; ++m)
                    for (int r = 0; r < B; ++r)
                        x[TDLS_STRIDED(r, x_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)] =
                            TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r);
                return RecomputedOutcome::tangent;
            }
        }
        for (int r = 0; r < B; ++r) {
            const bool acc = r >= R0 && r < R0 + R;
            TDLS_ARROW_LUPP_DYN_W(o.rb + r) =
                acc ? TDLS_ARROW_LUPP_DYN_W(o.racc + r - R0) : TDLS_ARROW_LUPP_DYN_Y(r);
        }
        if (count > 0 && !second_pass<1, false>(st, sizes(st, 1, 1), o, store, work_stride, work,
                                                work_stride, ystore, work_stride, 0, 1))
            return RecomputedOutcome::singular;
        if (!BorderSolver::solve_inplace(B, Sbb, work_stride, piv, piv_stride, rb, work_stride))
            return RecomputedOutcome::singular;
        // The accepted blocks, recomputed, the last one first: the places
        // of the store hold the refused ones in increasing order.
        int j = count - 1;
        for (int k = NB - 1; k >= 0; --k) {
            if (j >= 0 && TDLS_ARROW_LUPP_DYN_W(ro.slots + j) == T(k)) {
                --j;
                continue;
            }
            RecomputedBlock blk{streamed_block(z, o, so, work, work_stride), true};
            provider(k, blk);
            // y_k - A_kb x_b, then the column solved in place of A_kb(:, 0)
            for (int i = 0; i < S; ++i) {
                T v = TDLS_ARROW_LUPP_DYN_W(o.W + B * S + i);
                for (int c = 0; c < B; ++c)
                    v -= TDLS_ARROW_LUPP_DYN_W(o.W + c * S + i) * TDLS_ARROW_LUPP_DYN_W(o.rb + c);
                TDLS_ARROW_LUPP_DYN_W(o.s + i) = v;
            }
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_W(o.W + i) = TDLS_ARROW_LUPP_DYN_W(o.s + i);
            if (z.reg) {
                (void)physical_factor(z, o, work, work_stride, piv, piv_stride);
                physical_solve(z, o, work, work_stride, piv, piv_stride, 1, true);
            } else {
                T* const Dl = work + TDLS_STRIDED(o.Dl, work_stride);
                (void)BlockSolver::factorize(S, Dl, work_stride, piv, piv_stride);
                BlockSolver::substitute_inplace(S, Dl, work_stride, piv, piv_stride,
                                                work + TDLS_STRIDED(o.W, work_stride), work_stride);
            }
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_Y(B + k * S + i) = TDLS_ARROW_LUPP_DYN_W(o.W + i);
        }
        if (count > 0)
            recover_refused(shape, st, z, o, ro, work, work_stride, y, rhs_stride, count);
        for (int r = 0; r < B; ++r)
            TDLS_ARROW_LUPP_DYN_Y(r) = TDLS_ARROW_LUPP_DYN_W(o.rb + r);
        return RecomputedOutcome::solved;
    }

    /// \brief Condenses the streamed block k of a recomputed solve, nothing
    /// stored; a refused block is kept, as the provider gave it, in the next
    /// place of the store, or the solve is deferred when the store is full.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     so          the offsets of the streamed areas
    /// \param[in]     ro          the offsets of the store
    /// \param[in]     capacity    number of places of the store
    /// \param[out]    piv         pivot array
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] work        workspace, the block in its areas
    /// \param[in]     work_stride element stride of work
    /// \param[in]     k           block
    /// \param[in,out] count       number of refused blocks
    /// \return 0 when the block is condensed, 1 when it is refused and kept,
    ///         2 when it is refused beyond the capacity
    TDLS_HOST_DEVICE static constexpr int
    condense_or_keep_recomputed(const ArrowheadShape& shape, const Sizes& z, const Offsets& o,
                                const StreamedOffsets& so, const RecomputedOffsets& ro,
                                const int capacity, int* TDLS_RESTRICT piv, const int piv_stride,
                                T* TDLS_RESTRICT work, const int work_stride, const int k,
                                int& count) noexcept {
        const int B = z.B, S = z.S;
        const bool forward_check = !z.reg && col && z.pass_width == B + 1;
        for (int e = 0; e < S * S; ++e)
            TDLS_ARROW_LUPP_DYN_W(so.Dsave + e) = TDLS_ARROW_LUPP_DYN_W(o.Dl + e);
        if (forward_check)
            for (int e = 0; e < (B + 1) * S; ++e)
                TDLS_ARROW_LUPP_DYN_W(so.Wsave + e) = TDLS_ARROW_LUPP_DYN_W(o.W + e);
        if (condense<1, false, true>(shape, z, o, nullptr, 1, piv, piv_stride, work, work_stride,
                                     nullptr, 1, 0, k, 1, &so))
            return 0;
        ++count;
        if (count > capacity) return 2;
        // The block as given, in place j of the store (W is intact unless the
        // forward stage ran).
        const int j             = count - 1;
        const ArrowheadShape st = store_shape(shape, capacity);
        const int ow            = forward_check ? so.Wsave : o.W;
        for (int i = 0; i < S; ++i) {
            for (int jj = 0; jj < S; ++jj)
                TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::d(st, j, i, jj)) =
                    TDLS_ARROW_LUPP_DYN_W(so.Dsave + (col ? jj * S + i : i * S + jj));
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::akb(st, j, i, c)) =
                    TDLS_ARROW_LUPP_DYN_W(ow + c * S + i);
            TDLS_ARROW_LUPP_DYN_W(ro.ystore + B + j * S + i) =
                TDLS_ARROW_LUPP_DYN_W(ow + B * S + i);
        }
        if constexpr (diag) {
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::abk_diag(st, j, i)) =
                    TDLS_ARROW_LUPP_DYN_W(o.acoef + i);
        } else {
            for (int r = 0; r < B; ++r)
                for (int i = 0; i < S; ++i)
                    TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::abk(st, j, r, i)) =
                        TDLS_ARROW_LUPP_DYN_W(so.Abk + r * S + i);
        }
        TDLS_ARROW_LUPP_DYN_W(ro.slots + j) = T(k);
        return 1;
    }

    /// \brief Recovers the refused blocks of a recomputed solve from the
    /// store, the last one first: x_k = c_k - G_k x_b - H_k w, w gathering
    /// the couplings A_bj x_j (the R coupled rows) of the refused blocks
    /// recovered so far; the arithmetic of
    /// ArrowheadLUppSolverStatic::recover_refused.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     st          shape of the store
    /// \param[in]     z           sizes of the shape
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     ro          the offsets of the store
    /// \param[in,out] work        workspace: the store after the second pass
    /// \param[in]     work_stride element stride of work
    /// \param[out]    y           the unknowns of the refused blocks
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     count       number of refused blocks
    TDLS_HOST_DEVICE static constexpr void
    recover_refused(const ArrowheadShape& shape, const ArrowheadShape& st, const Sizes& z,
                    const Offsets& o, const RecomputedOffsets& ro, T* TDLS_RESTRICT work,
                    const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                    const int count) noexcept {
        (void)shape;
        const int B = z.B, S = z.S, R = z.R;
        for (int a = 0; a < R; ++a)
            TDLS_ARROW_LUPP_DYN_W(o.w + a) = T(0);
        for (int j = count - 1; j >= 0; --j) {
            const int k = static_cast<int>(TDLS_ARROW_LUPP_DYN_W(ro.slots + j));
            for (int i = 0; i < S; ++i) {
                T v = TDLS_ARROW_LUPP_DYN_W(ro.ystore + B + j * S + i);
                for (int c = 0; c < B; ++c)
                    v -= TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::akb(st, j, i, c)) *
                         TDLS_ARROW_LUPP_DYN_W(o.rb + c);
                for (int a = 0; a < R; ++a) {
                    const T h =
                        a < S ? TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::d(st, j, i, a))
                              : TDLS_ARROW_LUPP_DYN_W(o.extra + (j * S + i) * z.extra_cols + a - S);
                    v -= h * TDLS_ARROW_LUPP_DYN_W(o.w + a);
                }
                TDLS_ARROW_LUPP_DYN_W(o.s + i) = v;
            }
            if constexpr (diag) {
                for (int i = 0; i < S; ++i)
                    TDLS_ARROW_LUPP_DYN_W(o.w + i) +=
                        TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::abk_diag(st, j, i)) *
                        TDLS_ARROW_LUPP_DYN_W(o.s + i);
            } else {
                for (int r = 0; r < B; ++r)
                    for (int i = 0; i < S; ++i)
                        TDLS_ARROW_LUPP_DYN_W(o.w + r) +=
                            TDLS_ARROW_LUPP_DYN_W(ro.store + storage_map::abk(st, j, r, i)) *
                            TDLS_ARROW_LUPP_DYN_W(o.s + i);
            }
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_Y(B + k * S + i) = TDLS_ARROW_LUPP_DYN_W(o.s + i);
        }
    }

  private:
    /// \brief The single element of the trailing pack of the helpers shared
    /// by the stored and streamed paths: the provider or the offsets of the
    /// streamed areas, absent on the stored path, whose signatures and code
    /// so stay those of a solver without streaming.
    /// \tparam P type of the element
    /// \param[in] p the element
    /// \return p
    template<typename P>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL static constexpr P*
    streamed_arg(P* p) noexcept {
        return p;
    }

    /// \brief The view of the areas of the workspace a streamed block is
    /// written to.
    /// \param[in] z           sizes of the shape
    /// \param[in] o           the offsets of the workspace
    /// \param[in] so          the offsets of the streamed areas
    /// \param[in] work        workspace
    /// \param[in] work_stride element stride of work
    /// \return the block view
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL static constexpr StreamedBlock
    streamed_block(const Sizes& z, const Offsets& o, const StreamedOffsets& so,
                   T* TDLS_RESTRICT work, const int work_stride) noexcept {
        StreamedBlock blk{};
        blk.Dl     = work + TDLS_STRIDED(o.Dl, work_stride);
        blk.W      = work + TDLS_STRIDED(o.W, work_stride);
        blk.Abk    = work + TDLS_STRIDED(diag ? o.acoef : so.Abk, work_stride);
        blk.S      = z.S;
        blk.B      = z.B;
        blk.stride = work_stride;
        return blk;
    }

    /// \brief The solve shared by every entry point; the arithmetic of
    /// ArrowheadLUppSolverStatic::solve_core.
    /// \tparam NC        number of right-hand sides when fixed (1 for
    ///                   solve_inplace), 0 when given by nrhs
    /// \tparam canonical unit columns col0 .. col0 + nc - 1, border rows
    ///                   written to y (B rows per column)
    /// \param[in]     shape         shape of the matrix
    /// \param[in]     nrhs          number of right-hand sides when NC is 0
    /// \param[in,out] A             matrix in its storage
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    piv           pivot array
    /// \param[in]     piv_stride    element stride of piv
    /// \param[out]    work          workspace
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             right-hand sides on entry, solutions on
    ///                              exit (canonical: the border rows, out)
    /// \param[in]     rhs_stride    element stride of y
    /// \param[in]     xcol_stride   element stride between columns of y
    /// \tparam Provider  none on the stored path; streamed, the type of the
    ///                   callable (int k, StreamedBlock& blk) giving the
    ///                   blocks
    /// \param[in]     col0          first unit column (canonical)
    /// \param[out]    refused_count number of refused blocks
    /// \param[in]     provider      streamed: the source of the blocks
    /// \return false when the matrix is singular
    template<int NC, bool canonical, typename... Provider>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL static constexpr bool
    solve_core(const ArrowheadShape& shape, const int nrhs, T* TDLS_RESTRICT A, const int A_stride,
               int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT work,
               const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
               [[maybe_unused]] const std::size_t xcol_stride, [[maybe_unused]] const int col0,
               int& refused_count, Provider*... provider) noexcept {
        const int nc = NC > 0 ? NC : nrhs;
        if (!valid(shape, nc)) {
            refused_count = 0;
            return false;
        }
        const int ny = canonical ? 0 : nc;
        // Columns of y when fixed, for TDLS_ARROW_LUPP_DYN_YW.
        [[maybe_unused]] constexpr int ycols = NC;
        const Sizes z                        = sizes(shape, nc, ny);
        const Offsets o                      = offsets(shape, nc);
        const int B = z.B, R = z.R, R0 = z.R0, NB = z.NB;
        if constexpr (!in_place) {
            for (int r = 0; r < R; ++r) {
                for (int c = 0; c < B; ++c)
                    TDLS_ARROW_LUPP_DYN_W(o.Sacc + r * B + c) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, R0 + r, c));
                for (int m = 0; m < ny; ++m)
                    TDLS_ARROW_LUPP_DYN_W(o.racc + m * R + r) = TDLS_ARROW_LUPP_DYN_YW(m, R0 + r);
            }
        }
        refused_count = 0;
        if constexpr (sizeof...(Provider) > 0) {
            // Streamed: the provider writes block k in the workspace.
            const StreamedOffsets so = streamed_offsets(shape, nc);
            for (int k = 0; k < NB; ++k) {
                StreamedBlock blk = streamed_block(z, o, so, work, work_stride);
                (*streamed_arg(provider...))(k, blk);
                const bool ok = condense_or_store_streamed<NC, canonical>(
                    shape, z, o, so, A, A_stride, piv, piv_stride, work, work_stride, y, rhs_stride,
                    xcol_stride, k, nc);
                TDLS_ARROW_LUPP_DYN_W(o.refused + k) = ok ? T(0) : T(1);
                if (!ok) ++refused_count;
            }
        } else {
            for (int k = 0; k < NB; ++k) {
                const bool ok =
                    condense<NC, canonical>(shape, z, o, A, A_stride, piv, piv_stride, work,
                                            work_stride, y, rhs_stride, xcol_stride, k, nc);
                TDLS_ARROW_LUPP_DYN_W(o.refused + k) = ok ? T(0) : T(1);
                if (!ok) ++refused_count;
            }
        }
        // The border system: the accumulated rows, the others as stored;
        // column m of its right-hand sides at m * B.
        // The rows as stored and the accumulated rows in separate loops:
        // GCC 11 and 12 vectorize a choice of source per entry wrongly (-O3
        // with AVX2 or AVX-512).
        const int acc0 = in_place ? B : R0;
        const int acc1 = in_place ? B : R0 + R;
        for (int r = 0; r < acc0; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, r, c));
            for (int m = 0; m < nc; ++m) {
                if constexpr (canonical)
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = r == col0 + m ? T(1) : T(0);
                else
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = TDLS_ARROW_LUPP_DYN_YW(m, r);
            }
        }
        for (int r = acc0; r < acc1; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    TDLS_ARROW_LUPP_DYN_W(o.Sacc + (r - R0) * B + c);
            for (int m = 0; m < nc; ++m) {
                if constexpr (canonical)
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = r == col0 + m ? T(1) : T(0);
                else
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) =
                        TDLS_ARROW_LUPP_DYN_W(o.racc + m * R + r - R0);
            }
        }
        for (int r = acc1; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, r, c));
            for (int m = 0; m < nc; ++m) {
                if constexpr (canonical)
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = r == col0 + m ? T(1) : T(0);
                else
                    TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = TDLS_ARROW_LUPP_DYN_YW(m, r);
            }
        }
        if (refused_count > 0 &&
            !second_pass<NC, canonical>(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride,
                                        xcol_stride, nc))
            return false;
        if (nc == 1) {
            if (!BorderSolver::solve_inplace(B, work + TDLS_STRIDED(o.Sbb, work_stride),
                                             work_stride, piv, piv_stride,
                                             work + TDLS_STRIDED(o.rb, work_stride), work_stride))
                return false;
        } else {
            if (!BorderSolver::solve_inplace_multirhs(
                    B, nc, work + TDLS_STRIDED(o.Sbb, work_stride), work_stride, piv, piv_stride,
                    work + TDLS_STRIDED(o.rb, work_stride), work_stride,
                    TDLS_STRIDED(B, work_stride)))
                return false;
        }
        for (int m = 0; m < nc; ++m)
            for (int r = 0; r < B; ++r) {
                if constexpr (canonical)
                    y[TDLS_STRIDED(r, rhs_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)] =
                        TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r);
                else
                    TDLS_ARROW_LUPP_DYN_YW(m, r) = TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r);
            }
        if constexpr (!canonical)
            recover<NC>(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride, xcol_stride,
                        nc);
        return true;
    }

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
    /// \tparam NC        number of right-hand sides when fixed, 0 otherwise
    /// \tparam canonical no right-hand side in the blocks, and no recovery
    ///                   data stored
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in]     nrhs        number of right-hand sides when NC is 0
    /// \param[in]     so          streamed: the offsets of the streamed areas
    /// \tparam recompute no recovery data stored, A and y untouched
    ///                   (solve_inplace_recomputed)
    /// \tparam Streamed  none on the stored path; streamed, StreamedOffsets:
    ///                   the block is in the workspace already (D_k, the
    ///                   columns [A_kb | y_k], A_bk), nothing is read from A
    /// \return true when the block is condensed
    template<int NC, bool canonical, bool recompute = false, typename... Streamed>
    TDLS_HOST_DEVICE static constexpr bool
    condense(const ArrowheadShape& shape, const Sizes& z, const Offsets& o, T* TDLS_RESTRICT A,
             const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
             T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
             [[maybe_unused]] const std::size_t xcol_stride, const int k,
             [[maybe_unused]] const int nrhs, const Streamed*... so) noexcept {
        constexpr bool streamed = sizeof...(Streamed) > 0;
        const int B = z.B, S = z.S;
        // Right-hand sides carried through the blocks.
        const int ny                         = canonical ? 0 : NC > 0 ? NC : nrhs;
        [[maybe_unused]] constexpr int ycols = NC;
        const T u                            = condensation_pivot_threshold;
        T* const Dl                          = work + TDLS_STRIDED(o.Dl, work_stride);
        T* const W                           = work + TDLS_STRIDED(o.W, work_stride);
        // The block, in the order of its storage (a streamed block is there
        // already).
        if constexpr (!streamed) {
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j)
                    TDLS_ARROW_LUPP_DYN_W(o.Dl + (col ? j * S + i : i * S + j)) =
                        TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, j));
        }
        // Factored by TiledLUpp, or by physical interchanges
        // (blocks in registers): Dl then holds U in logical row order and the
        // check of L is made.
        bool ok = true;
        if (z.reg) {
            const int verdict = physical_factor(z, o, work, work_stride, piv, piv_stride);
            if (verdict == 2) return false;
            ok = verdict == 0;
        } else {
            if (!BlockSolver::factorize(S, Dl, work_stride, piv, piv_stride)) return false;
        }
        if constexpr (diag && !streamed) {
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_W(o.acoef + i) =
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abk_diag(shape, k, i));
        }
        const bool forward_check = !z.reg && col && z.pass_width == B + ny;
        // The multipliers L, and U in logical row order unless the forward
        // stage of the solve gives the border multipliers.
        if (z.reg) {
            for (int l = 0; l < S; ++l)
                for (int j = l; j < S; ++j)
                    TDLS_ARROW_LUPP_DYN_W(o.U + l * S + j) =
                        TDLS_ARROW_LUPP_DYN_W(o.Dl + l * S + j);
        } else {
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
        }
        if (forward_check) {
            if constexpr (!streamed) {
                for (int i = 0; i < S; ++i) {
                    for (int c = 0; c < B; ++c)
                        TDLS_ARROW_LUPP_DYN_W(o.W + c * S + i) =
                            TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c));
                    for (int m = 0; m < ny; ++m)
                        TDLS_ARROW_LUPP_DYN_W(o.W + (B + m) * S + i) =
                            TDLS_ARROW_LUPP_DYN_YW(m, B + k * S + i);
                }
            }
            BlockSolver::forward_inplace_multirhs(S, B + ny, Dl, work_stride, piv, piv_stride, W,
                                                  work_stride, TDLS_STRIDED(S, work_stride));
            // Column c of U^-T A_kb is row c of the border multipliers.
            for (int e = 0; e < B * S; ++e)
                ok = ok && u * detail::abs(TDLS_ARROW_LUPP_DYN_W(o.W + e)) <= T(1);
            if (!ok) return false;
            BlockSolver::backward_inplace_multirhs(S, B + ny, Dl, work_stride, piv, piv_stride, W,
                                                   work_stride, TDLS_STRIDED(S, work_stride));
            condense_pass<NC, canonical, recompute>(shape, z, o, A, A_stride, piv, piv_stride, work,
                                                    work_stride, y, rhs_stride, xcol_stride, k, 0,
                                                    B + ny, true, so...);
        } else {
            ok = border_multipliers(shape, z, o, A, A_stride, work, work_stride, k, so...) && ok;
            if (!ok) return false;
            // U, recovery data: not for the canonical solve.
            if constexpr (multiplier_update && !canonical && !recompute) {
                for (int l = 0; l < S; ++l)
                    for (int j = l; j < S; ++j)
                        TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, l, j)) =
                            TDLS_ARROW_LUPP_DYN_W(o.U + l * S + j);
            }
            for (int c0 = 0; c0 < B + ny; c0 += z.pass_width) {
                const int w = B + ny - c0 < z.pass_width ? B + ny - c0 : z.pass_width;
                condense_pass<NC, canonical, recompute>(shape, z, o, A, A_stride, piv, piv_stride,
                                                        work, work_stride, y, rhs_stride,
                                                        xcol_stride, k, c0, w, false, so...);
            }
        }
        return true;
    }

    /// \brief condense on a streamed block, kept for the second pass when it
    /// is refused: D_k is copied before its factorization (and [A_kb | y_k]
    /// before the forward solve when the pivot check reads it), so that a
    /// refused block is stored as the provider gave it, where the stored
    /// path leaves a refused block, without asking for it again.
    /// \tparam NC        number of right-hand sides when fixed, 0 otherwise
    /// \tparam canonical no right-hand side in the blocks, and no recovery
    ///                   data stored
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     so          the offsets of the streamed areas
    /// \param[in,out] A           matrix in its storage: the recovery data
    ///                            of the block (none when canonical), or
    ///                            the block itself when refused
    /// \param[in]     A_stride    element stride of A
    /// \param[out]    piv         pivot array
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] work        workspace, the block in its areas
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in]     nrhs        number of right-hand sides when NC is 0
    /// \return true when the block is condensed, false when it is refused
    ///         (and stored as given)
    template<int NC, bool canonical>
    TDLS_HOST_DEVICE static constexpr bool condense_or_store_streamed(
        const ArrowheadShape& shape, const Sizes& z, const Offsets& o, const StreamedOffsets& so,
        T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
        T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
        [[maybe_unused]] const std::size_t xcol_stride, const int k, const int nrhs) noexcept {
        const int B = z.B, S = z.S;
        const int ny                         = canonical ? 0 : NC > 0 ? NC : nrhs;
        [[maybe_unused]] constexpr int ycols = NC;
        const bool forward_check             = !z.reg && col && z.pass_width == B + ny;
        for (int e = 0; e < S * S; ++e)
            TDLS_ARROW_LUPP_DYN_W(so.Dsave + e) = TDLS_ARROW_LUPP_DYN_W(o.Dl + e);
        if (forward_check)
            for (int e = 0; e < (B + ny) * S; ++e)
                TDLS_ARROW_LUPP_DYN_W(so.Wsave + e) = TDLS_ARROW_LUPP_DYN_W(o.W + e);
        if (condense<NC, canonical>(shape, z, o, A, A_stride, piv, piv_stride, work, work_stride, y,
                                    rhs_stride, xcol_stride, k, nrhs, &so))
            return true;
        // Refused: the block as given, where the stored path leaves a
        // refused block (W is intact unless the forward stage ran).
        const int ow = forward_check ? so.Wsave : o.W;
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j)
                TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, j)) =
                    TDLS_ARROW_LUPP_DYN_W(so.Dsave + (col ? j * S + i : i * S + j));
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c)) =
                    TDLS_ARROW_LUPP_DYN_W(ow + c * S + i);
            for (int m = 0; m < ny; ++m)
                TDLS_ARROW_LUPP_DYN_YW(m, B + k * S + i) = TDLS_ARROW_LUPP_DYN_W(ow + B * S + i);
        }
        if constexpr (diag) {
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_DYN_A(storage_map::abk_diag(shape, k, i)) =
                    TDLS_ARROW_LUPP_DYN_W(o.acoef + i);
        } else {
            for (int r = 0; r < B; ++r)
                for (int i = 0; i < S; ++i)
                    TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, i)) =
                        TDLS_ARROW_LUPP_DYN_W(so.Abk + r * S + i);
        }
        return false;
    }

    /// \brief Border multipliers M = Cb U^-1 of block k, checked against
    /// 1 / condensation_pivot_threshold; kept in the workspace (R x S)
    /// under the row-major layout with the dense coupling, where the Schur
    /// update reads them under multiplier_update.
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace (U, acoef in; Mb out)
    /// \param[in]     work_stride element stride of work
    /// \param[in]     k           block
    /// \param[in]     so          streamed: the offsets of the streamed areas
    /// \tparam Streamed none on the stored path; streamed, StreamedOffsets:
    ///                  A_kb and A_bk read from the workspace, not from A
    /// \return true when no multiplier exceeds the bound
    template<typename... Streamed>
    TDLS_HOST_DEVICE static constexpr bool
    border_multipliers(const ArrowheadShape& shape, const Sizes& z, const Offsets& o,
                       const T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                       const int work_stride, const int k, const Streamed*... so) noexcept {
        constexpr bool streamed = sizeof...(Streamed) > 0;
        const int B = z.B, S = z.S;
        bool ok = true;
        if constexpr (col) {
            for (int c = 0; c < B; ++c) {
                for (int i = 0; i < S; ++i) {
                    if constexpr (streamed)
                        TDLS_ARROW_LUPP_DYN_W(o.m + i) = TDLS_ARROW_LUPP_DYN_W(o.W + c * S + i);
                    else
                        TDLS_ARROW_LUPP_DYN_W(o.m + i) =
                            TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c));
                }
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
                for (int j = 0; j < S; ++j) {
                    if constexpr (streamed)
                        TDLS_ARROW_LUPP_DYN_W(o.Mb + r * S + j) =
                            TDLS_ARROW_LUPP_DYN_W(streamed_arg(so...)->Abk + r * S + j);
                    else
                        TDLS_ARROW_LUPP_DYN_W(o.Mb + r * S + j) =
                            TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, j));
                }
                ok = multipliers_pass(z, o, work, work_stride, o.Mb + r * S, 0) && ok;
            }
        }
        return ok;
    }

    /// \brief Factors the block at o.Dl in place in the workspace
    /// (RowInterchange::Physical), by TiledLUpp with physical
    /// row interchanges (PhysicalBlockSolver): the operations of
    /// ArrowheadLUppSolverStatic::register_factor in the same order, so
    /// that both solvers agree bit for bit. The in-tile pivot rule of
    /// TiledLUpp over a single tile, the rows interchanged physically and
    /// whole; on exit the block holds L below its diagonal and U above it,
    /// in logical row order, 1 / pivot on the diagonal, and piv[j] the row
    /// interchanged with row j at step j.
    /// \param[in]     z           sizes of the shape
    /// \param[in]     o           the offsets of the workspace (Dl)
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[out]    piv         the interchanges
    /// \param[in]     piv_stride  element stride of piv
    /// \return 0 when L passes the check, 1 when it fails it, 2 when a
    ///         column has no candidate above the singularity floor of
    ///         TiledLUpp
    TDLS_HOST_DEVICE static constexpr int
    physical_factor(const Sizes& z, const Offsets& o, T* TDLS_RESTRICT work, const int work_stride,
                    int* TDLS_RESTRICT piv, const int piv_stride) noexcept {
        return PhysicalBlockSolver::physical_factorize_checked(
            z.S, work + TDLS_STRIDED(o.Dl, work_stride), work_stride, piv, piv_stride,
            condensation_pivot_threshold);
    }

    /// \brief Solves the w columns at o.W in place with the factors of
    /// physical_factor, the operations of
    /// ArrowheadLUppSolverStatic::register_solve in the same order: W :=
    /// D_k^-1 W, or L^-1 P W alone under multiplier_update; under the
    /// column-major layout, U^-T, then L^-T,
    /// then the interchanges in reverse order.
    /// \param[in]     z           sizes of the shape
    /// \param[in]     o           the offsets of the workspace (Dl, W)
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in]     piv         the interchanges
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in]     w           number of columns
    /// \param[in]     backward    whether the row-major solve runs its
    ///                            backward stage (always, but for the forward
    ///                            stage alone of the condensation under
    ///                            multiplier_update)
    TDLS_HOST_DEVICE static constexpr void
    physical_solve(const Sizes& z, const Offsets& o, T* TDLS_RESTRICT work, const int work_stride,
                   const int* TDLS_RESTRICT piv, const int piv_stride, const int w,
                   const bool backward = !multiplier_update) noexcept {
        const int S              = z.S;
        const T* TDLS_RESTRICT F = work + TDLS_STRIDED(o.Dl, work_stride);
        T* TDLS_RESTRICT W       = work + TDLS_STRIDED(o.W, work_stride);
        const std::size_t cs     = TDLS_STRIDED(S, work_stride);
        PhysicalBlockSolver::physical_forward(S, w, F, work_stride, piv, piv_stride, W, work_stride,
                                              cs);
        if (col || backward)
            PhysicalBlockSolver::physical_backward(S, w, F, work_stride, piv, piv_stride, W,
                                                   work_stride, cs);
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
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     r           accumulated row
    /// \param[in]     c           border column, or B + m for right-hand side m
    /// \param[in]     v           value subtracted
    template<int NC>
    TDLS_HOST_DEVICE static constexpr void
    schur_sub([[maybe_unused]] const ArrowheadShape& shape, const Sizes& z,
              [[maybe_unused]] const Offsets& o, [[maybe_unused]] T* TDLS_RESTRICT A,
              [[maybe_unused]] const int A_stride, [[maybe_unused]] T* TDLS_RESTRICT work,
              [[maybe_unused]] const int work_stride, [[maybe_unused]] T* TDLS_RESTRICT y,
              [[maybe_unused]] const int rhs_stride, [[maybe_unused]] const std::size_t xcol_stride,
              const int r, const int c, const T v) noexcept {
        [[maybe_unused]] constexpr int ycols = NC;
        if constexpr (in_place) {
            if (c < z.B)
                TDLS_ARROW_LUPP_DYN_A(storage_map::abb(shape, z.R0 + r, c)) -= v;
            else
                TDLS_ARROW_LUPP_DYN_YW(c - z.B, z.R0 + r) -= v;
        } else {
            if (c < z.B)
                TDLS_ARROW_LUPP_DYN_W(o.Sacc + r * z.B + c) -= v;
            else
                TDLS_ARROW_LUPP_DYN_W(o.racc + (NC == 1 ? 0 : (c - z.B) * z.R) + r) -= v;
        }
    }

    /// \brief Columns c0 .. c0 + w of [A_kb | y_k]: solved (unless solved
    /// holds), stored in place (not for the canonical solve, which recovers
    /// nothing), folded into the accumulated rows. The arithmetic of
    /// ArrowheadLUppSolverStatic::condense_pass.
    /// \tparam NC        number of right-hand sides when fixed, 0 otherwise
    /// \tparam canonical no right-hand side in the blocks, and nothing
    ///                   stored
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         pivots of the block
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in]     c0          first column of the pass
    /// \param[in]     w           columns of the pass
    /// \param[in]     solved      whether the workspace holds the solved
    ///                            columns already
    /// \param[in]     so          streamed: the offsets of the streamed areas
    /// \tparam recompute no recovery data stored (solve_inplace_recomputed)
    /// \tparam Streamed none on the stored path; streamed, StreamedOffsets:
    ///                  the columns are in W already, and A_bk in its area
    template<int NC, bool canonical, bool recompute = false, typename... Streamed>
    TDLS_HOST_DEVICE static constexpr void
    condense_pass(const ArrowheadShape& shape, const Sizes& z, const Offsets& o, T* TDLS_RESTRICT A,
                  const int A_stride, const int* TDLS_RESTRICT piv, const int piv_stride,
                  T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                  const int rhs_stride, [[maybe_unused]] const std::size_t xcol_stride, const int k,
                  const int c0, const int w, const bool solved, const Streamed*... so) noexcept {
        constexpr bool streamed              = sizeof...(Streamed) > 0;
        [[maybe_unused]] constexpr int ycols = NC;
        const int B = z.B, S = z.S, R = z.R;
        const T* const Dl = work + TDLS_STRIDED(o.Dl, work_stride);
        T* const W        = work + TDLS_STRIDED(o.W, work_stride);
        if (!solved) {
            if constexpr (streamed) {
                // Columns c0 .. c0 + w of the streamed block move to the
                // front of W: the columns before c0 are solved and stored
                // already, and c0 >= w when c0 > 0, so the two ranges do
                // not overlap.
                if (c0 > 0)
                    for (int i = 0; i < S; ++i)
                        for (int j = 0; j < w; ++j)
                            TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i) =
                                TDLS_ARROW_LUPP_DYN_W(o.W + (c0 + j) * S + i);
            } else {
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < w; ++j)
                        TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i) =
                            c0 + j < B
                                ? TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c0 + j))
                                : TDLS_ARROW_LUPP_DYN_YW(c0 + j - B, B + k * S + i);
            }
            if (z.reg) {
                physical_solve(z, o, work, work_stride, piv, piv_stride, w);
            } else if constexpr (multiplier_update) {
                BlockSolver::forward_inplace_multirhs(S, w, Dl, work_stride, piv, piv_stride, W,
                                                      work_stride, TDLS_STRIDED(S, work_stride));
            } else if constexpr (Config.gather_rhs) {
                T* const X = work + TDLS_STRIDED(o.X, work_stride);
                BlockSolver::substitute_multirhs(S, w, Dl, work_stride, piv, piv_stride, W, X,
                                                 work_stride, TDLS_STRIDED(S, work_stride));
                for (int e = 0; e < w * S; ++e)
                    TDLS_ARROW_LUPP_DYN_W(o.W + e) = TDLS_ARROW_LUPP_DYN_W(o.X + e);
            } else {
                BlockSolver::substitute_inplace_multirhs(S, w, Dl, work_stride, piv, piv_stride, W,
                                                         work_stride, TDLS_STRIDED(S, work_stride));
            }
        }
        if constexpr (!canonical && !recompute) {
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < w; ++j) {
                    if (c0 + j < B)
                        TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c0 + j)) =
                            TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i);
                    else
                        TDLS_ARROW_LUPP_DYN_YW(c0 + j - B, B + k * S + i) =
                            TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i);
                }
        }
        if constexpr (multiplier_update) {
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < w; ++j) {
                    T sum = T(0);
                    for (int l = 0; l < S; ++l)
                        sum += TDLS_ARROW_LUPP_DYN_W(o.Mb + r * S + l) *
                               TDLS_ARROW_LUPP_DYN_W(o.W + j * S + l);
                    schur_sub<NC>(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride,
                                  xcol_stride, r, c0 + j, sum);
                }
        } else if constexpr (diag) {
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < w; ++j)
                    schur_sub<NC>(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride,
                                  xcol_stride, i, c0 + j,
                                  TDLS_ARROW_LUPP_DYN_W(o.acoef + i) *
                                      TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i));
        } else {
            for (int r = 0; r < B; ++r) {
                for (int i = 0; i < S; ++i) {
                    if constexpr (streamed)
                        TDLS_ARROW_LUPP_DYN_W(o.a + i) =
                            TDLS_ARROW_LUPP_DYN_W(streamed_arg(so...)->Abk + r * S + i);
                    else
                        TDLS_ARROW_LUPP_DYN_W(o.a + i) =
                            TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, i));
                }
                for (int j = 0; j < w; ++j) {
                    T sum = T(0);
                    for (int i = 0; i < S; ++i)
                        sum +=
                            TDLS_ARROW_LUPP_DYN_W(o.a + i) * TDLS_ARROW_LUPP_DYN_W(o.W + j * S + i);
                    schur_sub<NC>(shape, z, o, A, A_stride, work, work_stride, y, rhs_stride,
                                  xcol_stride, r, c0 + j, sum);
                }
            }
        }
    }

    /// \brief Recovers the blocks from the border solutions, the last block
    /// first. The arithmetic of ArrowheadLUppSolverStatic::recover.
    /// \tparam NC number of right-hand sides when fixed, 0 otherwise
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in]     A           matrix holding the recovery data
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace (rb, refused flags, H_k)
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           the recovery right-hand sides on entry,
    ///                            the solutions on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     nrhs        number of right-hand sides when NC is 0
    template<int NC>
    TDLS_HOST_DEVICE static constexpr void
    recover(const ArrowheadShape& shape, const Sizes& z, const Offsets& o, const T* TDLS_RESTRICT A,
            const int A_stride, T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
            const int rhs_stride, [[maybe_unused]] const std::size_t xcol_stride,
            [[maybe_unused]] const int nrhs) noexcept {
        const int B = z.B, S = z.S, NB = z.NB, R = z.R;
        const int nc                         = NC > 0 ? NC : nrhs;
        [[maybe_unused]] constexpr int ycols = NC;
        // w and s: column m at m * R and m * S.
        for (int e = 0; e < R * nc; ++e)
            TDLS_ARROW_LUPP_DYN_W(o.w + e) = T(0);
        for (int k = NB - 1; k >= 0; --k) {
            for (int i = 0; i < S; ++i) {
                for (int m = 0; m < nc; ++m)
                    TDLS_ARROW_LUPP_DYN_W(o.s + m * S + i) =
                        TDLS_ARROW_LUPP_DYN_YW(m, B + k * S + i);
                for (int c = 0; c < B; ++c) {
                    const T a = TDLS_ARROW_LUPP_DYN_A(storage_map::akb(shape, k, i, c));
                    for (int m = 0; m < nc; ++m)
                        TDLS_ARROW_LUPP_DYN_W(o.s + m * S + i) -=
                            a * TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + c);
                }
            }
            if (TDLS_ARROW_LUPP_DYN_W(o.refused + k) != T(0)) {
                // x_k = c_k - G_k xb - H_k w, then w += A_bk x_k.
                for (int i = 0; i < S; ++i)
                    for (int a = 0; a < R; ++a) {
                        const T h = a < S ? TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, i, a))
                                          : TDLS_ARROW_LUPP_DYN_W(
                                                o.extra + (k * S + i) * z.extra_cols + a - S);
                        for (int m = 0; m < nc; ++m)
                            TDLS_ARROW_LUPP_DYN_W(o.s + m * S + i) -=
                                h * TDLS_ARROW_LUPP_DYN_W(o.w + m * R + a);
                    }
                if constexpr (diag) {
                    for (int i = 0; i < S; ++i) {
                        const T coef = TDLS_ARROW_LUPP_DYN_A(storage_map::abk_diag(shape, k, i));
                        for (int m = 0; m < nc; ++m)
                            TDLS_ARROW_LUPP_DYN_W(o.w + m * R + i) +=
                                coef * TDLS_ARROW_LUPP_DYN_W(o.s + m * S + i);
                    }
                } else {
                    for (int r = 0; r < B; ++r)
                        for (int i = 0; i < S; ++i) {
                            const T coef = TDLS_ARROW_LUPP_DYN_A(storage_map::abk(shape, k, r, i));
                            for (int m = 0; m < nc; ++m)
                                TDLS_ARROW_LUPP_DYN_W(o.w + m * R + r) +=
                                    coef * TDLS_ARROW_LUPP_DYN_W(o.s + m * S + i);
                        }
                }
            } else if constexpr (multiplier_update) {
                // x_k = U^-1 s, U in the upper triangle of D_k.
                for (int l = S - 1; l >= 0; --l) {
                    for (int j = l + 1; j < S; ++j) {
                        const T ulj = TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, l, j));
                        for (int m = 0; m < nc; ++m)
                            TDLS_ARROW_LUPP_DYN_W(o.s + m * S + l) -=
                                ulj * TDLS_ARROW_LUPP_DYN_W(o.s + m * S + j);
                    }
                    const T ull = TDLS_ARROW_LUPP_DYN_A(storage_map::d(shape, k, l, l));
                    for (int m = 0; m < nc; ++m)
                        TDLS_ARROW_LUPP_DYN_W(o.s + m * S + l) *= ull;
                }
            }
            for (int i = 0; i < S; ++i)
                for (int m = 0; m < nc; ++m)
                    TDLS_ARROW_LUPP_DYN_YW(m, B + k * S + i) =
                        TDLS_ARROW_LUPP_DYN_W(o.s + m * S + i);
        }
    }

    /// \brief Second pass: eliminates every refused block together with the
    /// border rows, in the front at the start of the workspace, the
    /// right-hand sides in its last columns. The arithmetic of
    /// ArrowheadLUppSolverStatic::second_pass.
    /// \tparam NC        number of right-hand sides when fixed, 0 otherwise
    /// \tparam canonical right-hand sides zero on the rows of the blocks,
    ///                   and no recovery data
    /// \param[in]     shape       shape of the matrix
    /// \param[in]     z           its sizes
    /// \param[in]     o           the offsets of the workspace
    /// \param[in,out] A           matrix in its storage
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] work        workspace (front, H_k, Sbb, rb, flags)
    /// \param[in]     work_stride element stride of work
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     nrhs        number of right-hand sides when NC is 0
    /// \return false when a column of a refused block has no nonzero
    ///         candidate: the matrix is singular
    template<int NC, bool canonical>
    TDLS_HOST_DEVICE TDLS_NO_INLINE static constexpr bool
    second_pass(const ArrowheadShape& shape, const Sizes& z, const Offsets& o, T* TDLS_RESTRICT A,
                const int A_stride, T* TDLS_RESTRICT work, const int work_stride,
                T* TDLS_RESTRICT y, const int rhs_stride,
                [[maybe_unused]] const std::size_t xcol_stride,
                [[maybe_unused]] const int nrhs) noexcept {
        const int B = z.B, S = z.S, NB = z.NB, R = z.R, R0 = z.R0;
        const int front_rows = z.front_rows, front_cols = z.front_cols;
        const int nc                         = NC > 0 ? NC : nrhs;
        const int rhs0                       = S + B + R;
        [[maybe_unused]] constexpr int ycols = NC;
        const T u                            = condensation_pivot_threshold;
        // The singularity floor of the block solver: a smaller pivot, zero
        // or subnormal, is singular there, and its reciprocal could overflow.
        constexpr T singular_floor = BlockSolver::singular_floor;
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_F(S + r, S + c) = TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c);
            for (int a = 0; a < R; ++a)
                TDLS_ARROW_LUPP_DYN_F(S + r, S + B + a) = r == R0 + a ? T(1) : T(0);
            for (int m = 0; m < nc; ++m)
                TDLS_ARROW_LUPP_DYN_F(S + r, rhs0 + m) = TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r);
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
                for (int m = 0; m < nc; ++m) {
                    if constexpr (canonical)
                        TDLS_ARROW_LUPP_DYN_F(i, rhs0 + m) = T(0);
                    else
                        TDLS_ARROW_LUPP_DYN_F(i, rhs0 + m) =
                            TDLS_ARROW_LUPP_DYN_YW(m, B + k * S + i);
                }
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
                if (!(vmax >= singular_floor)) return false;
                const int p = pblk >= 0 && vblk >= u * vmax && vblk >= singular_floor ? pblk : pmax;
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
            if constexpr (!canonical) {
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
                    for (int m = 0; m < nc; ++m)
                        TDLS_ARROW_LUPP_DYN_YW(m, B + k * S + i) =
                            TDLS_ARROW_LUPP_DYN_F(i, rhs0 + m);
                }
            }
        }
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                TDLS_ARROW_LUPP_DYN_W(o.Sbb + r * B + c) = TDLS_ARROW_LUPP_DYN_F(S + r, S + c);
            for (int m = 0; m < nc; ++m)
                TDLS_ARROW_LUPP_DYN_W(o.rb + m * B + r) = TDLS_ARROW_LUPP_DYN_F(S + r, rhs0 + m);
        }
        return true;
    }
};



#undef TDLS_ARROW_LUPP_DYN_A
#undef TDLS_ARROW_LUPP_DYN_Y
#undef TDLS_ARROW_LUPP_DYN_YW
#undef TDLS_ARROW_LUPP_DYN_W
#undef TDLS_ARROW_LUPP_DYN_F
#undef TDLS_ARROW_LUPP_DYN_LU
#undef TDLS_ARROW_LUPP_DYN_RC



} // namespace tdls



#endif // TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_DYNAMIC_HPP
