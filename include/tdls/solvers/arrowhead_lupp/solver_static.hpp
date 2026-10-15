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
///   2. The pivots are checked on the factors: no multiplier of the
///      elimination of the block exceeds 1 / condensation_pivot_threshold,
///      for its own rows (L) and for the border rows that share its
///      columns (M = Cb U^-1). A block that fails, singular included, is
///      refused and left untouched for the second pass.
///   3. Otherwise the block is condensed: its contribution goes to the
///      Schur complement of the border, its recovery data in place of A_kb
///      and y_k (see below; none for the canonical solves).
/// Second pass, only when blocks were refused, in the workspace: each
/// refused block is eliminated together with the border rows, by partial
/// pivoting over its S rows and the B border rows. A border row may serve
/// as pivot; a row of the block then takes its place in the border. A
/// column whose best candidate falls below BlockSolver::singular_floor,
/// zero or subnormal, makes the matrix singular. The border keeps B rows
/// whatever the number of refused blocks. Its rows
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
/// with schur_in_place, the coupled border rows of A hold the Schur
/// complement of the accepted blocks. The other entries are only read.
/// The canonical solves recover nothing and store no recovery data: they
/// write in A only the Schur complement under schur_in_place and,
/// streamed, the refused blocks. Without schur_in_place, a stored A thus
/// comes out unchanged and serves the next columns.
///
/// The caller pre-offsets every remote pointer with the system index and
/// passes a runtime stride, as for the other families; the
/// `internal_rhs` / `internal_work` / `internal_matrix` template booleans
/// declare a caller-local array instead (stride ignored). The workspace
/// holds the front of the second pass: it is touched only when blocks are
/// refused.
///
/// Streamed blocks. The *_streamed entry points take the blocks from a
/// provider, called exactly once per block and in order when the solver
/// condenses it, instead of reading them from A: the caller stores only
/// the border, and the provider writes every entry of the block, zeros
/// included, in the working arrays of the solver (StreamedBlock). A
/// refused block is stored as given for the second pass. The arithmetic
/// is that of the stored path, bit for bit when the compiler contracts
/// multiply-adds alike in the two paths (-ffp-contract=off, or
/// --fmad=false for nvcc, guarantees it; GCC with -march=native does
/// not). The streamed path is selected at compile time: the stored entry
/// points keep their signatures and their code. The entry points are
/// noexcept: the provider must not throw.



#include <cmath>
#include <type_traits>
#include <utility>

#include <tdls/core/macros.hpp>
#include <tdls/core/math.hpp>
#include <tdls/core/target.hpp>
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
#if defined(TDLS_DETAIL_COMPILER_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
#if defined(TDLS_DETAIL_COMPILER_NVCC)
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
                         (ycols == 1 ? std::size_t(0) : TDLS_STRIDED_WIDE(m, xcol_stride))]
/// \def TDLS_ARROW_LUPP_W
/// \brief Entry e of the workspace: contiguous under internal residency,
/// strided otherwise.
#define TDLS_ARROW_LUPP_W(e) work[internal_work ? unsigned(e) : TDLS_STRIDED(e, work_stride)]
/// \def TDLS_ARROW_LUPP_F
/// \brief Entry (r, c) of the front of the second pass, row-major at the
/// start of the workspace, fcols columns per row.
#define TDLS_ARROW_LUPP_F(r, c) TDLS_ARROW_LUPP_W((r) * fcols + (c))
/// \def TDLS_ARROW_LUPP_SIDED
/// \brief Under the side dispatch of nvc++ (TDLS_TARGET_GPU), when the host
/// and the device choose differently (sided), returns the call, the
/// variadic argument, run by the solver of the side of the code: SolverGpu
/// on the device, SolverCpu on the host; nothing elsewhere and in constant
/// expressions.
#if defined(TDLS_DETAIL_SIDE_DISPATCH)
#define TDLS_ARROW_LUPP_SIDED(...)                                                                 \
    if constexpr (sided) {                                                                         \
        if (!std::is_constant_evaluated()) {                                                       \
            if (detail::on_device_side()) return SolverGpu::__VA_ARGS__;                           \
            return SolverCpu::__VA_ARGS__;                                                         \
        }                                                                                          \
    }
#else
#define TDLS_ARROW_LUPP_SIDED(...)
#endif
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



namespace detail {

/// \brief Block k of a streamed solve of ArrowheadLUppSolverStatic
/// (ArrowheadLUppSolverStatic::StreamedBlock): the working arrays of the
/// solver themselves, which the caller fills through the accessors when
/// the solver asks for the block; the factorization and the solve then
/// run in place. Only a save is copied (D_k, and [A_kb | y_k] when the
/// column-major check reads the forward stage), to store a refused
/// block as given. Entries in logical order whatever the
/// layout and the storage. The arrays come uninitialized: the caller
/// writes every entry, zeros included, by assignment.
/// \tparam T    scalar type
/// \tparam S    unknowns of a block
/// \tparam B    border unknowns
/// \tparam col  column-major layout
/// \tparam diag diagonal coupling
template<typename T, int S, int B, bool col, bool diag>
struct ArrowheadStreamedBlock {
    T Dl[S * S];             ///< D_k in the layout of the factorization
    T W[(B + 1) * S];        ///< [A_kb | y_k], column c at c * S
    T Abk[diag ? S : B * S]; ///< A_bk: coupled rows (diagonal) or B x S (dense)
    /// \param[in] i row of the block
    /// \param[in] j column of the block
    /// \return entry (i, j) of D_k
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& d(const int i, const int j) noexcept {
        return Dl[col ? j * S + i : i * S + j];
    }
    /// \param[in] i row of the block
    /// \param[in] c border column
    /// \return entry (i, c) of A_kb
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& akb(const int i, const int c) noexcept {
        return W[c * S + i];
    }
    /// \param[in] i row of the block
    /// \return entry i of y_k, the right-hand side on the rows of the
    ///         block
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& y(const int i) noexcept {
        return W[B * S + i];
    }
    /// \param[in] i row and column of the block
    /// \return the coupling A_bk(coupling_row + i, i) (diagonal coupling)
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& coupling(const int i) noexcept {
        return Abk[i];
    }
    /// \param[in] r border row
    /// \param[in] i column of the block
    /// \return entry (r, i) of A_bk (dense coupling)
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr T& abk(const int r, const int i) noexcept {
        return Abk[r * S + i];
    }
};


/// \brief Block k of a recomputed solve (solve_inplace_recomputed,
/// solve_inplace_recomputed_or_tangent): a StreamedBlock, the same
/// accessors, and whether the solver asks for the block again, in its
/// recovery. A provider taking a StreamedBlock& serves too; one taking a
/// RecomputedBlock&, or auto&, reads the flag. The streamed entry points
/// keep the StreamedBlock alone, so that their code is that of a solver
/// without the recomputed ones.
/// \tparam T    scalar type
/// \tparam S    unknowns of a block
/// \tparam B    border unknowns
/// \tparam col  column-major layout
/// \tparam diag diagonal coupling
template<typename T, int S, int B, bool col, bool diag>
struct ArrowheadRecomputedBlock : ArrowheadStreamedBlock<T, S, B, col, diag> {
    /// Set when the solver asks for the block again, in its recovery: the
    /// provider may then skip what it accumulates once per pass, a
    /// residual for instance. False while it condenses the block.
    bool recovery = false;
};

/// \brief Outcome of ArrowheadLUppSolverStatic::solve_inplace_streamed_or_tangent.
enum class ArrowheadStreamedOutcome {
    solved,  ///< y holds the solution (converged() returned false)
    tangent, ///< converged: x holds the canonical border columns, y
             ///< intermediate values
    singular ///< the matrix is singular
};

/// \brief Outcome of ArrowheadLUppSolverStatic::solve_inplace_recomputed and
/// solve_inplace_recomputed_or_tangent.
enum class ArrowheadRecomputedOutcome {
    solved,   ///< y holds the solution
    tangent,  ///< converged (solve_inplace_recomputed_or_tangent): x holds
              ///< the tangent columns, y is untouched
    singular, ///< the matrix is singular; y is untouched
    deferred  ///< more blocks refused than the capacity of the store; y is
              ///< untouched, for a solve that stores them
              ///< (solve_inplace_streamed)
};

} // namespace detail



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
    /// Blocks eliminated in registers (ArrowheadLUppConfig::block_row_interchange,
    /// Auto resolved for this block size, precision and compilation pass).
    static constexpr bool reg = detail::arrowhead_register_blocks(Config, S);
    /// Width of the passes over [A_kb | y_k], -1 resolved
    /// (ArrowheadLUppConfig::rhs_pass_width).
    static constexpr int rhs_pass = detail::arrowhead_rhs_pass_width(Config, reg);
#if defined(TDLS_DETAIL_SIDE_DISPATCH)
    /// The configuration with the choices of the GPU made explicit (the
    /// side dispatch of nvc++, TDLS_TARGET_GPU).
    static constexpr ArrowheadLUppConfig<T> config_gpu =
        detail::arrowhead_resolved_config<T, true>(Config, S);
    /// The configuration with the choices of the CPU made explicit.
    static constexpr ArrowheadLUppConfig<T> config_cpu =
        detail::arrowhead_resolved_config<T, false>(Config, S);
    /// Whether the two sides choose differently: the solves then run the
    /// solver of the explicit configuration of their side.
    static constexpr bool sided =
        config_gpu.block_row_interchange != config_cpu.block_row_interchange ||
        config_gpu.rhs_pass_width != config_cpu.rhs_pass_width;
    /// The solver of the device side.
    using SolverGpu = ArrowheadLUppSolverStatic<T, Shape, config_gpu>;
    /// The solver of the host side.
    using SolverCpu = ArrowheadLUppSolverStatic<T, Shape, config_cpu>;
#endif
    /// Every instantiation reaches the cores of the others (the side
    /// dispatch of nvc++).
    template<typename U, ArrowheadShape OtherShape, ArrowheadLUppConfig<U> OtherConfig>
    friend struct ArrowheadLUppSolverStatic;
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
    static constexpr int pass_width = rhs_pass == 0 ? B + 1 : rhs_pass;

  private:
    /// Columns [A_kb | Y_k] solved per pass with ny right-hand sides in
    /// the blocks (pass_width for ny = 1).
    template<int ny>
    static constexpr int pass_width_for = rhs_pass == 0 || rhs_pass > B + ny ? B + ny : rhs_pass;
    /// Column-major and a single pass: [A_kb | Y_k] is loaded once, and the
    /// pivot check reads its border part from the forward stage (not with
    /// the blocks in registers, whose check computes the border multipliers
    /// apart).
    template<int ny>
    static constexpr bool forward_check_for = !reg && col && pass_width_for<ny> == B + ny;
    /// Whether the condensation stores the recovery data of the accepted
    /// blocks (in place of A_kb and y_k, and U in D_k under the row-major
    /// layout with the dense coupling): not without right-hand side in the
    /// blocks (ny = 0), the canonical solve, which recovers nothing.
    template<int ny>
    static constexpr bool stores_recovery_for = ny > 0;
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
    static_assert(Config.rhs_pass_width >= -1 && Config.rhs_pass_width <= Shape.border + 1,
                  "ArrowheadLUppSolverStatic: rhs_pass_width must be in [-1, border + 1]");

    using BlockSolver  = TiledLUppSolverStatic<T, S, detail::arrowhead_block_config(Config)>;
    using BorderSolver = TiledLUppSolverStatic<T, B, detail::arrowhead_border_config(Config)>;
    /// Solver of the diagonal blocks in registers: TiledLUpp with
    /// physical row interchanges.
    using PhysicalBlockSolver =
        TiledLUppSolverStatic<T, S, detail::arrowhead_physical_block_config(Config)>;

    // The recomputed solve runs the second pass of the solver of its store
    // of refused blocks, another instantiation.
    template<typename U, ArrowheadShape OtherShape, ArrowheadLUppConfig<U> OtherConfig>
    friend struct ArrowheadLUppSolverStatic;
    /// Configuration of the store of the refused blocks of a recomputed
    /// solve: Packed, the Schur complement apart, the other knobs as Config.
    static constexpr ArrowheadLUppConfig<T> store_config = [] {
        ArrowheadLUppConfig<T> c = Config;
        c.storage                = ArrowheadStorage::Packed;
        c.schur_in_place         = false;
        c.streamed_out_of_line   = false;
        return c;
    }();
    /// \brief The shape of the store of capacity refused blocks, by value
    /// (see shape_value).
    /// \tparam capacity number of blocks the store holds (at least 1)
    /// \return {B, S, capacity, R0}
    template<int capacity>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr ArrowheadShape
    store_shape() noexcept {
        return {B, S, capacity, R0};
    }
    /// \brief Entries of the workspace of a recomputed solve with a store of
    /// capacity refused blocks and nrhs tangent columns (see
    /// recomputed_workspace_size): the store is instantiated only for a
    /// capacity of at least 1.
    /// \tparam capacity number of refused blocks the workspace holds
    /// \tparam nrhs     columns of the front of the second pass
    /// \return the number of entries, 0 for a capacity of 0
    template<int capacity, int nrhs>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr int recomputed_size() noexcept {
        if constexpr (capacity == 0) {
            return 0;
        } else {
            using Store =
                ArrowheadLUppSolverStatic<T, ArrowheadShape{B, S, capacity, R0}, store_config>;
            return Store::template multirhs_workspace_size<nrhs> + Store::storage_size +
                   Store::dimension;
        }
    }
    /// The solver of the store of capacity refused blocks: a matrix of the
    /// same border and block, with capacity blocks, in the Packed storage.
    template<int capacity>
    using RefusedStore =
        ArrowheadLUppSolverStatic<T, ArrowheadShape{B, S, capacity, R0}, store_config>;

  public:
    /// \brief Block k of a streamed solve (detail::ArrowheadStreamedBlock):
    /// the working arrays of the solver, filled by the provider through the
    /// accessors. The type depends on T, the sizes, the layout and the
    /// coupling only, so that the solvers of two configurations that differ
    /// otherwise hand their providers the same type.
    using StreamedBlock = detail::ArrowheadStreamedBlock<T, S, B, col, diag>;
    /// \brief Block k of a recomputed solve (detail::ArrowheadRecomputedBlock):
    /// a StreamedBlock and the recovery flag.
    using RecomputedBlock = detail::ArrowheadRecomputedBlock<T, S, B, col, diag>;

    /// \brief Solves A x = y in place, the blocks streamed: provider(k, blk)
    /// fills block k (StreamedBlock) when the solver needs it, k = 0, 1, ...
    /// in order, and the solver condenses it at once. The caller writes
    /// beforehand only the border: A_bb in A and the border rows of y. An
    /// accepted block leaves its recovery data in its storage; a refused one
    /// is stored as given for the second pass (the provider is called
    /// exactly once per block): the arithmetic, the pivot check and the
    /// second pass are those of solve_inplace, bit for bit under the same
    /// contraction of multiply-adds (see the file documentation).
    /// \tparam internal_rhs    y is a caller-local array (stride ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \tparam Provider        callable (int k, StreamedBlock& blk)
    /// \param[in]     provider      the source of the blocks
    /// \param[in,out] A             storage: A_bb on entry; recovery data on exit
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of workspace_size entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the solution on exit
    /// \param[in]     rhs_stride    element stride of y
    /// \param[out]    refused_count number of refused blocks
    /// \return false when the matrix is singular
    template<bool internal_rhs, bool internal_work, bool internal_matrix, typename Provider>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    solve_inplace_streamed(Provider&& provider, T* TDLS_RESTRICT A, const int A_stride,
                           T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                           const int rhs_stride, int& refused_count) noexcept {
        if constexpr (Config.streamed_out_of_line)
            return solve_core_out_of_line<1, false, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, 0, 0, refused_count, &provider);
        else
            return solve_core<1, false, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, 0, 0, refused_count, &provider);
    }

    /// \brief Outcome of solve_inplace_streamed_or_tangent
    /// (detail::ArrowheadStreamedOutcome).
    using StreamedOutcome = detail::ArrowheadStreamedOutcome;

    /// \brief One streamed pass of a Newton iteration: the blocks condensed
    /// as solve_inplace_streamed does, then converged() is called (the
    /// provider may compute the residual meanwhile). When it returns false,
    /// the pass ends as solve_inplace_streamed (y holds the correction).
    /// When it returns true, the correction is not computed: the border
    /// system condensed by the pass gives instead columns col0 .. col0 +
    /// ntan - 1 of the border block of A^-1 in x (the consistent tangent of a
    /// behaviour loaded through the border), refused blocks included (work
    /// then holds multirhs_workspace_size<ntan> entries). The provider is
    /// called exactly once per block, in order, and converged() once, after
    /// the last block. Bit for bit the results of solve_inplace_streamed,
    /// and those of solve_border_canonical_multirhs_streamed under the
    /// default rhs_pass_width (within rounding errors under another, see
    /// ArrowheadLUppConfig::rhs_pass_width).
    /// \tparam ntan            number of tangent columns, in [1, border]
    /// \tparam internal_rhs    y and x are caller-local arrays, column m of x
    ///                         from m * border (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \tparam Provider        callable (int k, StreamedBlock& blk)
    /// \tparam Converged       callable () returning bool
    /// \tparam outlined        internal: set by the out-of-line call of
    ///                         ArrowheadLUppConfig::streamed_out_of_line, never
    ///                         by the caller
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     converged     called after the last block: whether the
    ///                              tangent replaces the correction
    /// \param[in,out] A             storage: A_bb on entry; overwritten
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of multirhs_workspace_size<ntan>
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
    /// \return solved, tangent, or singular when the matrix is singular
    template<int ntan, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider, typename Converged, bool outlined = false>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr StreamedOutcome
    solve_inplace_streamed_or_tangent(Provider&& provider, Converged&& converged,
                                      T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                                      const int work_stride, T* TDLS_RESTRICT y,
                                      const int rhs_stride, const int col0, T* TDLS_RESTRICT x,
                                      const int x_stride, const std::size_t xcol_stride,
                                      int& refused_count) noexcept {
        static_assert(ntan >= 1 && ntan <= Shape.border,
                      "ArrowheadLUppSolverStatic: ntan must be in [1, border]");
        TDLS_ARROW_LUPP_SIDED(
            template solve_inplace_streamed_or_tangent<
                ntan, internal_rhs, internal_work, internal_matrix, Provider, Converged, outlined>(
                std::forward<Provider>(provider), std::forward<Converged>(converged), A, A_stride,
                work, work_stride, y, rhs_stride, col0, x, x_stride, xcol_stride, refused_count))
        if constexpr (Config.streamed_out_of_line && !outlined)
            return streamed_or_tangent_out_of_line<ntan, internal_rhs, internal_work,
                                                   internal_matrix>(
                provider, converged, A, A_stride, work, work_stride, y, rhs_stride, col0, x,
                x_stride, xcol_stride, refused_count);
        constexpr int ny                     = 1;
        [[maybe_unused]] constexpr int ycols = 1;
        T Sacc[in_place ? 1 : R * B];
        T racc[in_place ? 1 : R];
        if constexpr (in_place) {
            Sacc[0] = T(0);
            racc[0] = T(0);
        } else {
            TDLS_ARROW_LUPP_FOR(
                r, 0, R,
                TDLS_ARROW_LUPP_FOR(c, 0, B,
                                    Sacc[r * B + c] = TDLS_ARROW_LUPP_A(
                                        storage_map::abb(shape_value(), R0 + r, c));) racc[r] =
                    TDLS_ARROW_LUPP_YW(0, R0 + r);)
        }
        constexpr int words = (NB + 31) / 32;
        unsigned refused[words];
        TDLS_ARROW_LUPP_FOR(e, 0, words, refused[e] = 0u;)
        refused_count = 0;
        for (int k = 0; k < NB; ++k) {
            StreamedBlock blk;
            provider(k, blk);
            if (!condense_or_store_streamed<ny, internal_rhs, internal_matrix>(
                    A, A_stride, y, rhs_stride, 0, k, Sacc, racc, blk)) {
                refused[k >> 5] |= 1u << (k & 31);
                ++refused_count;
            }
        }
        T Sbb[B * B];
        int pb[B];
        // The rows as stored and the accumulated rows in separate loops, as
        // in solve_core.
        constexpr int acc0 = in_place ? B : R0;
        constexpr int acc1 = in_place ? B : R0 + R;
        TDLS_ARROW_LUPP_FOR(r, 0, acc0,
                            TDLS_ARROW_LUPP_FOR(c, 0, B,
                                                Sbb[r * B + c] = TDLS_ARROW_LUPP_A(
                                                    storage_map::abb(shape_value(), r, c));))
        TDLS_ARROW_LUPP_FOR(r, acc0, acc1,
                            TDLS_ARROW_LUPP_FOR(c, 0, B, Sbb[r * B + c] = Sacc[(r - R0) * B + c];))
        TDLS_ARROW_LUPP_FOR(r, acc1, B,
                            TDLS_ARROW_LUPP_FOR(c, 0, B,
                                                Sbb[r * B + c] = TDLS_ARROW_LUPP_A(
                                                    storage_map::abb(shape_value(), r, c));))
        if (converged()) {
            T rt[B * ntan];
            TDLS_ARROW_LUPP_FOR(
                m, 0, ntan,
                TDLS_ARROW_LUPP_FOR(r, 0, B, rt[m * B + r] = r == col0 + m ? T(1) : T(0);))
            // the refused blocks, stored as is, by the second pass of the
            // canonical solve, which writes neither x nor y
            if (refused_count > 0 &&
                !second_pass<ntan, true, internal_rhs, internal_work, internal_matrix>(
                    A, A_stride, work, work_stride, y, rhs_stride, 0, refused, Sbb, rt,
                    BlockSolver::singular_floor))
                return StreamedOutcome::singular;
            bool ok;
            if constexpr (ntan == 1)
                ok = BorderSolver::template solve_inplace<true, true, true>(Sbb, 1, pb, 1, rt, 1);
            else
                ok = BorderSolver::template solve_inplace_multirhs<ntan, true, true, true>(
                    Sbb, 1, pb, 1, rt, 1, B);
            if (!ok) return StreamedOutcome::singular;
            TDLS_ARROW_LUPP_FOR(
                m, 0, ntan,
                TDLS_ARROW_LUPP_FOR(r, 0, B,
                                    x[internal_rhs ? unsigned(m * B + r)
                                                   : TDLS_STRIDED(r, x_stride) +
                                                         TDLS_STRIDED_WIDE(m, xcol_stride)] =
                                        rt[m * B + r];))
            return StreamedOutcome::tangent;
        }
        T rb[B];
        TDLS_ARROW_LUPP_FOR(r, 0, acc0, rb[r] = TDLS_ARROW_LUPP_YW(0, r);)
        TDLS_ARROW_LUPP_FOR(r, acc0, acc1, rb[r] = racc[r - R0];)
        TDLS_ARROW_LUPP_FOR(r, acc1, B, rb[r] = TDLS_ARROW_LUPP_YW(0, r);)
        if (refused_count > 0 &&
            !second_pass<1, false, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, 0, refused, Sbb, rb,
                BlockSolver::singular_floor))
            return StreamedOutcome::singular;
        if (!BorderSolver::template solve_inplace<true, true, true>(Sbb, 1, pb, 1, rb, 1))
            return StreamedOutcome::singular;
        TDLS_ARROW_LUPP_FOR(r, 0, B, TDLS_ARROW_LUPP_YW(0, r) = rb[r];)
        recover<1, internal_rhs, internal_work, internal_matrix>(A, A_stride, work, work_stride, y,
                                                                 rhs_stride, 0, refused, rb);
        return StreamedOutcome::solved;
    }

    /// \brief solve_border_canonical_multirhs with the blocks streamed (see
    /// solve_inplace_streamed; blk.y is not read).
    /// \tparam nrhs            number of unit columns, in [1, border]
    /// \tparam internal_rhs    x is a caller-local array, column m from
    ///                         m * border (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \tparam Provider        callable (int k, StreamedBlock& blk)
    /// \param[in]     provider      the source of the blocks
    /// \param[in,out] A             storage: A_bb on entry; on exit the
    ///                              refused blocks as given, the accepted
    ///                              ones not written (see
    ///                              solve_border_canonical_multirhs)
    /// \param[in]     A_stride      element stride of A
    /// \param[out]    work          workspace of multirhs_workspace_size<nrhs>
    ///                              entries
    /// \param[in]     work_stride   element stride of work
    /// \param[in]     col0          first unit column, col0 + nrhs <= border
    /// \param[out]    x             the border rows, border entries per
    ///                              column
    /// \param[in]     rhs_stride    element stride of x
    /// \param[in]     xcol_stride   element stride between columns of x
    /// \param[out]    refused_count number of refused blocks
    /// \return false when the matrix is singular
    template<int nrhs, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool solve_border_canonical_multirhs_streamed(
        Provider&& provider, T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
        const int work_stride, const int col0, T* TDLS_RESTRICT x, const int rhs_stride,
        const std::size_t xcol_stride, int& refused_count) noexcept {
        static_assert(nrhs >= 1 && nrhs <= Shape.border,
                      "ArrowheadLUppSolverStatic: nrhs must be in [1, border]");
        if constexpr (Config.streamed_out_of_line)
            return solve_core_out_of_line<nrhs, true, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, x, rhs_stride, xcol_stride, col0, refused_count,
                &provider);
        else
            return solve_core<nrhs, true, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, x, rhs_stride, xcol_stride, col0, refused_count,
                &provider);
    }

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
    /// Column m comes out bitwise identical to solve_inplace on it alone
    /// under the default rhs_pass_width, and within rounding errors under
    /// another (see ArrowheadLUppConfig::rhs_pass_width).
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
                           const std::size_t xcol_stride, int& refused_count) noexcept {
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
                           const std::size_t xcol_stride) noexcept {
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
                   const int rhs_stride, const std::size_t xcol_stride,
                   int& refused_count) noexcept {
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < dimension; ++i) {
                if constexpr (internal_rhs)
                    x[m * dimension + i] = b[m * dimension + i];
                else
                    x[TDLS_STRIDED(i, rhs_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)] =
                        b[TDLS_STRIDED(i, rhs_stride) + TDLS_STRIDED_WIDE(m, xcol_stride)];
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
                   const int rhs_stride, const std::size_t xcol_stride) noexcept {
        int refused_count = 0;
        return solve_multirhs<nrhs, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, b, x, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Columns col0 .. col0 + nrhs - 1 of the border block of A^-1,
    /// the inverse of the Schur complement of the border (col0 + nrhs <=
    /// border), counting the refused blocks: rows 0 .. border - 1 of the
    /// solutions of A X = [e_col0 .. e_col0+nrhs-1]. These right-hand sides
    /// vanish on the rows of the blocks: the blocks are condensed without
    /// right-hand side, and nothing is recovered, so no recovery data is
    /// stored. The cost is the condensation of the matrix and nrhs
    /// substitutions of the border system: the tangent operator of a
    /// behaviour whose loading enters the border equations only. Each
    /// column equals the border rows of solve_inplace on the unit column; a
    /// zero may differ in sign, the blocks contributing exact zeros where
    /// solve_inplace subtracts zero products. Under another rhs_pass_width
    /// than the default, the two agree within rounding errors (see
    /// ArrowheadLUppConfig::rhs_pass_width). A is only read, except the
    /// coupled border rows under schur_in_place, which receive the Schur
    /// complement of the accepted blocks: otherwise the same A serves the
    /// next columns.
    /// \tparam nrhs            number of unit columns, in [1, border]
    /// \tparam internal_rhs    x is a caller-local array, column m from
    ///                         m * border (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix A is a caller-local array (stride ignored)
    /// \param[in,out] A             matrix in its storage; unchanged on exit,
    ///                              but under schur_in_place
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
                                    const int rhs_stride, const std::size_t xcol_stride,
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
    /// \param[in,out] A           matrix in its storage; unchanged on exit,
    ///                            but under schur_in_place
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
                                    const int rhs_stride, const std::size_t xcol_stride) noexcept {
        int refused_count = 0;
        return solve_border_canonical_multirhs<nrhs, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, col0, x, rhs_stride, xcol_stride, refused_count);
    }

    /// \brief Outcome of solve_inplace_recomputed and
    /// solve_inplace_recomputed_or_tangent (detail::ArrowheadRecomputedOutcome).
    using RecomputedOutcome = detail::ArrowheadRecomputedOutcome;

    /// \brief Entries of the workspace of solve_inplace_recomputed with a
    /// store of capacity refused blocks, and of
    /// solve_inplace_recomputed_or_tangent with nrhs tangent columns: the
    /// front of the second pass, the refused blocks in the Packed storage of
    /// a matrix of capacity blocks and their right-hand sides; none for a
    /// capacity of 0.
    template<int capacity, int nrhs = 1>
    static constexpr int recomputed_workspace_size = recomputed_size<capacity, nrhs>();

    /// \brief Solves A x = y in place, the blocks recomputed: provider(k,
    /// blk) fills block k (RecomputedBlock) once while the solver condenses
    /// it, k = 0, 1, ... in order, and once more, blk.recovery set, while it
    /// recovers the unknowns of the block, in decreasing k. Nothing of an
    /// accepted block is stored: the solver factors the block again and
    /// solves x_k = D_k^-1 (y_k - A_kb x_b). The matrix is never stored, and
    /// the workspace holds only the refused blocks, at most capacity of
    /// them, which the second pass eliminates as solve_inplace_streamed
    /// does. The provider must give the same entries every time it fills
    /// the same block, y_k included. The caller gives the border: A_bb,
    /// entry (r, c) at (r * border + c) * Abb_stride, or (c * border + r) *
    /// Abb_stride under the column-major layout, and the border rows of y.
    /// Nothing is written to y until the solution: y is untouched when the
    /// matrix is singular, or when more than capacity blocks are refused
    /// (deferred), so that the caller can solve that system with
    /// solve_inplace_streamed. A deferred solve stops at the refused block
    /// beyond the capacity: the provider is not called for the later
    /// blocks.
    ///
    /// The border solution, and the unknowns of the refused blocks, are
    /// those of solve_inplace_streamed bit for bit (under the same
    /// contraction of multiply-adds); the unknowns of an accepted block
    /// agree within rounding errors, solved from one column instead of
    /// recovered from the solved columns [A_kb | y_k].
    ///
    /// On GPU, the solve then streams nothing of the matrix through memory:
    /// one thread per system keeps the block and the Schur complement in
    /// registers (block_row_interchange). The cost is a second call of the
    /// provider, and a second factorization, per block: it pays where
    /// memory is the bound, on GPU, not on CPU, where the caches hold the
    /// recovery data.
    /// \tparam capacity        number of refused blocks the workspace holds,
    ///                         in [0, blocks]
    /// \tparam internal_rhs    y is a caller-local array (stride ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix Abb is a caller-local array (stride ignored)
    /// \tparam Provider        callable (int k, StreamedBlock& blk)
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     Abb           the border block A_bb, B x B in the layout
    ///                              of the configuration
    /// \param[in]     Abb_stride    element stride of Abb
    /// \param[out]    work          workspace of
    ///                              recomputed_workspace_size<capacity>
    ///                              entries (none for a capacity of 0)
    /// \param[in]     work_stride   element stride of work
    /// \param[in,out] y             border rows of the right-hand side on
    ///                              entry, the solution on exit when solved
    /// \param[in]     rhs_stride    element stride of y
    /// \param[out]    refused_count number of refused blocks
    /// \return solved, singular, or deferred when more than capacity blocks
    ///         are refused
    template<int capacity, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr RecomputedOutcome
    solve_inplace_recomputed(Provider&& provider, const T* TDLS_RESTRICT Abb, const int Abb_stride,
                             T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                             const int rhs_stride, int& refused_count) noexcept {
        return recomputed_core<0, capacity, internal_rhs, internal_work, internal_matrix>(
            provider, Abb, Abb_stride, work, work_stride, y, rhs_stride, 0, nullptr, 1, 0,
            refused_count);
    }

    /// \brief One recomputed pass of a Newton iteration: the blocks
    /// condensed as solve_inplace_recomputed condenses them, then
    /// converged() is called. When it returns false, the pass ends as
    /// solve_inplace_recomputed: the provider is asked for every accepted
    /// block once more and y holds the correction. When it returns true, the
    /// border system condensed by the pass gives instead columns col0 ..
    /// col0 + ntan - 1 of the border block of A^-1 in x (the consistent
    /// tangent of a behaviour loaded through the border), refused blocks
    /// included, and y is untouched: the provider is then called once per
    /// block. The tangent is that of solve_inplace_streamed_or_tangent bit
    /// for bit (under the same contraction of multiply-adds). When more
    /// than capacity blocks are refused, the pass stops there, deferred,
    /// before converged() is called; y and x are untouched.
    /// \tparam ntan            number of tangent columns, in [1, border]
    /// \tparam capacity        number of refused blocks the workspace holds,
    ///                         in [0, blocks]
    /// \tparam internal_rhs    y and x are caller-local arrays, column m of x
    ///                         from m * border (strides ignored)
    /// \tparam internal_work   work is a caller-local array (stride ignored)
    /// \tparam internal_matrix Abb is a caller-local array (stride ignored)
    /// \tparam Provider        callable (int k, StreamedBlock& blk)
    /// \tparam Converged       callable () returning bool
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     converged     called after the last block: whether the
    ///                              tangent replaces the correction
    /// \param[in]     Abb           the border block A_bb, B x B in the layout
    ///                              of the configuration
    /// \param[in]     Abb_stride    element stride of Abb
    /// \param[out]    work          workspace of
    ///                              recomputed_workspace_size<capacity, ntan>
    ///                              entries (none for a capacity of 0)
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
    /// \return solved, tangent, singular, or deferred when more than
    ///         capacity blocks are refused
    template<int ntan, int capacity, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider, typename Converged>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr RecomputedOutcome
    solve_inplace_recomputed_or_tangent(Provider&& provider, Converged&& converged,
                                        const T* TDLS_RESTRICT Abb, const int Abb_stride,
                                        T* TDLS_RESTRICT work, const int work_stride,
                                        T* TDLS_RESTRICT y, const int rhs_stride, const int col0,
                                        T* TDLS_RESTRICT x, const int x_stride,
                                        const std::size_t xcol_stride,
                                        int& refused_count) noexcept {
        static_assert(ntan >= 1 && ntan <= Shape.border,
                      "ArrowheadLUppSolverStatic: ntan must be in [1, border]");
        return recomputed_core<ntan, capacity, internal_rhs, internal_work, internal_matrix>(
            provider, Abb, Abb_stride, work, work_stride, y, rhs_stride, col0, x, x_stride,
            xcol_stride, refused_count, &converged);
    }

  private:
// gcc cannot always prove that the provider writes every entry of the
// streamed block before it is read, and reports the reads through
// -Wmaybe-uninitialized at -O3 (see condense_or_store_streamed): the report
// is spurious. The suppression is scoped to this function and to that
// warning only.
#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    /// \brief The recomputed solve shared by solve_inplace_recomputed (no
    /// converged(), ntan = 0) and solve_inplace_recomputed_or_tangent.
    /// \tparam ntan            number of tangent columns, 0 without converged()
    /// \tparam capacity        number of refused blocks the workspace holds
    /// \tparam internal_rhs    y and x are caller-local arrays
    /// \tparam internal_work   work is a caller-local array
    /// \tparam internal_matrix Abb is a caller-local array
    /// \tparam Provider        callable (int k, StreamedBlock& blk)
    /// \tparam Converged       none, or callable () returning bool
    /// \param[in]     provider      the source of the blocks
    /// \param[in]     Abb           the border block
    /// \param[in]     Abb_stride    element stride of Abb
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
    template<int ntan, int capacity, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider, typename... Converged>
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr RecomputedOutcome
    recomputed_core(Provider& provider, const T* TDLS_RESTRICT Abb, const int Abb_stride,
                    [[maybe_unused]] T* TDLS_RESTRICT work, [[maybe_unused]] const int work_stride,
                    T* TDLS_RESTRICT y, const int rhs_stride, [[maybe_unused]] const int col0,
                    [[maybe_unused]] T* TDLS_RESTRICT x, [[maybe_unused]] const int x_stride,
                    [[maybe_unused]] const std::size_t xcol_stride, int& refused_count,
                    Converged*... converged) noexcept {
        TDLS_ARROW_LUPP_SIDED(
            template recomputed_core<ntan, capacity, internal_rhs, internal_work, internal_matrix>(
                provider, Abb, Abb_stride, work, work_stride, y, rhs_stride, col0, x, x_stride,
                xcol_stride, refused_count, converged...))
        static_assert(capacity >= 0 && capacity <= NB,
                      "ArrowheadLUppSolverStatic: the capacity of the store of refused blocks "
                      "must be in [0, blocks]");
        static_assert(!in_place, "ArrowheadLUppSolverStatic: the recomputed solve accumulates the "
                                 "Schur complement apart (schur_in_place off)");
        // Columns of the front reserved in the workspace: the tangent
        // columns of the canonical second pass, one at least.
        constexpr int fnc = ntan > 1 ? ntan : 1;
        // Rows R0 .. R0 + R of the Schur complement and of its right-hand
        // side.
        T Sacc[R * B];
        T racc[R];
        TDLS_ARROW_LUPP_FOR(r, 0, R,
                            TDLS_ARROW_LUPP_FOR(c, 0, B,
                                                Sacc[r * B + c] = border_entry<internal_matrix>(
                                                    Abb, Abb_stride, R0 + r, c);) racc[r] =
                                TDLS_ARROW_LUPP_Y(R0 + r);)
        constexpr int words = (NB + 31) / 32;
        unsigned refused[words];
        TDLS_ARROW_LUPP_FOR(e, 0, words, refused[e] = 0u;)
        int count     = 0;
        refused_count = 0;
        for (int k = 0; k < NB; ++k) {
            RecomputedBlock blk;
            blk.recovery = false;
            provider(k, blk);
            // A save of the block, to keep it as given if it is refused: D_k,
            // and [A_kb | y_k] when the check reads the forward stage.
            constexpr bool forward_check = forward_check_for<1>;
            T Dsave[S * S];
            T Wsave[forward_check ? (B + 1) * S : 1];
            TDLS_ARROW_LUPP_FOR(e, 0, S * S, Dsave[e] = blk.Dl[e];)
            if constexpr (forward_check) {
                TDLS_ARROW_LUPP_FOR(e, 0, (B + 1) * S, Wsave[e] = blk.W[e];)
            } else {
                Wsave[0] = T(0);
            }
            if (condense_streamed<1, true, true, true>(nullptr, 1, nullptr, 1, 0, k, Sacc, racc,
                                                       blk))
                continue;
            // Refused: kept in the store as the provider gave it, or the
            // solve is deferred.
            ++count;
            refused_count = count;
            refused[k >> 5] |= 1u << (k & 31);
            if constexpr (capacity == 0) {
                return RecomputedOutcome::deferred;
            } else {
                if (count > capacity) return RecomputedOutcome::deferred;
                TDLS_ARROW_LUPP_FOR(e, 0, S * S, blk.Dl[e] = Dsave[e];)
                if constexpr (forward_check) {
                    TDLS_ARROW_LUPP_FOR(e, 0, (B + 1) * S, blk.W[e] = Wsave[e];)
                }
                keep_refused<capacity, fnc, internal_work>(work, work_stride, count - 1, blk);
            }
        }
        // The border system: the accumulated rows, the others as given.
        T Sbb[B * B];
        TDLS_ARROW_LUPP_FOR(
            r, 0, B, const bool acc = r >= R0 && r < R0 + R; TDLS_ARROW_LUPP_FOR(
                c, 0, B,
                Sbb[r * B + c] = acc ? Sacc[(acc ? r - R0 : 0) * B + c]
                                     : border_entry<internal_matrix>(Abb, Abb_stride, r, c);))
        if constexpr (sizeof...(Converged) > 0) {
            if ((*streamed_arg(converged...))()) {
                // The tangent: unit columns, the refused blocks of the store
                // by the second pass of the canonical solve.
                T rt[B * fnc];
                TDLS_ARROW_LUPP_FOR(
                    m, 0, fnc,
                    TDLS_ARROW_LUPP_FOR(r, 0, B, rt[m * B + r] = r == col0 + m ? T(1) : T(0);))
                if constexpr (capacity > 0) {
                    if (count > 0 && !second_pass_refused<capacity, fnc, fnc, true, internal_work>(
                                         work, work_stride, count, Sbb, rt))
                        return RecomputedOutcome::singular;
                }
                int pt[B];
                bool ok;
                if constexpr (fnc == 1)
                    ok = BorderSolver::template solve_inplace<true, true, true>(Sbb, 1, pt, 1, rt,
                                                                                1);
                else
                    ok = BorderSolver::template solve_inplace_multirhs<fnc, true, true, true>(
                        Sbb, 1, pt, 1, rt, 1, B);
                if (!ok) return RecomputedOutcome::singular;
                TDLS_ARROW_LUPP_FOR(
                    m, 0, fnc,
                    TDLS_ARROW_LUPP_FOR(r, 0, B,
                                        x[internal_rhs ? unsigned(m * B + r)
                                                       : TDLS_STRIDED(r, x_stride) +
                                                             TDLS_STRIDED_WIDE(m, xcol_stride)] =
                                            rt[m * B + r];))
                return RecomputedOutcome::tangent;
            }
        }
        T rb[B];
        TDLS_ARROW_LUPP_FOR(r, 0, B, const bool acc = r >= R0 && r < R0 + R;
                            rb[r] = acc ? racc[acc ? r - R0 : 0] : TDLS_ARROW_LUPP_Y(r);)
        if constexpr (capacity > 0) {
            if (count > 0 && !second_pass_refused<capacity, fnc, 1, false, internal_work>(
                                 work, work_stride, count, Sbb, rb))
                return RecomputedOutcome::singular;
        }
        int pb[B];
        if (!BorderSolver::template solve_inplace<true, true, true>(Sbb, 1, pb, 1, rb, 1))
            return RecomputedOutcome::singular;
        // The accepted blocks, recomputed, the last one first; then the
        // refused ones from the store; then the border.
        for (int k = NB - 1; k >= 0; --k) {
            if ((refused[k >> 5] >> (k & 31)) & 1u) continue;
            RecomputedBlock blk;
            blk.recovery = true;
            provider(k, blk);
            T xk[S];
            TDLS_ARROW_LUPP_FOR(i, 0, S, xk[i] = blk.W[B * S + i];
                                TDLS_ARROW_LUPP_FOR(c, 0, B, xk[i] -= blk.W[c * S + i] * rb[c];))
            recover_block(blk.Dl, xk);
            TDLS_ARROW_LUPP_FOR(i, 0, S, TDLS_ARROW_LUPP_Y(B + k * S + i) = xk[i];)
        }
        if constexpr (capacity > 0) {
            if (count > 0)
                recover_refused<capacity, fnc, internal_rhs, internal_work>(
                    work, work_stride, y, rhs_stride, refused, count, rb);
        }
        TDLS_ARROW_LUPP_FOR(r, 0, B, TDLS_ARROW_LUPP_Y(r) = rb[r];)
        return RecomputedOutcome::solved;
    }
#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic pop
#endif

    /// \brief Entry (r, c) of the border block given to
    /// solve_inplace_recomputed, in the layout of the configuration.
    /// \tparam internal_matrix Abb is a caller-local array (stride ignored)
    /// \param[in] Abb        the border block
    /// \param[in] Abb_stride element stride of Abb
    /// \param[in] r          row
    /// \param[in] c          column
    /// \return A_bb(r, c)
    template<bool internal_matrix>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr T
    border_entry(const T* TDLS_RESTRICT Abb, const int Abb_stride, const int r,
                 const int c) noexcept {
        const int e = col ? c * B + r : r * B + c;
        return Abb[internal_matrix ? unsigned(e) : TDLS_STRIDED(e, Abb_stride)];
    }

    /// \brief Keeps a refused block of solve_inplace_recomputed, as the
    /// provider gave it, in place j of the store: D_k, A_kb, A_bk in the
    /// Packed storage of the store, y_k in its right-hand side. The store
    /// follows the front of the second pass, fnc columns wide.
    /// \tparam capacity      capacity of the store
    /// \tparam fnc           columns of the front reserved before the store
    /// \tparam internal_work work is a caller-local array (stride ignored)
    /// \param[out] work        workspace
    /// \param[in]  work_stride element stride of work
    /// \param[in]  j           place in the store
    /// \param[in]  blk         the block as the provider gave it
    template<int capacity, int fnc, bool internal_work>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    keep_refused(T* TDLS_RESTRICT work, const int work_stride, const int j,
                 const StreamedBlock& blk) noexcept {
        using Store       = RefusedStore<capacity>;
        using Map         = typename Store::storage_map;
        constexpr int off = Store::template multirhs_workspace_size<fnc>;
        constexpr int oy  = off + Store::storage_size;
        TDLS_ARROW_LUPP_FOR(
            i, 0, S,
            TDLS_ARROW_LUPP_FOR(j2, 0, S,
                                TDLS_ARROW_LUPP_W(off + Map::d(store_shape<capacity>(), j, i, j2)) =
                                    blk.Dl[col ? j2 * S + i : i * S + j2];)
                TDLS_ARROW_LUPP_FOR(c, 0, B,
                                    TDLS_ARROW_LUPP_W(off + Map::akb(store_shape<capacity>(), j, i,
                                                                     c)) = blk.W[c * S + i];)
                    TDLS_ARROW_LUPP_W(oy + B + j * S + i) = blk.W[B * S + i];)
        if constexpr (diag) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_W(off + Map::abk_diag(store_shape<capacity>(), j, i)) = blk.Abk[i];)
        } else {
            TDLS_ARROW_LUPP_FOR(
                r, 0, B,
                TDLS_ARROW_LUPP_FOR(i, 0, S,
                                    TDLS_ARROW_LUPP_W(off + Map::abk(store_shape<capacity>(), j, r,
                                                                     i)) = blk.Abk[r * S + i];))
        }
    }

    /// \brief The second pass of solve_inplace_recomputed: the count
    /// refused blocks of the store eliminated with the border rows by the
    /// second pass of the solver of the store, which runs on the
    /// workspace.
    /// \tparam capacity      capacity of the store
    /// \tparam fnc           columns of the front reserved before the store
    /// \tparam nc            number of right-hand sides
    /// \tparam canonical     unit columns of the border (the tangent): no
    ///                       recovery data
    /// \tparam internal_work work is a caller-local array (stride ignored)
    /// \param[in,out] work        workspace
    /// \param[in]     work_stride element stride of work
    /// \param[in]     count       number of refused blocks
    /// \param[in,out] Sbb         the border matrix, the final one on exit
    /// \param[in,out] rb          its right-hand sides (column m at m * B),
    ///                            the final ones on exit
    /// \return false when the matrix is singular
    template<int capacity, int fnc, int nc, bool canonical, bool internal_work>
    TDLS_HOST_DEVICE static constexpr bool
    second_pass_refused(T* TDLS_RESTRICT work, const int work_stride, const int count,
                        T* TDLS_RESTRICT Sbb, T* TDLS_RESTRICT rb) noexcept {
        using Store         = RefusedStore<capacity>;
        constexpr int off   = Store::template multirhs_workspace_size<fnc>;
        constexpr int oy    = off + Store::storage_size;
        constexpr int words = (capacity + 31) / 32;
        unsigned held[words];
        for (int e = 0; e < words; ++e)
            held[e] = 0u;
        for (int j = 0; j < count; ++j)
            held[j >> 5] |= 1u << (j & 31);
        T* const store  = work + (internal_work ? off : TDLS_STRIDED(off, work_stride));
        T* const ystore = work + (internal_work ? oy : TDLS_STRIDED(oy, work_stride));
        return Store::template second_pass<nc, canonical, internal_work, internal_work,
                                           internal_work>(store, work_stride, work, work_stride,
                                                          ystore, work_stride, 0, held, Sbb, rb,
                                                          BlockSolver::singular_floor);
    }

// The block the provider wrote for the recovery (see the suppression
// before condense_streamed): scoped to recover_block and to that warning.
#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    /// \brief The unknowns of a recomputed accepted block: x := D_k^-1 x,
    /// D_k given as the provider filled it, factored again as the
    /// condensation factored it.
    /// \param[in,out] Dl the block in the layout of the factorization;
    ///                   its factors on exit
    /// \param[in,out] x  y_k - A_kb x_b on entry, x_k on exit
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    recover_block(T* TDLS_RESTRICT Dl, T* TDLS_RESTRICT x) noexcept {
        int pl[S];
        if constexpr (reg) {
            (void)register_factor(Dl, pl);
            register_solve<1, true>(Dl, pl, x);
        } else {
            (void)BlockSolver::template factorize<true, true>(Dl, 1, pl, 1);
            BlockSolver::template substitute_inplace<true, true, true>(Dl, 1, pl, 1, x, 1);
        }
    }

#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic pop
#endif

    /// \brief Recovers the refused blocks of solve_inplace_recomputed from
    /// the store, the last one first: x_k = c_k - G_k x_b - H_k w, w
    /// gathering the couplings A_bj x_j (the R coupled rows) of the
    /// refused blocks recovered so far, as recover does.
    /// \tparam capacity      capacity of the store
    /// \tparam fnc           columns of the front reserved before the store
    /// \tparam internal_rhs  y is a caller-local array (stride ignored)
    /// \tparam internal_work work is a caller-local array (stride ignored)
    /// \param[in]     work        workspace: the store, after the second pass
    /// \param[in]     work_stride element stride of work
    /// \param[out]    y           the unknowns of the refused blocks
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     refused     bit k set for a refused block k
    /// \param[in]     count       number of refused blocks
    /// \param[in]     rb          the border solution
    template<int capacity, int fnc, bool internal_rhs, bool internal_work>
    TDLS_HOST_DEVICE static constexpr void
    recover_refused(const T* TDLS_RESTRICT work, const int work_stride, T* TDLS_RESTRICT y,
                    const int rhs_stride, const unsigned* TDLS_RESTRICT refused, const int count,
                    const T* TDLS_RESTRICT rb) noexcept {
        using Store         = RefusedStore<capacity>;
        using Map           = typename Store::storage_map;
        constexpr int off   = Store::template multirhs_workspace_size<fnc>;
        constexpr int oy    = off + Store::storage_size;
        constexpr int fcols = front_cols_for<1>;
        T w[R];
        for (int a = 0; a < R; ++a)
            w[a] = T(0);
        int j = count;
        for (int k = NB - 1; k >= 0; --k) {
            if (!((refused[k >> 5] >> (k & 31)) & 1u)) continue;
            --j;
            T s[S];
            for (int i = 0; i < S; ++i) {
                s[i] = TDLS_ARROW_LUPP_W(oy + B + j * S + i);
                for (int c = 0; c < B; ++c)
                    s[i] -=
                        TDLS_ARROW_LUPP_W(off + Map::akb(store_shape<capacity>(), j, i, c)) * rb[c];
                for (int a = 0; a < R; ++a) {
                    const T h =
                        a < S ? TDLS_ARROW_LUPP_W(off + Map::d(store_shape<capacity>(), j, i, a))
                              : TDLS_ARROW_LUPP_W(front_rows * fcols + (j * S + i) * extra_cols +
                                                  a - S);
                    s[i] -= h * w[a];
                }
            }
            if constexpr (diag) {
                for (int i = 0; i < S; ++i)
                    w[i] += TDLS_ARROW_LUPP_W(off + Map::abk_diag(store_shape<capacity>(), j, i)) *
                            s[i];
            } else {
                for (int r = 0; r < B; ++r)
                    for (int i = 0; i < S; ++i)
                        w[r] +=
                            TDLS_ARROW_LUPP_W(off + Map::abk(store_shape<capacity>(), j, r, i)) *
                            s[i];
            }
            for (int i = 0; i < S; ++i)
                TDLS_ARROW_LUPP_Y(B + k * S + i) = s[i];
        }
    }

  private:
    /// \brief solve_inplace_streamed_or_tangent kept out of line on GPU
    /// (ArrowheadLUppConfig::streamed_out_of_line): the same call, its
    /// parameters taken by reference.
    template<int ntan, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider, typename Converged>
    TDLS_HOST_DEVICE TDLS_NO_INLINE_GPU static constexpr StreamedOutcome
    streamed_or_tangent_out_of_line(Provider& provider, Converged& converged, T* TDLS_RESTRICT A,
                                    const int A_stride, T* TDLS_RESTRICT work,
                                    const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                                    const int col0, T* TDLS_RESTRICT x, const int x_stride,
                                    const std::size_t xcol_stride, int& refused_count) noexcept {
        return solve_inplace_streamed_or_tangent<ntan, internal_rhs, internal_work, internal_matrix,
                                                 Provider&, Converged&, true>(
            provider, converged, A, A_stride, work, work_stride, y, rhs_stride, col0, x, x_stride,
            xcol_stride, refused_count);
    }

    /// \brief solve_core of a streamed entry point kept out of line on GPU
    /// (ArrowheadLUppConfig::streamed_out_of_line); same parameters.
    template<int nc, bool canonical, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename Provider>
    TDLS_HOST_DEVICE TDLS_NO_INLINE_GPU static constexpr bool
    solve_core_out_of_line(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                           const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                           const std::size_t xcol_stride, const int col0, int& refused_count,
                           Provider* provider) noexcept {
        return solve_core<nc, canonical, internal_rhs, internal_work, internal_matrix>(
            A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, col0, refused_count,
            provider);
    }

    /// \brief The single element of the trailing pack of the helpers shared
    /// by the stored and streamed paths: the provider or the streamed block,
    /// absent on the stored path, whose signatures and code so stay those
    /// of a solver without streaming.
    /// \tparam P type of the element
    /// \param[in] p the element
    /// \return p
    template<typename P>
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL static constexpr P*
    streamed_arg(P* p) noexcept {
        return p;
    }

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
    /// \tparam Provider        none on the stored path; streamed, the type
    ///                         of the callable (int k, StreamedBlock& blk)
    ///                         giving the blocks
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
    /// \param[in]     provider      streamed: the source of the blocks
    /// \return false when the matrix is singular
    template<int nc, bool canonical, bool internal_rhs, bool internal_work, bool internal_matrix,
             typename... Provider>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL static constexpr bool
    solve_core(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work, const int work_stride,
               T* TDLS_RESTRICT y, const int rhs_stride,
               [[maybe_unused]] const std::size_t xcol_stride, [[maybe_unused]] const int col0,
               int& refused_count, Provider*... provider) noexcept {
        TDLS_ARROW_LUPP_SIDED(
            template solve_core<nc, canonical, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, col0, refused_count,
                provider...))
        constexpr bool streamed = sizeof...(Provider) > 0;
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
            if constexpr (streamed) {
                static_assert(
                    ny <= 1,
                    "ArrowheadLUppSolverStatic: streamed blocks carry one right-hand side");
                StreamedBlock blk;
                (*streamed_arg(provider...))(k, blk);
                if (!condense_or_store_streamed<ny, internal_rhs, internal_matrix>(
                        A, A_stride, y, rhs_stride, xcol_stride, k, Sacc, racc, blk)) {
                    refused[k >> 5] |= 1u << (k & 31);
                    ++refused_count;
                }
            } else {
                if (!condense<ny, internal_rhs, internal_matrix>(A, A_stride, y, rhs_stride,
                                                                 xcol_stride, k, Sacc, racc)) {
                    refused[k >> 5] |= 1u << (k & 31);
                    ++refused_count;
                }
            }
        }
        // The border system: the accumulated rows, the others as stored;
        // column m of its right-hand sides at m * B.
        T Sbb[B * B];
        T rb[B * nc];
        // The rows as stored and the accumulated rows in separate loops:
        // GCC 11 and 12 vectorize a choice of source per entry wrongly (-O3
        // with AVX2 or AVX-512, unroll_loops off).
        constexpr int acc0 = in_place ? B : R0;
        constexpr int acc1 = in_place ? B : R0 + R;
        TDLS_ARROW_LUPP_FOR(
            r, 0, acc0,
            TDLS_ARROW_LUPP_FOR(
                c, 0, B, Sbb[r * B + c] = TDLS_ARROW_LUPP_A(storage_map::abb(shape_value(), r, c));)
                TDLS_ARROW_LUPP_FOR(
                    m, 0, nc, if constexpr (canonical) rb[m * B + r] = r == col0 + m ? T(1) : T(0);
                    else rb[m * B + r] = TDLS_ARROW_LUPP_YW(m, r);))
        TDLS_ARROW_LUPP_FOR(r, acc0, acc1,
                            TDLS_ARROW_LUPP_FOR(c, 0, B, Sbb[r * B + c] = Sacc[(r - R0) * B + c];)
                                TDLS_ARROW_LUPP_FOR(m, 0, nc,
                                                    if constexpr (canonical) rb[m * B + r] =
                                                        r == col0 + m ? T(1) : T(0);
                                                    else rb[m * B + r] = racc[m * R + r - R0];))
        TDLS_ARROW_LUPP_FOR(
            r, acc1, B,
            TDLS_ARROW_LUPP_FOR(
                c, 0, B, Sbb[r * B + c] = TDLS_ARROW_LUPP_A(storage_map::abb(shape_value(), r, c));)
                TDLS_ARROW_LUPP_FOR(
                    m, 0, nc, if constexpr (canonical) rb[m * B + r] = r == col0 + m ? T(1) : T(0);
                    else rb[m * B + r] = TDLS_ARROW_LUPP_YW(m, r);))
        if (refused_count > 0 &&
            !second_pass<nc, canonical, internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, y, rhs_stride, xcol_stride, refused, Sbb, rb,
                BlockSolver::singular_floor))
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
                TDLS_ARROW_LUPP_FOR(r, 0, B,
                                    y[internal_rhs ? unsigned(m * B + r)
                                                   : TDLS_STRIDED(r, rhs_stride) +
                                                         TDLS_STRIDED_WIDE(m, xcol_stride)] =
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
             [[maybe_unused]] const std::size_t xcol_stride, const int k, T* TDLS_RESTRICT Sacc,
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
        // Factored by TiledLUpp, or in registers: then Dl holds U in
        // logical row order and the check of L is made.
        bool ok = true;
        if constexpr (reg) {
            const int verdict = register_factor(Dl, pl);
            if (verdict == 2) return false;
            ok = verdict == 0;
        } else {
            if (!BlockSolver::template factorize<true, true>(Dl, 1, pl, 1)) return false;
        }
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
        // gives the border multipliers, or the block is factored in
        // registers (U is then Dl).
        T U[forward_check || reg ? 1 : S * S];
        if constexpr (forward_check || reg) U[0] = T(0);
        if constexpr (!reg) ok = lower_pass<!forward_check>(Dl, pl, U);
        const T* const Uf = reg ? Dl : U;
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
            ok = border_multipliers<internal_matrix>(A, A_stride, k, Uf, acoef, Mb) && ok;
            if (!ok) return false;
            if constexpr (multiplier_update && stores_recovery_for<ny>)
                store_upper<internal_matrix>(A, A_stride, k, Uf);
            condense_pass<0, false, ny, internal_rhs, internal_matrix>(
                A, A_stride, y, rhs_stride, xcol_stride, k, Dl, pl, acoef, Mb, Sacc, racc, nullptr);
        }
        return true;
    }

// gcc cannot always prove that the provider writes every entry of the
// streamed block before it is read or copied, and reports the reads through
// -Wmaybe-uninitialized at -O3 (gcc 11; gcc 13 with -march=native; gcc 10
// on the reads of the block factored in registers). The provider writes
// the block (StreamedBlock), and the copies only keep it as given: the
// report is spurious. The suppression is scoped to the three functions
// below and to that warning only.
#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    /// \brief condense on a streamed block, in place in its arrays: the
    /// factorization in blk.Dl, the solve of the columns [A_kb | y_k] in
    /// blk.W, all at once, or in passes of rhs_pass_width columns through
    /// condense_pass when the pivot check does not read the forward stage
    /// (the arithmetic of the stored path, column by column), then the
    /// recovery data stored (unless ny = 0) and the Schur complement
    /// updated as condense_pass does.
    /// \tparam ny              right-hand sides carried through the blocks
    ///                         (0 or 1)
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A           matrix in its storage: the recovery data of
    ///                            the block on exit (none when ny = 0)
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] y           right-hand sides: the recovery right-hand
    ///                            side of the block on exit
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in,out] Sacc        accumulated rows of the Schur complement
    /// \param[in,out] racc        accumulated rows of its right-hand sides
    /// \param[in,out] blk         the block as the provider filled it;
    ///                            overwritten
    /// \tparam recompute       no recovery data stored, A and y untouched
    ///                         (solve_inplace_recomputed)
    /// \return true when the block is condensed, false when it fails the
    ///         pivot check
    template<int ny, bool internal_rhs, bool internal_matrix, bool recompute = false>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    condense_streamed(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y,
                      const int rhs_stride, [[maybe_unused]] const std::size_t xcol_stride,
                      const int k, T* TDLS_RESTRICT Sacc, T* TDLS_RESTRICT racc,
                      StreamedBlock& blk) noexcept {
        constexpr bool forward_check         = forward_check_for<ny>;
        constexpr int cols                   = B + ny;
        [[maybe_unused]] constexpr int ycols = ny == 0 ? 1 : ny;
        const StreamedBlock* const bp        = &blk;
        T* const Dl                          = blk.Dl;
        T* const W                           = blk.W;
        int pl[S];
        bool ok = true;
        if constexpr (reg) {
            const int verdict = register_factor(Dl, pl);
            if (verdict == 2) return false;
            ok = verdict == 0;
        } else {
            if (!BlockSolver::template factorize<true, true>(Dl, 1, pl, 1)) return false;
        }
        const T* const acoef = blk.Abk;
        T U[forward_check || reg ? 1 : S * S];
        if constexpr (forward_check || reg) U[0] = T(0);
        if constexpr (!reg) ok = lower_pass<!forward_check>(Dl, pl, U);
        const T* const Uf = reg ? Dl : U;
        T Mb[multiplier_update ? R * S : 1];
        if constexpr (!multiplier_update) Mb[0] = T(0);
        if constexpr (forward_check) {
            BlockSolver::template forward_inplace_multirhs<cols, true, true, true>(Dl, 1, pl, 1, W,
                                                                                   1, S);
            const T u = condensation_pivot_threshold;
            TDLS_ARROW_LUPP_FOR(e, 0, B * S, ok = ok && u * detail::abs(W[e]) <= T(1);)
            if (!ok) return false;
            BlockSolver::template backward_inplace_multirhs<cols, true, true, true>(Dl, 1, pl, 1, W,
                                                                                    1, S);
        } else {
            ok = border_multipliers<internal_matrix>(A, A_stride, k, Uf, acoef, Mb, bp) && ok;
            if (!ok) return false;
            if constexpr (pass_width_for<ny> < cols) {
                // rhs_pass_width columns at a time, read from the block as
                // the stored path reads them from the storage (only one
                // slice alive at a time); all of them at once otherwise
                if constexpr (multiplier_update && stores_recovery_for<ny> && !recompute)
                    store_upper<internal_matrix>(A, A_stride, k, Uf);
                condense_pass<0, false, ny, internal_rhs, internal_matrix, recompute>(
                    A, A_stride, y, rhs_stride, xcol_stride, k, Dl, pl, acoef, Mb, Sacc, racc,
                    nullptr, bp);
                return true;
            }
            if constexpr (multiplier_update) {
                if constexpr (stores_recovery_for<ny> && !recompute)
                    store_upper<internal_matrix>(A, A_stride, k, Uf);
                if constexpr (reg)
                    register_solve<cols>(Dl, pl, W);
                else
                    BlockSolver::template forward_inplace_multirhs<cols, true, true, true>(
                        Dl, 1, pl, 1, W, 1, S);
            } else if constexpr (reg) {
                register_solve<cols>(Dl, pl, W);
            } else if constexpr (Config.gather_rhs) {
                T X[cols * S];
                BlockSolver::template substitute_multirhs<cols, true, true, true>(Dl, 1, pl, 1, W,
                                                                                  X, 1, S);
                TDLS_ARROW_LUPP_FOR(e, 0, cols * S, W[e] = X[e];)
            } else {
                BlockSolver::template substitute_inplace_multirhs<cols, true, true, true>(
                    Dl, 1, pl, 1, W, 1, S);
            }
        }
        // the recovery data (not for the canonical solve, nor for the
        // recomputed one) and the Schur complement, all the columns (the
        // entries are distinct per column: the order of the passes of the
        // stored path does not matter)
        if constexpr (stores_recovery_for<ny> && !recompute) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(j, 0, cols,
                                    if (j < B) TDLS_ARROW_LUPP_A(
                                        storage_map::akb(shape_value(), k, i, j)) = W[j * S + i];
                                    else TDLS_ARROW_LUPP_YW(j - B, B + k * S + i) = W[j * S + i];))
        }
        if constexpr (multiplier_update) {
            TDLS_ARROW_LUPP_FOR(
                r, 0, R,
                TDLS_ARROW_LUPP_FOR(
                    j, 0, cols, T sum = T(0);
                    TDLS_ARROW_LUPP_FOR(l, 0, S, sum += Mb[r * S + l] * W[j * S + l];)
                        schur_sub<ny, internal_rhs, internal_matrix>(
                            A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc, r, j, sum);))
        } else if constexpr (diag) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(j, 0, cols,
                                    schur_sub<ny, internal_rhs, internal_matrix>(
                                        A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc, i, j,
                                        acoef[i] * W[j * S + i]);))
        } else {
            TDLS_ARROW_LUPP_FOR(
                r, 0, B, T a[S];
                TDLS_ARROW_LUPP_FOR(i, 0, S, a[i] = blk.Abk[r * S + i];) TDLS_ARROW_LUPP_FOR(
                    j, 0, cols, T sum = T(0);
                    TDLS_ARROW_LUPP_FOR(i, 0, S, sum += a[i] * W[j * S + i];)
                        schur_sub<ny, internal_rhs, internal_matrix>(
                            A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc, r, j, sum);))
        }
        return true;
    }

    /// \brief condense_streamed, the block kept for the second pass when it
    /// is refused: D_k is copied before its factorization (and [A_kb | y_k]
    /// before the forward solve when the check reads it), so that a refused
    /// block is stored as the provider gave it without asking for it again.
    /// \tparam ny              right-hand sides carried through the blocks
    ///                         (0 or 1)
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A           matrix in its storage: the recovery data of
    ///                            the block (none when ny = 0), or the block
    ///                            itself when refused
    /// \param[in]     A_stride    element stride of A
    /// \param[in,out] y           right-hand sides
    /// \param[in]     rhs_stride  element stride of y
    /// \param[in]     xcol_stride element stride between columns of y
    /// \param[in]     k           block
    /// \param[in,out] Sacc        accumulated rows of the Schur complement
    /// \param[in,out] racc        accumulated rows of its right-hand sides
    /// \param[in,out] blk         the block as the provider filled it
    /// \return true when the block is condensed, false when it is refused
    ///         (and stored as is)
    template<int ny, bool internal_rhs, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    condense_or_store_streamed(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y,
                               const int rhs_stride, const std::size_t xcol_stride, const int k,
                               T* TDLS_RESTRICT Sacc, T* TDLS_RESTRICT racc,
                               StreamedBlock& blk) noexcept {
        constexpr bool forward_check = forward_check_for<ny>;
        constexpr int wsave          = forward_check ? (B + ny) * S : 1;
        T Dsave[S * S];
        T Wsave[wsave];
        TDLS_ARROW_LUPP_FOR(e, 0, S * S, Dsave[e] = blk.Dl[e];)
        if constexpr (forward_check) {
            TDLS_ARROW_LUPP_FOR(e, 0, wsave, Wsave[e] = blk.W[e];)
        } else {
            Wsave[0] = T(0);
        }
        if (condense_streamed<ny, internal_rhs, internal_matrix>(A, A_stride, y, rhs_stride,
                                                                 xcol_stride, k, Sacc, racc, blk))
            return true;
        TDLS_ARROW_LUPP_FOR(e, 0, S * S, blk.Dl[e] = Dsave[e];)
        if constexpr (forward_check) { TDLS_ARROW_LUPP_FOR(e, 0, wsave, blk.W[e] = Wsave[e];) }
        store_block<ny, internal_rhs, internal_matrix>(A, A_stride, y, rhs_stride, xcol_stride, k,
                                                       blk);
        return false;
    }

    /// \brief Stores a refused streamed block where the second pass reads
    /// it: the state the stored path leaves a refused block in.
    /// \tparam ny              right-hand sides carried through the blocks
    ///                         (0 or 1)
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \param[out] A           matrix in its storage: D_k, A_kb and A_bk of
    ///                         the block
    /// \param[in]  A_stride    element stride of A
    /// \param[out] y           right-hand sides: y_k when ny is 1
    /// \param[in]  rhs_stride  element stride of y
    /// \param[in]  xcol_stride element stride between columns of y
    /// \param[in]  k           block
    /// \param[in]  blk         the block as the provider filled it
    template<int ny, bool internal_rhs, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    store_block(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                [[maybe_unused]] const std::size_t xcol_stride, const int k,
                const StreamedBlock& blk) noexcept {
        [[maybe_unused]] constexpr int ycols = ny == 0 ? 1 : ny;
        TDLS_ARROW_LUPP_FOR(
            i, 0, S,
            TDLS_ARROW_LUPP_FOR(j, 0, S,
                                TDLS_ARROW_LUPP_A(storage_map::d(shape_value(), k, i, j)) =
                                    blk.Dl[col ? j * S + i : i * S + j];)
                TDLS_ARROW_LUPP_FOR(c, 0, B,
                                    TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c)) =
                                        blk.W[c * S + i];))
        if constexpr (diag) {
            TDLS_ARROW_LUPP_FOR(i, 0, S,
                                TDLS_ARROW_LUPP_A(storage_map::abk_diag(shape_value(), k, i)) =
                                    blk.Abk[i];)
        } else {
            TDLS_ARROW_LUPP_FOR(
                r, 0, B,
                TDLS_ARROW_LUPP_FOR(i, 0, S,
                                    TDLS_ARROW_LUPP_A(storage_map::abk(shape_value(), k, r, i)) =
                                        blk.Abk[r * S + i];))
        }
        if constexpr (ny == 1) {
            TDLS_ARROW_LUPP_FOR(i, 0, S, TDLS_ARROW_LUPP_YW(0, B + k * S + i) = blk.W[B * S + i];)
        }
    }
#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic pop
#endif

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
    /// \tparam Streamed        none on the stored path; streamed, the block
    ///                         type: A_kb and A_bk read from blk, not from A
    /// \param[in]  A        matrix in its storage
    /// \param[in]  A_stride element stride of A
    /// \param[in]  k        block
    /// \param[in]  U        U of the block in logical row order
    /// \param[in]  acoef    couplings of the block (diagonal coupling)
    /// \param[out] Mb       the multipliers (row-major, dense coupling)
    /// \param[in]  blk      streamed: the block
    /// \return true when no multiplier exceeds the bound
    template<bool internal_matrix, typename... Streamed>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr bool
    border_multipliers(const T* TDLS_RESTRICT A, const int A_stride, const int k,
                       const T* TDLS_RESTRICT U, const T* TDLS_RESTRICT acoef, T* TDLS_RESTRICT Mb,
                       const Streamed* TDLS_RESTRICT... blk) noexcept {
        constexpr bool streamed = sizeof...(Streamed) > 0;
        const T u               = condensation_pivot_threshold;
        bool ok                 = true;
        if constexpr (col) {
            TDLS_ARROW_LUPP_FOR(
                c, 0, B, T m[S];
                TDLS_ARROW_LUPP_FOR(
                    i, 0, S, if constexpr (streamed) m[i] = streamed_arg(blk...)->W[c * S + i];
                    else m[i] = TDLS_ARROW_LUPP_A(storage_map::akb(shape_value(), k, i, c));) ok =
                    multipliers_pass(m, 0, U, u) && ok;)
        } else if constexpr (diag) {
            // Border row coupling_row + i holds one entry, in column i.
            TDLS_ARROW_LUPP_FOR(i, 0, S, T m[S];
                                TDLS_ARROW_LUPP_FOR(j, 0, S, m[j] = j == i ? acoef[i] : T(0);) ok =
                                    multipliers_pass(m, i, U, u) && ok;)
        } else {
            TDLS_ARROW_LUPP_FOR(r, 0, B,
                                TDLS_ARROW_LUPP_FOR(j, 0, S,
                                                    if constexpr (streamed) Mb[r * S + j] =
                                                        streamed_arg(blk...)->Abk[r * S + j];
                                                    else Mb[r * S + j] = TDLS_ARROW_LUPP_A(
                                                        storage_map::abk(shape_value(), k, r, j));)
                                    ok = multipliers_pass(Mb + r * S, 0, U, u) && ok;)
        }
        return ok;
    }

// The block factored in registers is, on the streamed and recomputed
// paths, the block the provider wrote (see the suppression before
// condense_streamed): gcc 10 reports its reads through
// -Wmaybe-uninitialized, spuriously. Scoped to the functions below and to
// that warning only.
#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    /// \brief |x|, as detail::abs: std::fabs at run time, a ternary under
    /// constant evaluation. The register path takes the magnitudes of the
    /// block the provider wrote through it rather than through detail::abs:
    /// gcc 10 places the report of a maybe uninitialized value read by an
    /// inlined call at the body of the callee, which the suppression above
    /// then covers.
    /// \param[in] x value
    /// \return the absolute value of x
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr T
    magnitude(const T x) noexcept {
#if defined(__has_builtin)
#if __has_builtin(__builtin_is_constant_evaluated)
        if (__builtin_is_constant_evaluated()) return x < T(0) ? -x : x;
#endif
#elif defined(TDLS_DETAIL_COMPILER_MSVC) && _MSC_VER >= 1925
        if (__builtin_is_constant_evaluated()) return x < T(0) ? -x : x;
#endif
        return std::fabs(x);
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
        if constexpr (reg) {
            // The same operations over full ranges, the triangle selected
            // inside, so that U stays in registers (see register_factor).
            TDLS_ARROW_LUPP_FOR(
                l, 0, S, if (l >= first) {
                    m[l] *= U[l * S + l];
                    ok = ok && u * magnitude(m[l]) <= T(1);
                    TDLS_ARROW_LUPP_FOR(j, 0, S, if (j > l) m[j] -= m[l] * U[l * S + j];)
                })
        } else {
            TDLS_ARROW_LUPP_FOR(l, first, S, m[l] *= U[l * S + l];
                                ok = ok && u * detail::abs(m[l]) <= T(1);
                                TDLS_ARROW_LUPP_FOR(j, l + 1, S, m[j] -= m[l] * U[l * S + j];))
        }
        return ok;
    }

    /// \brief Stores U of block k, in logical row order, in the upper
    /// triangle of D_k (its diagonal holds 1 / pivot): the recovery data
    /// of the row-major layout with the dense coupling.
    /// \tparam internal_matrix residency of A
    /// \param[out] A        matrix in its storage
    /// \param[in]  A_stride element stride of A
    /// \param[in]  k        block
    /// \param[in]  U        U of the block in logical row order
    template<bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    store_upper(T* TDLS_RESTRICT A, const int A_stride, const int k,
                const T* TDLS_RESTRICT U) noexcept {
        if constexpr (reg) {
            TDLS_ARROW_LUPP_FOR(l, 0, S,
                                TDLS_ARROW_LUPP_FOR(j, 0, S,
                                                    if (j >= l) TDLS_ARROW_LUPP_A(storage_map::d(
                                                        shape_value(), k, l, j)) = U[l * S + j];))
        } else {
            TDLS_ARROW_LUPP_FOR(l, 0, S,
                                TDLS_ARROW_LUPP_FOR(j, l, S,
                                                    TDLS_ARROW_LUPP_A(storage_map::d(
                                                        shape_value(), k, l, j)) = U[l * S + j];))
        }
    }

    /// \brief Factors F in place in registers (ArrowheadLUppConfig::
    /// block_row_interchange), by TiledLUpp with physical row interchanges
    /// (PhysicalBlockSolver). F is the block as TiledLUpp factors it, D_k or
    /// D_k^T under the column-major layout, S x S by rows. The pivot rule
    /// is the in-tile rule of TiledLUpp over a single tile; the rows are
    /// interchanged physically and whole. On exit F holds L below its
    /// diagonal and U above it, in logical row order, with 1 / pivot on the
    /// diagonal (the format of TiledLUpp, row by row), and p[j] is the row
    /// interchanged with row j at step j. The multipliers of L are checked
    /// as lower_pass checks them, inside the elimination. Once the loops are
    /// unrolled,
    /// every index is known at compile time and F stays in registers.
    /// \param[in,out] F the block on entry, its factors on exit
    /// \param[out]    p the interchanges
    /// \return 0 when L passes the check, 1 when it fails it, 2 when a
    ///         column has no candidate above the singularity floor of
    ///         TiledLUpp (BlockSolver::singular_floor)
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    register_factor(T* TDLS_RESTRICT F, int* TDLS_RESTRICT p) noexcept {
        return PhysicalBlockSolver::template physical_factorize_checked<true, true>(
            F, 1, p, 1, condensation_pivot_threshold);
    }

    /// \brief Solves w columns in place in registers with the factors of
    /// register_factor, column c at c * S: W := D_k^-1 W, or the forward
    /// stage alone, L^-1 P W, under the row-major layout with the dense
    /// coupling (multiplier_update). Under the column-major layout F = D_k^T
    /// and P F = L U, so D_k = U^T L^T P: U^-T, then L^-T, then the
    /// interchanges in reverse order. The two stages of PhysicalBlockSolver.
    /// \tparam w        number of columns
    /// \tparam backward whether the row-major solve runs its backward stage
    ///                  (always, but for the forward stage alone of the
    ///                  condensation with the dense coupling)
    /// \param[in]     F factors
    /// \param[in]     p interchanges
    /// \param[in,out] W columns
    template<int w, bool backward = !multiplier_update>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    register_solve(const T* TDLS_RESTRICT F, const int* TDLS_RESTRICT p,
                   T* TDLS_RESTRICT W) noexcept {
        PhysicalBlockSolver::template forward_inplace_multirhs<w, true, true, true>(F, 1, p, 1, W,
                                                                                    1, S);
        if constexpr (col || backward)
            PhysicalBlockSolver::template backward_inplace_multirhs<w, true, true, true>(F, 1, p, 1,
                                                                                         W, 1, S);
    }

#if defined(TDLS_DETAIL_COMPILER_GCC)
#pragma GCC diagnostic pop
#endif

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
              [[maybe_unused]] const std::size_t xcol_stride, T* TDLS_RESTRICT Sacc,
              T* TDLS_RESTRICT racc, const int r, const int c, const T v) noexcept {
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
    /// the dense coupling, W_k otherwise), stored in place unless ny = 0
    /// (stores_recovery_for), folded into the accumulated rows. The columns
    /// go in passes of pass_width_for<ny> so that only one slice is alive
    /// at a time. When solved, Wsolved already holds all of W_k (one pass).
    /// \tparam c0              first column of the pass
    /// \tparam solved          whether Wsolved holds the solved columns
    /// \tparam ny              right-hand sides carried through the blocks
    /// \tparam internal_rhs    residency of y
    /// \tparam internal_matrix residency of A
    /// \tparam recompute       no recovery data stored
    ///                         (solve_inplace_recomputed)
    /// \tparam Streamed        none on the stored path; streamed, the block
    ///                         type: the columns and A_bk read from blk, not
    ///                         from A and y
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
    /// \param[in]     blk         streamed: the block
    template<int c0, bool solved, int ny, bool internal_rhs, bool internal_matrix,
             bool recompute = false, typename... Streamed>
    TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr void
    condense_pass(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                  [[maybe_unused]] const std::size_t xcol_stride, const int k,
                  const T* TDLS_RESTRICT Dl, const int* TDLS_RESTRICT pl,
                  const T* TDLS_RESTRICT acoef, const T* TDLS_RESTRICT Mb, T* TDLS_RESTRICT Sacc,
                  T* TDLS_RESTRICT racc, T* TDLS_RESTRICT Wsolved,
                  const Streamed* TDLS_RESTRICT... blk) noexcept {
        constexpr bool streamed = sizeof...(Streamed) > 0;
        constexpr int cols      = B + ny;
        constexpr int w         = cols - c0 < pass_width_for<ny> ? cols - c0 : pass_width_for<ny>;
        [[maybe_unused]] constexpr int ycols = ny;
        T Wl[solved ? 1 : w * S];
        T* W = solved ? Wsolved : Wl;
        if constexpr (!solved) {
            if constexpr (streamed) {
                // the slice of the streamed block: [A_kb | y_k] by columns
                TDLS_ARROW_LUPP_FOR(
                    i, 0, S,
                    TDLS_ARROW_LUPP_FOR(j, 0, w,
                                        W[j * S + i] = streamed_arg(blk...)->W[(c0 + j) * S + i];))
            } else {
                TDLS_ARROW_LUPP_FOR(
                    i, 0, S,
                    TDLS_ARROW_LUPP_FOR(
                        j, 0, w,
                        W[j * S + i] = c0 + j < B ? TDLS_ARROW_LUPP_A(storage_map::akb(
                                                        shape_value(), k, i, c0 + j))
                                                  : TDLS_ARROW_LUPP_YW(c0 + j - B, B + k * S + i);))
            }
            if constexpr (reg) {
                register_solve<w>(Dl, pl, W);
            } else if constexpr (multiplier_update) {
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
        if constexpr (stores_recovery_for<ny> && !recompute) {
            TDLS_ARROW_LUPP_FOR(
                i, 0, S,
                TDLS_ARROW_LUPP_FOR(
                    j, 0, w,
                    if (c0 + j < B) TDLS_ARROW_LUPP_A(
                        storage_map::akb(shape_value(), k, i, c0 + j)) = W[j * S + i];
                    else TDLS_ARROW_LUPP_YW(c0 + j - B, B + k * S + i) = W[j * S + i];))
        }
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
                    i, 0, S, if constexpr (streamed) a[i] = streamed_arg(blk...)->Abk[r * S + i];
                    else a[i] = TDLS_ARROW_LUPP_A(storage_map::abk(shape_value(), k, r, i));)
                    TDLS_ARROW_LUPP_FOR(j, 0, w, T sum = T(0);
                                        TDLS_ARROW_LUPP_FOR(i, 0, S, sum += a[i] * W[j * S + i];)
                                            schur_sub<ny, internal_rhs, internal_matrix>(
                                                A, A_stride, y, rhs_stride, xcol_stride, Sacc, racc,
                                                r, c0 + j, sum);))
        }
        if constexpr (c0 + w < cols)
            condense_pass<c0 + w, false, ny, internal_rhs, internal_matrix, recompute>(
                A, A_stride, y, rhs_stride, xcol_stride, k, Dl, pl, acoef, Mb, Sacc, racc, nullptr,
                blk...);
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
            [[maybe_unused]] const std::size_t xcol_stride, const unsigned* TDLS_RESTRICT refused,
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
    /// \param[in,out] A              matrix in its storage; the refused blocks
    ///                               receive their recovery data
    /// \param[in]     A_stride       element stride of A
    /// \param[out]    work           workspace: the front, then H_k beyond its
    ///                               S columns
    /// \param[in]     work_stride    element stride of work
    /// \param[in,out] y              right-hand sides
    /// \param[in]     rhs_stride     element stride of y
    /// \param[in]     xcol_stride    element stride between columns of y
    /// \param[in]     refused        bit k set for a refused block k
    /// \param[in,out] Sbb            the border matrix, the final one on exit
    /// \param[in,out] rb             its right-hand sides (column m at m * B),
    ///                               the final ones on exit
    /// \param[in]     singular_floor the singularity floor of the block solver
    ///                               (BlockSolver::singular_floor): a smaller
    ///                               pivot, zero or subnormal, is singular, and
    ///                               its reciprocal could overflow. An argument
    ///                               rather than a constant: hoisted as a
    ///                               constant, it makes the AMDGPU backend of
    ///                               LLVM 15 (ROCm 5.3) fail with "unhandled
    ///                               SGPR spill to memory"
    /// \return false when a column of a refused block has no nonzero
    ///         candidate: the matrix is singular
    template<int nc, bool canonical, bool internal_rhs, bool internal_work, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_NO_INLINE static constexpr bool
    second_pass(T* TDLS_RESTRICT A, const int A_stride, T* TDLS_RESTRICT work,
                const int work_stride, T* TDLS_RESTRICT y, const int rhs_stride,
                [[maybe_unused]] const std::size_t xcol_stride,
                const unsigned* TDLS_RESTRICT refused, T* TDLS_RESTRICT Sbb, T* TDLS_RESTRICT rb,
                const T singular_floor) noexcept {
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
                if (!(vmax >= singular_floor)) return false;
                const int p = pblk >= 0 && vblk >= u * vmax && vblk >= singular_floor ? pblk : pmax;
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
#undef TDLS_ARROW_LUPP_YW
#undef TDLS_ARROW_LUPP_W
#undef TDLS_ARROW_LUPP_F
#undef TDLS_ARROW_LUPP_FOR
#undef TDLS_ARROW_LUPP_SIDED



} // namespace tdls

#if defined(TDLS_DETAIL_COMPILER_NVCC)
#pragma nv_diagnostic pop
#endif
#if defined(TDLS_DETAIL_COMPILER_CLANG)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_ARROWHEAD_LUPP_SOLVER_STATIC_HPP
