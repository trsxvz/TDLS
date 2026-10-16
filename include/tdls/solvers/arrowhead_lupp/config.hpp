#ifndef TDLS_SOLVERS_ARROWHEAD_LUPP_CONFIG_HPP
#define TDLS_SOLVERS_ARROWHEAD_LUPP_CONFIG_HPP



/// \file
/// \brief Shape and compile-time configuration of the ArrowheadLUpp solver
/// family.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// An arrowhead matrix, or bordered block-diagonal matrix, has `border`
/// unknowns first, then `blocks` diagonal blocks of `block` unknowns:
///
///     [ Abb  Ab0  Ab1 ... ]
///     [ A0b  D0           ]
///     [ A1b       D1      ]
///     [ ...           ... ]
///
/// Two blocks never couple. ArrowheadShape holds the four integers of
/// this pattern; it is a template argument of the compile-time solver and
/// a function argument of the runtime one, as the dimension N of
/// TiledLUpp. Every knob is a member of the ArrowheadLUppConfig
/// aggregate, passed as the `Config` non-type template argument of both
/// solvers, with designated initializers in declaration order:
///
///   constexpr auto config = tdls::ArrowheadLUppConfig<double>{
///       .coupling = tdls::ArrowheadCoupling::Diagonal,
///       .storage  = tdls::ArrowheadStorage::Packed};
///
/// Both aggregates are structural types: every member is public, and two
/// values with equal members name the same instantiation. The
/// floating-point knobs are held as tdls::StructuralReal values, as in
/// the other families.



#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/core/structural_real.hpp>
#include <tdls/solvers/options.hpp>
#include <tdls/solvers/tiled_lupp/config.hpp>



namespace tdls {



/// \brief Storage of an arrowhead matrix.
///
/// Every storage holds the entries of the arrowhead pattern, the zeros
/// inside the stored blocks included. The pattern is known from the shape,
/// so the position of every entry is computed, never looked up: Csr and
/// Bsr need no index array at run time. ArrowheadStorageMap computes the
/// positions, and writes the canonical index arrays of Csr and Bsr, to
/// build or check a matrix in these formats. Each storage reads by rows
/// under the row-major layout and by columns under the column-major one.
enum class ArrowheadStorage {
    Full,   ///< the full n x n matrix, zeros included: (r, c) at r * n + c, or c * n + r
    Packed, ///< Abb, the couplings A_bk, then the A_kb, then the D_k, each by rows or by columns
    Csr,    ///< compressed sparse rows, border rows first; compressed sparse columns
            ///< under the column-major layout
    Bsr     ///< block sparse rows, square blocks of size `block`; block sparse columns under
            ///< the column-major layout; needs a border multiple of the block (and a
            ///< coupling_row multiple of it under the diagonal coupling)
};



/// \brief Pattern of the border-to-block couplings A_bk, the entries of
/// the border rows in the columns of a block.
enum class ArrowheadCoupling {
    Dense,   ///< every border row may touch every unknown of every block
    Diagonal ///< unknown i of a block touches border row coupling_row + i only
};



/// \brief Condensation of an accepted block: what its Schur update and its
/// recovery data spare, flops or memory traffic
/// (ArrowheadLUppConfig::condensation).
///
/// The two differ under the row-major layout with the dense coupling only.
/// The other paths run the whole solve W_k = D_k^-1 [A_kb | y_k] either
/// way: under the diagonal coupling it already costs fewer flops than the
/// update by the multipliers, and under the column-major layout it is the
/// only path.
enum class ArrowheadCondensation {
    MinFlops,  ///< the forward stage V_k = L^-1 P [A_kb | y_k] alone, the Schur
               ///< complement updated by M V_k with the border multipliers
               ///< M = A_bk U^-1 of the pivot check; V_k in place of A_kb and
               ///< y_k, U in the upper triangle of D_k, and the backward stage,
               ///< on one column, in the recovery
    MinTraffic ///< the whole solve W_k = D_k^-1 [A_kb | y_k], the Schur complement
               ///< updated by A_bk W_k; W_k in place of A_kb and y_k, D_k left
               ///< as stored: U is neither written nor read again
};



/// \brief Shape of an arrowhead matrix: a template argument of the
/// compile-time solver, a function argument of the runtime one.
///
/// The dimension is n = border + block * blocks. ArrowheadStorageMap::valid
/// checks a shape against a storage and a coupling.
struct ArrowheadShape {
    /// Number of border unknowns, ordered first.
    int border = 1;
    /// Number of unknowns of a diagonal block.
    int block = 1;
    /// Number of diagonal blocks.
    int blocks = 1;
    /// First border row coupled to the blocks, under the diagonal coupling
    /// (rows coupling_row .. coupling_row + block - 1). It stays 0 under
    /// the dense coupling.
    int coupling_row = 0;

    /// \brief Dimension of the matrix.
    /// \return border + block * blocks
    [[nodiscard]] TDLS_HOST_DEVICE constexpr int n() const noexcept {
        return border + block * blocks;
    }
};



/// \brief Compile-time knobs of the ArrowheadLUpp solvers, carrying the
/// tuned defaults.
/// \tparam T scalar type (float, double or long double)
template<typename T>
struct ArrowheadLUppConfig {

    /// Pattern of the couplings A_bk, see ArrowheadCoupling. The dense
    /// coupling accepts every arrowhead matrix; the diagonal one reads and
    /// updates `block` border rows only.
    ArrowheadCoupling coupling = ArrowheadCoupling::Dense;

    /// Storage of the matrix, see ArrowheadStorage. Packed and Csr hold
    /// the pattern entries alone, as many in both: the least memory. Bsr
    /// holds as many under the dense coupling, whole coupling blocks under
    /// the diagonal one.
    ArrowheadStorage storage = ArrowheadStorage::Full;

    /// Order of the stored matrix: by rows or by columns, for every
    /// storage (ColMajor turns Csr into CSC and Bsr into BSC, with blocks
    /// stored by columns). The diagonal blocks are factored by TiledLUpp
    /// in the same layout, as stored, and the condensation takes the path
    /// that suits it (see solver_static.hpp).
    MatrixLayout layout = MatrixLayout::RowMajor;

    /// Threshold of the pivot check of the condensation, in [0, 1]. A block is
    /// eliminated before the border, its pivots chosen among its own rows only.
    /// Each pivot must reach this fraction of the largest candidate of its
    /// column, the border rows included, so that no multiplier exceeds 1 /
    /// condensation_pivot_threshold in magnitude. A block that fails is refused
    /// (a delayed pivot, as in multifrontal solvers) and eliminated in the
    /// second pass, where the border rows may serve as pivots. 0 refuses only
    /// the singular blocks and those whose multipliers are not finite. The
    /// second pass prefers a row of the block as pivot when it reaches this
    /// fraction of the largest candidate.
    ///
    /// On the jacobians of an implicit Berveiller-Zaoui polycrystal, 0.01
    /// refuses no block. 0.1 bounds the multipliers by 10 instead of 100,
    /// but refuses 0.075 % of the blocks under the column-major layout
    /// with a border of 12: on GPU, almost every warp then holds a system
    /// that takes the second pass (+16 % at 240 phases, RTX 5080). 0.3
    /// refuses 13 to 42 % of them.
    StructuralReal<T> condensation_pivot_threshold = T(0.01);

    /// Condensation of an accepted block, see ArrowheadCondensation; it
    /// acts under the row-major layout with the dense coupling only.
    /// MinFlops, the default, solves the forward stage alone and stores U
    /// in D_k for the recovery. MinTraffic runs the whole solve: the
    /// backward stage on B more columns, about S^2 B / 2 more
    /// multiply-adds per block, but U is neither written by the
    /// condensation nor read by the recovery. Measured with MinTraffic
    /// against MinFlops on an RTX 5080, in float and with the register
    /// budget of double: streamed, 1.46 to 2.2 times faster with blocks of
    /// 6, 1.06 to 1.5 times with blocks of 12, the same speed with blocks of
    /// 24;
    /// stored, 1.1 to 1.6 times faster with the systems one after the other
    /// (AoS), but 0.72 to 0.77 times in float with the systems interleaved
    /// (SoA) and blocks of 12. In double, streamed: 1.37 to 1.73 times in
    /// AoS, 1.53 times in SoA with blocks of 12, but 0.83 times in SoA with
    /// blocks of 6. On CPU no trend (0.5 to 1.5 times, slower on small
    /// blocks in float). The pivot check is the same, so both refuse the
    /// same blocks; the solutions differ by rounding errors.
    ArrowheadCondensation condensation = ArrowheadCondensation::MinFlops;

    /// Tile size of TiledLUpp on the diagonal blocks, when they are not
    /// eliminated in registers (block_row_interchange). Tiles of 4 beat tiles
    /// of 3 by 1.1 to 1.3 times from blocks of 14 (RTX 5080, float and the
    /// register budget of double, the three paths). TiledLUpp searches the
    /// pivot inside the tile first: a tile of the size of the block pivots
    /// over the whole block, as the physical interchanges do. On blocks that
    /// pivot, this refuses far fewer blocks (random blocks of 16, dense
    /// coupling: 0.16 refused per system with tiles of 3, 0.01 with one
    /// tile) and runs 1.3 to 3 times faster than tiles of 3 at S = 12 and 16
    /// (RTX 5080, float, the matrix stored).
    int block_tile_size = 4;

    /// Row interchanges of TiledLUpp on the diagonal blocks
    /// (TiledLUppConfig::row_interchange). Physical eliminates each block
    /// in registers: the whole block one tile, the rows interchanged as in
    /// LAPACK, every index known at compile time once the loops are
    /// unrolled, the right-hand sides [A_kb | y_k] solved the same way;
    /// the pivots of one tile of the size of the block
    /// (block_relative_pivot_threshold, block_oot_pivot_threshold;
    /// block_tile_size and gather_rhs ignored). Logical eliminates it in
    /// tiles of block_tile_size, the block on the stack of the thread on
    /// GPU, only the working tiles in registers. Auto, the default, takes
    /// Physical when the block is small enough (detail::physical_rows_fit):
    /// up to blocks of 14 in float and 10 in double on GPU, 8 in float and
    /// 12 in double on CPU, long double as double. The pivot check, the
    /// recovery data and the second pass are the same either way, and the
    /// runtime solver makes the same choice and runs the same operations,
    /// so that both solvers stay bitwise identical. Measured gains of the
    /// physical interchanges over tiles of 3, blocks that keep their rows,
    /// border of 12: 2 to 6 times on the recomputed path, 1.1 to 2.9 times
    /// streamed, 1 to 1.6 times stored on GPU (RTX 5080, float and the
    /// register budget of double), 1.5 times (median) on CPU up to these
    /// sizes. Beyond them the block no longer fits in the registers of a
    /// thread and the compilers spill it whole: 5 to 14 times slower from
    /// blocks of 16 in float on GPU; on CPU, the tiled solver vectorizes
    /// better in float.
    RowInterchange block_row_interchange = RowInterchange::Auto;

    /// Relative threshold of the in-tile pivot choice of TiledLUpp on the
    /// blocks (TiledLUppConfig::relative_pivot_threshold), in (0, 1]. The
    /// row in place keeps the pivot down to this fraction of the best
    /// in-tile candidate: on the blocks of a polycrystal no row moves, so
    /// the pivot arrays stay alike across a warp and the reads of the
    /// block copy at pivot-dependent offsets stay coalesced. The pivot
    /// check of the condensation bounds the multipliers by
    /// 1 / condensation_pivot_threshold anyway.
    StructuralReal<T> block_relative_pivot_threshold = T(0.01);

    /// Out-of-tile trigger of TiledLUpp on the blocks
    /// (TiledLUppConfig::oot_pivot_threshold), two orders below the default
    /// of TiledLUpp: the out-of-tile recovery, a divergent path, fires only
    /// on a nearly zero in-tile column.
    StructuralReal<T> block_oot_pivot_threshold = std::is_same_v<T, float> ? T(1e-6f) : T(1e-12);

    /// Tile size of TiledLUpp on the border system.
    int border_tile_size = 3;

    /// Right-hand sides [A_kb | y_k] of a block solved per pass, in
    /// [-1, border + 1] for the compile-time solver (the runtime solver
    /// takes any width from -1, a width above border + nrhs meaning one
    /// pass); 0 solves all border + 1 at once, -1 (the default) chooses:
    /// passes of 2 on GPU when the blocks are eliminated in registers, one
    /// pass otherwise. With the blocks in registers on GPU, passes of 2
    /// keep fewer values alive: 1.2 to 1.9 times faster than one pass
    /// under the row-major layout (median, RTX 5080, float and the register
    /// budget of double); under the column-major layout, from 0.8 times
    /// (blocks of 10, recomputed) to 15 times (blocks of 14, streamed, where
    /// one pass spills the block). On CPU one pass is the best (passes of
    /// 2: 0.8 times its speed). With the blocks eliminated by tiles, under
    /// the column-major layout, passes gain nothing: one pass reads the
    /// pivot check from the forward stage of the solve, several need U and
    /// a separate check. The two order the operations
    /// differently once a block spans several tiles: a block whose largest
    /// multiplier lies within rounding of 1 / condensation_pivot_threshold
    /// can pass the check in one pass and fail it in several, and the
    /// solutions then differ by rounding errors. A width of 2 or more can
    /// also give one pass to an entry point and several to another, whose
    /// right-hand sides add columns; with 0, every entry point runs one
    /// pass and the same check.
    int rhs_pass_width = -1;

    /// Permutes the right-hand sides of a block by its pivots through a
    /// predicated gather into a second buffer (true) or by cycles in place
    /// (false); used where the condensation runs the whole solve
    /// (row-major with the diagonal coupling or with MinTraffic,
    /// column-major in several passes). It changes the register allocation
    /// more than the cost of the permutation: on the RTX 5080, -15 to -20 %
    /// with a border of 12 and passes of 2, but up to +18 % with a border of
    /// 6 at 240 phases, whether the pivots move or not. To measure per shape.
    bool gather_rhs = false;

    /// Accumulates the Schur complement in place, in the border entries of
    /// A and y (true), or in a local array of the coupled border rows,
    /// alive across the sweep over the blocks (false). In place frees
    /// registers but reads and writes these rows for every block: on the
    /// RTX 5080 it costs 3 to 28 % under the diagonal coupling and saves 2
    /// to 15 % under the dense one, where the array holds the whole
    /// border; neutral on CPU.
    bool schur_in_place = false;

    /// Unroll policy, as in the other families: true, the loops over the
    /// entries of a block and of the border carry a forced-unroll pragma,
    /// the guard that keeps them in registers on GPU; false, no pragma, for
    /// faster compiles. Passed on to the TiledLUpp solvers of the blocks
    /// and of the border. Ignored by the runtime solver.
    bool unroll_loops = true;

    /// Streamed entry points of the compile-time solver kept out of line on
    /// GPU (CUDA, HIP, SYCL device code; ignored elsewhere and by the
    /// runtime solver). The
    /// provider is compiled inside the loop over the blocks, and the GPU
    /// inliner may merge the whole solve into a large caller, whose
    /// registers it then allocates worse than the two functions apart. Off,
    /// the generated code is that of a solver without the knob; on, a
    /// polycrystal of 240 grains ran 19 % faster on an RTX 5080. To measure
    /// per provider.
    bool streamed_out_of_line = false;
};



namespace detail {

/// \brief TiledLUpp configuration of the diagonal blocks of an
/// ArrowheadLUpp solver.
/// \tparam T scalar type
/// \param[in] config configuration of the ArrowheadLUpp solver
/// \return the block tile size, the two block thresholds, the layout and
///         the unroll policy of config, the other knobs at their default
template<typename T>
[[nodiscard]] constexpr TiledLUppConfig<T>
arrowhead_block_config(const ArrowheadLUppConfig<T>& config) noexcept {
    TiledLUppConfig<T> c;
    c.tile_size                = config.block_tile_size;
    c.relative_pivot_threshold = config.block_relative_pivot_threshold;
    c.oot_pivot_threshold      = config.block_oot_pivot_threshold;
    c.unroll_loops             = config.unroll_loops;
    c.layout                   = config.layout;
    return c;
}

/// \brief TiledLUpp configuration of the diagonal blocks eliminated in
/// registers: that of arrowhead_block_config with physical row
/// interchanges (the whole block one tile, every index known at compile
/// time).
/// \tparam T scalar type
/// \param[in] config configuration of the ArrowheadLUpp solver
/// \return arrowhead_block_config(config), its rows interchanged physically
template<typename T>
[[nodiscard]] constexpr TiledLUppConfig<T>
arrowhead_physical_block_config(const ArrowheadLUppConfig<T>& config) noexcept {
    TiledLUppConfig<T> c = arrowhead_block_config(config);
    c.row_interchange    = RowInterchange::Physical;
    return c;
}

/// \brief Whether the diagonal blocks of S unknowns are eliminated in
/// registers (ArrowheadLUppConfig::block_row_interchange, Auto resolved).
/// \tparam T scalar type
/// \param[in] config configuration of the ArrowheadLUpp solver
/// \param[in] S      unknowns of a block
/// \return true for the physical row interchanges
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool
arrowhead_register_blocks(const ArrowheadLUppConfig<T>& config, const int S) noexcept {
    return physical_rows<T>(config.block_row_interchange, S);
}

/// \brief The width of the passes over [A_kb | y_k]
/// (ArrowheadLUppConfig::rhs_pass_width), -1 resolved: 2 on GPU with the
/// blocks in registers, whatever the layout, 0 (one pass) otherwise.
/// \tparam T scalar type
/// \param[in] config configuration of the ArrowheadLUpp solver
/// \param[in] reg    whether the blocks are eliminated in registers
/// \return the width, 0 for one pass
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE constexpr int
arrowhead_rhs_pass_width(const ArrowheadLUppConfig<T>& config,
                         [[maybe_unused]] const bool reg) noexcept {
    if (config.rhs_pass_width >= 0) return config.rhs_pass_width;
#if TDLS_TARGET_GPU
    return reg ? 2 : 0;
#else
    return 0;
#endif
}

/// \brief Whether the diagonal blocks of S unknowns are eliminated in
/// registers, decided at run time on the side of the code
/// (detail::physical_rows_here): the choice of the runtime solver.
///
/// The runtime helpers take the fields of the configuration by value: a
/// configuration given as a template argument and bound to a reference at
/// run time needs the storage of the template parameter object on the
/// device, which nvc++ does not emit (the image is then rejected when
/// loaded).
/// \tparam T scalar type
/// \param[in] interchange ArrowheadLUppConfig::block_row_interchange
/// \param[in] S           unknowns of a block
/// \return true for the physical row interchanges
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool
arrowhead_register_blocks_here(const RowInterchange interchange, const int S) noexcept {
    return physical_rows_here<T>(interchange, S);
}

/// \brief arrowhead_rhs_pass_width decided at run time on the side of the
/// code, as arrowhead_register_blocks_here.
/// \param[in] width ArrowheadLUppConfig::rhs_pass_width
/// \param[in] reg   whether the blocks are eliminated in registers
/// \return the width, 0 for one pass
[[nodiscard]] TDLS_HOST_DEVICE constexpr int
arrowhead_rhs_pass_width_here(const int width, [[maybe_unused]] const bool reg) noexcept {
    if (width >= 0) return width;
#if defined(TDLS_DETAIL_SIDE_DISPATCH)
    if (!std::is_constant_evaluated()) return on_device_side() && reg ? 2 : 0;
#endif
#if TDLS_TARGET_GPU
    return reg ? 2 : 0;
#else
    return 0;
#endif
}

/// \brief The configuration with the choices of the given hardware made
/// explicit: block_row_interchange Physical or Logical, rhs_pass_width
/// non-negative (the delegation of the side dispatch of nvc++).
/// \tparam T   scalar type
/// \tparam gpu the hardware: GPU (true) or CPU (false)
/// \param[in] config configuration of the ArrowheadLUpp solver
/// \param[in] S      unknowns of a block
/// \return the configuration, Auto and -1 resolved
template<typename T, bool gpu>
[[nodiscard]] constexpr ArrowheadLUppConfig<T>
arrowhead_resolved_config(ArrowheadLUppConfig<T> config, const int S) noexcept {
    const bool reg               = physical_rows_on<T, gpu>(config.block_row_interchange, S);
    config.block_row_interchange = reg ? RowInterchange::Physical : RowInterchange::Logical;
    if (config.rhs_pass_width < 0) config.rhs_pass_width = gpu && reg ? 2 : 0;
    return config;
}

/// \brief TiledLUpp configuration of the border system of an
/// ArrowheadLUpp solver, a local row-major array whatever the layout.
/// \tparam T scalar type
/// \param[in] config configuration of the ArrowheadLUpp solver
/// \return the border tile size and the unroll policy of config, the
///         other knobs at their default
template<typename T>
[[nodiscard]] constexpr TiledLUppConfig<T>
arrowhead_border_config(const ArrowheadLUppConfig<T>& config) noexcept {
    TiledLUppConfig<T> c;
    c.tile_size    = config.border_tile_size;
    c.unroll_loops = config.unroll_loops;
    return c;
}

} // namespace detail



} // namespace tdls



#endif // TDLS_SOLVERS_ARROWHEAD_LUPP_CONFIG_HPP
