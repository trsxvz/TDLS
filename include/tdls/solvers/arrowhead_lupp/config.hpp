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

    /// Storage of the matrix, see ArrowheadStorage. Packed holds the
    /// pattern entries alone, without index arrays: the least memory, the
    /// fastest on CPU, equal to Csr and Bsr on GPU.
    ArrowheadStorage storage = ArrowheadStorage::Full;

    /// Order of the stored matrix: by rows or by columns, for every
    /// storage (ColMajor turns Csr into CSC and Bsr into BSC, with blocks
    /// stored by columns). The diagonal blocks are factored by TiledLUpp
    /// in the same layout, as stored, and the condensation takes the path
    /// that suits it (see solver_static.hpp).
    MatrixLayout layout = MatrixLayout::RowMajor;

    /// Threshold of the pivot check of the condensation, in [0, 1]. A
    /// block is eliminated before the border, its pivots chosen among its
    /// own rows only. Each pivot must reach this fraction of the largest
    /// candidate of its column, the border rows included, so that no
    /// multiplier exceeds 1 / condensation_pivot_threshold in magnitude.
    /// A block that fails is refused (a delayed pivot, as in multifrontal
    /// solvers) and eliminated in the second pass, where the border rows
    /// may serve as pivots. 0 refuses only the singular blocks. The second
    /// pass prefers a row of the block as pivot when it reaches this
    /// fraction of the largest candidate.
    ///
    /// On the jacobians of an implicit Berveiller-Zaoui polycrystal, 0.01
    /// refuses no block. 0.1 bounds the multipliers by 10 instead of 100,
    /// but refuses 0.075 % of the blocks under the column-major layout
    /// with a border of 12: on GPU, almost every warp then holds a system
    /// that takes the second pass (+16 % at 240 phases, RTX 5080). 0.3
    /// refuses 13 to 42 % of them.
    StructuralReal<T> condensation_pivot_threshold = T(0.01);

    /// Tile size of TiledLUpp on the diagonal blocks.
    int block_tile_size = 3;

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
    /// [0, border + 1]; 0 solves all border + 1 at once. One pass is the
    /// best on CPU, where passes cost 13 to 170 %. Under the row-major
    /// layout, small passes keep fewer values alive on GPU: with a border
    /// of 12, passes of 2 are 18 to 28 % faster, 4 % with a border of 6
    /// (RTX 5080). Under the column-major layout they gain nothing: one
    /// pass reads the pivot check from the forward stage of the solve,
    /// several need U and a separate check.
    int rhs_pass_width = 0;

    /// Permutes the right-hand sides of a block by its pivots through a
    /// predicated gather into a second buffer (true) or by cycles in place
    /// (false); used where the condensation runs the whole solve
    /// (row-major with the diagonal coupling, column-major in several
    /// passes). It changes the register allocation more than the cost of
    /// the permutation: on the RTX 5080, -15 to -20 % with a border of 12
    /// and passes of 2, but up to +18 % with a border of 6 at 240 phases,
    /// whether the pivots move or not. To measure per shape.
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
