/// \file
/// \brief Bridge suite: each knob of ArrowheadLUppConfig changes what it
/// should and nothing else.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The performance knobs keep the arithmetic: every rhs_pass_width from 0
/// to border + 1, gather_rhs, unroll_loops off and schur_in_place give
/// the solutions and the counts of refused blocks of the default
/// configuration, bit for bit, under both layouts and both couplings, on
/// systems whose pivot checks stay clear of the threshold (under the
/// column-major layout, a pass width can move a block across it: see
/// ArrowheadLUppConfig::rhs_pass_width). schur_in_place leaves the Schur
/// complement in the border entries of A, so only the solutions compare. The thresholds change the pivots on
/// purpose, and are checked against the reference LU: the refusals grow
/// with condensation_pivot_threshold, 0 refusing only the singular
/// blocks, and every threshold of the pivot check and of the blocks
/// keeps the backward error at the level of the reference, but the
/// threshold 0, which keeps the pivot-poor blocks and their growth.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

using tdls::ArrowheadLUppConfig;
using tdls::ArrowheadShape;

/// \brief The base configuration of a cell: Packed storage, a coupling,
/// a layout.
/// \param[in] diag diagonal coupling
/// \param[in] col  column-major layout
/// \return the configuration
constexpr ArrowheadLUppConfig<double> base(const bool diag, const bool col) {
    ArrowheadLUppConfig<double> c;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.storage  = tdls::ArrowheadStorage::Packed;
    return c;
}

/// \brief Result of one solve.
struct Run {
    bool ok     = false;
    int refused = 0;
    std::vector<double> x;
};

/// \brief Solves a problem under a configuration.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in] p problem
/// \return the result
template<ArrowheadShape Shape, ArrowheadLUppConfig<double> Config>
Run run(const tdls_tests::ArrowProblem<double>& p) {
    using Solver = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    auto M       = tdls_tests::store_arrow<typename Solver::storage_map>(p);
    std::vector<double> w(Solver::workspace_size);
    Run r;
    r.x  = p.b;
    r.ok = Solver::template solve_inplace<true, true, true>(M.data(), 1, w.data(), 1, r.x.data(), 1,
                                                            r.refused);
    return r;
}

/// \brief Bitwise comparison of a variant with the base configuration on
/// a standard batch.
/// \tparam Shape   shape
/// \tparam Base    base configuration
/// \tparam Variant configuration differing by a performance knob
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<ArrowheadShape Shape, ArrowheadLUppConfig<double> Base,
         ArrowheadLUppConfig<double> Variant>
void same_case(const int count, const std::uint64_t seed) {
    constexpr bool diag = Base.coupling == tdls::ArrowheadCoupling::Diagonal;
    for (int s = 0; s < count; ++s) {
        const auto p = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const Run a = run<Shape, Base>(p), b = run<Shape, Variant>(p);
        TDLS_CHECK(a.ok && b.ok);
        TDLS_CHECK(a.refused == b.refused);
        TDLS_CHECK(tdls_tests::same_bits(a.x, b.x));
    }
}

/// \brief A configuration with one knob changed.
/// \param[in] c     configuration
/// \param[in] pass  rhs_pass_width
/// \return the configuration
constexpr ArrowheadLUppConfig<double> with_pass(ArrowheadLUppConfig<double> c, const int pass) {
    c.rhs_pass_width = pass;
    return c;
}
/// \brief The configuration with gather_rhs.
/// \param[in] c configuration
/// \return the configuration
constexpr ArrowheadLUppConfig<double> with_gather(ArrowheadLUppConfig<double> c) {
    c.gather_rhs = true;
    return c;
}
/// \brief The configuration with schur_in_place.
/// \param[in] c configuration
/// \return the configuration
constexpr ArrowheadLUppConfig<double> with_in_place(ArrowheadLUppConfig<double> c) {
    c.schur_in_place = true;
    return c;
}
/// \brief The configuration without unroll pragmas.
/// \param[in] c configuration
/// \return the configuration
constexpr ArrowheadLUppConfig<double> without_unroll(ArrowheadLUppConfig<double> c) {
    c.unroll_loops = false;
    return c;
}
/// \brief The configuration with a condensation threshold.
/// \param[in] c configuration
/// \param[in] u condensation_pivot_threshold
/// \return the configuration
constexpr ArrowheadLUppConfig<double> with_threshold(ArrowheadLUppConfig<double> c,
                                                     const double u) {
    c.condensation_pivot_threshold = u;
    return c;
}
/// \brief The configuration with block thresholds and tile sizes.
/// \param[in] c        configuration
/// \param[in] relative block_relative_pivot_threshold
/// \param[in] oot      block_oot_pivot_threshold
/// \param[in] tile     block_tile_size
/// \param[in] border   border_tile_size
/// \return the configuration
constexpr ArrowheadLUppConfig<double> with_blocks(ArrowheadLUppConfig<double> c,
                                                  const double relative, const double oot,
                                                  const int tile, const int border) {
    c.block_relative_pivot_threshold = relative;
    c.block_oot_pivot_threshold      = oot;
    c.block_tile_size                = tile;
    c.border_tile_size               = border;
    return c;
}

/// \brief Every pass width of a shape against one pass.
/// \tparam Shape shape
/// \tparam Base  base configuration
/// \tparam ws    pass widths
/// \param[in] seed generator seed
template<ArrowheadShape Shape, ArrowheadLUppConfig<double> Base, int... ws>
void passes_case(const std::uint64_t seed) {
    (same_case<Shape, Base, with_pass(Base, ws)>(6, seed + static_cast<std::uint64_t>(ws)), ...);
}

/// \brief Refusals and backward errors under a threshold configuration.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in]  count number of systems
/// \param[in]  seed  generator seed
/// \param[out] be    worst backward error
/// \return the number of refused blocks
template<ArrowheadShape Shape, ArrowheadLUppConfig<double> Config>
int refusals(const int count, const std::uint64_t seed, double& be) {
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    int refused         = 0;
    be                  = 0.0;
    for (int s = 0; s < count; ++s) {
        const auto p = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const Run r  = run<Shape, Config>(p);
        TDLS_CHECK(r.ok);
        refused += r.refused;
        be = std::max(be, tdls_tests::backward_error(p.A.data(), r.x.data(), p.b.data(), p.n()));
    }
    return refused;
}

constexpr ArrowheadShape polycrystal{12, 6, 10, 6};
constexpr ArrowheadShape dense_shape{8, 6, 10, 0};
constexpr ArrowheadShape thin{3, 6, 8, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/config_knobs/rhs_pass_width") {
    passes_case<polycrystal, base(true, false), 1, 2, 3, 5, 12, 13>(900100);
    passes_case<polycrystal, base(true, true), 1, 2, 4, 7, 13>(900200);
    passes_case<dense_shape, base(false, false), 1, 2, 4, 9>(900300);
    passes_case<dense_shape, base(false, true), 1, 3, 8, 9>(900400);
    passes_case<thin, base(false, true), 1, 2, 4>(900500);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/config_knobs/gather_rhs") {
    same_case<polycrystal, base(true, false), with_gather(base(true, false))>(9, 901100);
    same_case<polycrystal, with_pass(base(true, true), 2),
              with_gather(with_pass(base(true, true), 2))>(9, 901200);
    same_case<dense_shape, with_pass(base(false, true), 3),
              with_gather(with_pass(base(false, true), 3))>(9, 901300);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/config_knobs/schur_in_place") {
    same_case<polycrystal, base(true, false), with_in_place(base(true, false))>(9, 902100);
    same_case<polycrystal, base(true, true), with_in_place(base(true, true))>(9, 902200);
    same_case<dense_shape, base(false, false), with_in_place(base(false, false))>(9, 902300);
    same_case<dense_shape, with_pass(base(false, true), 2),
              with_in_place(with_pass(base(false, true), 2))>(9, 902400);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/config_knobs/unroll_loops") {
    same_case<polycrystal, base(true, false), without_unroll(base(true, false))>(9, 903100);
    same_case<dense_shape, base(false, true), without_unroll(base(false, true))>(9, 903200);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/config_knobs/condensation_pivot_threshold") {
    double be0 = 0, be1 = 0, be2 = 0;
    const int r0 = refusals<polycrystal, with_threshold(base(true, false), 0.0)>(12, 904100, be0);
    const int r1 = refusals<polycrystal, base(true, false)>(12, 904100, be1);
    const int r2 = refusals<polycrystal, with_threshold(base(true, false), 1.0)>(12, 904100, be2);
    // 0 refuses only the singular blocks, one per third system of the
    // batch; the pivot-poor blocks then pass, with their growth.
    TDLS_CHECK(r0 == 4);
    TDLS_CHECK(r1 >= r0 + 4);
    TDLS_CHECK(r2 >= r1);
    TDLS_CHECK_LE(be1, 1e-12);
    TDLS_CHECK_LE(be2, 1e-12);
    // The pivot-poor blocks kept at 0 grow the error, within bounds here.
    TDLS_CHECK_LE(be0, 1e-8);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/config_knobs/blocks_and_border") {
    double be = 0;
    (void)refusals<polycrystal, with_blocks(base(true, false), 1.0, 1e-12, 3, 3)>(12, 905100, be);
    TDLS_CHECK_LE(be, 1e-12);
    (void)refusals<polycrystal, with_blocks(base(true, true), 0.1, 1e-6, 2, 4)>(12, 905200, be);
    TDLS_CHECK_LE(be, 1e-12);
    (void)refusals<dense_shape, with_blocks(base(false, false), 0.5, 1e-10, 6, 8)>(12, 905300, be);
    TDLS_CHECK_LE(be, 1e-12);
    (void)refusals<thin, with_blocks(base(false, true), 0.01, 1e-12, 1, 1)>(12, 905400, be);
    TDLS_CHECK_LE(be, 1e-12);
}

TDLS_TEST_MAIN
