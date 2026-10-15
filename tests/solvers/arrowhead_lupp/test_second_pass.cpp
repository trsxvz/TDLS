/// \file
/// \brief Anchor suite of the second pass of ArrowheadLUpp: the refused
/// blocks, eliminated with the border rows.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A block that fails the pivot check is never a failure: the second
/// pass eliminates it together with the border rows, a border row taking
/// a pivot when the block has none, and the solver succeeds whenever the
/// full matrix is invertible. The cases push that pass to its limits on
/// every shape of the grid, both patterns, both layouts, both solvers:
/// singular blocks with their zero pivot at the start, the middle or the
/// end, one block with two zero pivots, as many singular blocks as the
/// couplings have rank (the matrix stays invertible), one more (the
/// matrix becomes singular and both solvers must say so), and every block
/// refused at once by strong couplings. On the invertible matrices the
/// reference LU succeeds too and the backward errors stay at its level;
/// the solvers agree bit for bit, and the counts of refused blocks are
/// checked where they are known. Two 2 x 2 matrices put a subnormal pivot
/// in the second pass.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

using tdls::ArrowheadShape;

/// \brief Solves one problem with both solvers and checks it against the
/// reference LU.
/// \tparam Shape shape
/// \tparam diag  diagonal coupling (and pattern)
/// \tparam col   column-major layout
/// \param[in] p               problem
/// \param[in] expect_singular whether the matrix is singular
/// \param[in] expect_refused  expected refused blocks, -1 when unknown
template<ArrowheadShape Shape, bool diag, bool col>
void check_problem(const tdls_tests::ArrowProblem<double>& p, const bool expect_singular,
                   const int expect_refused) {
    constexpr auto config = [] {
        tdls::ArrowheadLUppConfig<double> c;
        c.block_row_interchange = tdls::RowInterchange::Logical;
        c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
        c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
        c.storage  = tdls::ArrowheadStorage::Packed;
        return c;
    }();
    using Static  = tdls::ArrowheadLUppSolverStatic<double, Shape, config>;
    using Dynamic = tdls::ArrowheadLUppSolverDynamic<double, config>;
    const int n   = Shape.n();
    std::vector<double> ws(Static::workspace_size),
        wd(static_cast<std::size_t>(Dynamic::workspace_size(Shape))), xs(n), xd(n), xr(n);
    std::vector<int> piv(static_cast<std::size_t>(Dynamic::pivot_size(Shape))), pivr(n);
    auto Ms = tdls_tests::store_arrow<typename Static::storage_map>(p);
    auto Md = Ms;
    auto Ar = p.A;
    int rs = 0, rd = 0;
    const bool oks = Static::template solve<true, true, true>(Ms.data(), 1, ws.data(), 1,
                                                              p.b.data(), xs.data(), 1, rs);
    const bool okd = Dynamic::solve(Shape, Md.data(), 1, piv.data(), 1, wd.data(), 1, p.b.data(),
                                    xd.data(), 1, rd);
    const bool okr = tdls_tests::reference_solve(Ar.data(), pivr.data(), p.b.data(), xr.data(), n);
    // A singular matrix of the cases is structurally singular: the
    // solvers find an exactly zero pivot column, while the rounding of the
    // reference may leave it a tiny pivot. The reference verdict is
    // required on the invertible matrices only.
    if (!expect_singular) TDLS_CHECK(okr);
    TDLS_CHECK(oks == !expect_singular);
    TDLS_CHECK(okd == !expect_singular);
    TDLS_CHECK(rs == rd);
    if (expect_refused >= 0) TDLS_CHECK(rs == expect_refused);
    if (oks && okr) {
        const double be  = tdls_tests::backward_error(p.A.data(), xs.data(), p.b.data(), n);
        const double ber = tdls_tests::backward_error(p.A.data(), xr.data(), p.b.data(), n);
        TDLS_CHECK_LE(be, std::max(1e-13, 100 * ber));
        TDLS_CHECK(tdls_tests::same_bits(xs, xd));
    }
}

/// \brief The cases of one shape and pattern, under one layout.
/// \tparam Shape shape
/// \tparam diag  diagonal coupling (and pattern)
/// \tparam col   column-major layout
/// \param[in] seed generator seed
template<ArrowheadShape Shape, bool diag, bool col>
void shape_cases(const std::uint64_t seed) {
    constexpr int S = Shape.block, NB = Shape.blocks;
    constexpr int R = diag ? S : Shape.border; // rank of the couplings
    using tdls_tests::ArrowBlocks;
    using tdls_tests::make_arrow_problem;
    using tdls_tests::zero_block_unknown;
    {
        // Zero pivots at the start, the middle and the end of three
        // blocks, each index used once (under the diagonal coupling, two
        // blocks singular on the same index make the matrix singular).
        auto p          = make_arrow_problem<double>(Shape, diag, ArrowBlocks::Dominant, seed);
        const int at[3] = {0, S / 2, S - 1};
        int used        = 0;
        for (int t = 0; t < 3 && used < R && t < NB; ++t) {
            if (t > 0 && at[t] == at[t - 1]) continue;
            zero_block_unknown(p, (t * NB) / 3, at[t]);
            ++used;
        }
        check_problem<Shape, diag, col>(p, false, used);
    }
    if constexpr (S >= 3 && R >= 2) {
        auto p = make_arrow_problem<double>(Shape, diag, ArrowBlocks::Dominant, seed + 1);
        zero_block_unknown(p, NB / 2, 0);
        zero_block_unknown(p, NB / 2, S - 1);
        check_problem<Shape, diag, col>(p, false, 1);
    }
    if constexpr (NB >= R) {
        // R singular blocks, each on its own index modulo S: the nullity
        // reaches the rank of the couplings, the matrix stays invertible.
        auto p = make_arrow_problem<double>(Shape, diag, ArrowBlocks::Dominant, seed + 2);
        for (int t = 0; t < R; ++t)
            zero_block_unknown(p, (t * NB) / R, t % S);
        check_problem<Shape, diag, col>(p, false, -1);
    }
    if constexpr (NB >= R + 1) {
        // R + 1 singular blocks: the nullity exceeds the rank of the
        // couplings, the matrix is singular.
        auto p = make_arrow_problem<double>(Shape, diag, ArrowBlocks::Dominant, seed + 3);
        for (int t = 0; t <= R; ++t)
            zero_block_unknown(p, (t * NB) / (R + 1), t % S);
        check_problem<Shape, diag, col>(p, true, -1);
    }
    {
        // Strong couplings: every block refused.
        auto p = make_arrow_problem<double>(Shape, diag, ArrowBlocks::Dominant, seed + 4);
        tdls_tests::strong_couplings(p, 1e5);
        check_problem<Shape, diag, col>(p, false, NB);
    }
}

/// \brief A subnormal pivot in the second pass. The block of
/// [[1, 1e-307], [1, 3e-309]], of solution (1, 0), is subnormal and
/// refused; the border row, of normal pivot, takes the pivot. The block of
/// diag(1, 3e-309) is subnormal and uncoupled: no pivot reaches the
/// singular floor of TiledLUpp, and the solvers find the matrix singular,
/// as TiledLUpp does.
/// \tparam diag diagonal coupling (and pattern)
/// \tparam col  column-major layout
template<bool diag, bool col>
void subnormal_cases() {
    constexpr ArrowheadShape shape{1, 1, 1, 0};
    const tdls_tests::ArrowProblem<double> coupled{
        shape, diag, {1.0, 1e-307, 1.0, 3e-309}, {1.0, 1.0}};
    check_problem<shape, diag, col>(coupled, false, 1);
    const tdls_tests::ArrowProblem<double> uncoupled{
        shape, diag, {1.0, 0.0, 0.0, 3e-309}, {1.0, 1.0}};
    check_problem<shape, diag, col>(uncoupled, true, -1);
}

} // namespace

/// Emits the row-major and column-major cases of one shape and pattern.
#define TDLS_ARROW_SECOND_PASS_CASES(B, S, NB, B0, DIAG, NAME, SEED)                               \
    TDLS_TEST_CASE("arrowheadlupp/oracle/second_pass/B=" #B ",S=" #S ",NB=" #NB "," NAME ",row") { \
        shape_cases<ArrowheadShape{B, S, NB, B0}, DIAG, false>(SEED);                              \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/second_pass/B=" #B ",S=" #S ",NB=" #NB "," NAME ",col") { \
        shape_cases<ArrowheadShape{B, S, NB, B0}, DIAG, true>(SEED + 10);                          \
    }

TDLS_ARROW_SECOND_PASS_CASES(1, 1, 12, 0, true, "diagonal", 1000100)
TDLS_ARROW_SECOND_PASS_CASES(1, 1, 12, 0, false, "dense", 1000200)
TDLS_ARROW_SECOND_PASS_CASES(2, 1, 20, 1, true, "diagonal", 1000300)
TDLS_ARROW_SECOND_PASS_CASES(3, 6, 8, 0, false, "dense", 1000400)
TDLS_ARROW_SECOND_PASS_CASES(8, 6, 10, 2, true, "diagonal", 1000500)
TDLS_ARROW_SECOND_PASS_CASES(8, 6, 10, 0, false, "dense", 1000600)
TDLS_ARROW_SECOND_PASS_CASES(6, 12, 5, 0, false, "dense", 1000700)
TDLS_ARROW_SECOND_PASS_CASES(12, 4, 9, 4, true, "diagonal", 1000800)
TDLS_ARROW_SECOND_PASS_CASES(12, 4, 9, 0, false, "dense", 1000900)
TDLS_ARROW_SECOND_PASS_CASES(12, 6, 30, 6, true, "diagonal", 1001000)
TDLS_ARROW_SECOND_PASS_CASES(12, 6, 30, 0, false, "dense", 1001100)

TDLS_TEST_CASE("arrowheadlupp/oracle/second_pass/subnormal_pivots") {
    subnormal_cases<false, false>();
    subnormal_cases<false, true>();
    subnormal_cases<true, false>();
    subnormal_cases<true, true>();
}

TDLS_TEST_MAIN
