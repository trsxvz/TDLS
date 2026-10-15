/// \file
/// \brief Bridge suite: the documented equivalences of the ArrowheadLUpp
/// entry points.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// solve copies b into x and solves in place: its solution and the
/// matrix it leaves must match solve_inplace bit for bit, and b must come
/// back untouched. The diagnostics-free overloads must match the counting
/// ones bit for bit. Both solvers, the standard batch (second pass
/// included).

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"

namespace {

/// \brief The four entry points of both solvers on a standard batch.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void entry_case(const int count, const std::uint64_t seed) {
    using Static        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Dynamic       = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    std::vector<double> ws(Static::workspace_size),
        wd(static_cast<std::size_t>(Dynamic::workspace_size(Shape)));
    std::vector<int> piv(static_cast<std::size_t>(Dynamic::pivot_size(Shape)));
    for (int s = 0; s < count; ++s) {
        const auto p  = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const auto M0 = tdls_tests::store_arrow<typename Static::storage_map>(p);
        // Static: solve_inplace counting, the reference of the case.
        auto M1  = M0;
        auto y1  = p.b;
        int r1   = 0;
        bool ok1 = Static::template solve_inplace<true, true, true>(M1.data(), 1, ws.data(), 1,
                                                                    y1.data(), 1, r1);
        TDLS_CHECK(ok1);
        // Static: solve_inplace without the count.
        auto M2 = M0;
        auto y2 = p.b;
        TDLS_CHECK((Static::template solve_inplace<true, true, true>(M2.data(), 1, ws.data(), 1,
                                                                     y2.data(), 1)));
        // Static: solve, counting and not.
        auto M3 = M0, M4 = M0;
        auto b = p.b;
        std::vector<double> x3(p.b.size()), x4(p.b.size());
        int r3 = 0;
        TDLS_CHECK((Static::template solve<true, true, true>(M3.data(), 1, ws.data(), 1, b.data(),
                                                             x3.data(), 1, r3)));
        TDLS_CHECK((Static::template solve<true, true, true>(M4.data(), 1, ws.data(), 1, b.data(),
                                                             x4.data(), 1)));
        TDLS_CHECK(r3 == r1);
        TDLS_CHECK(tdls_tests::same_bits(b, p.b));
        // Dynamic: the same four.
        auto M5 = M0, M6 = M0, M7 = M0, M8 = M0;
        auto y5 = p.b, y6 = p.b;
        std::vector<double> x7(p.b.size()), x8(p.b.size());
        int r5 = 0, r7 = 0;
        TDLS_CHECK(Dynamic::solve_inplace(Shape, M5.data(), 1, piv.data(), 1, wd.data(), 1,
                                          y5.data(), 1, r5));
        TDLS_CHECK(
            Dynamic::solve_inplace(Shape, M6.data(), 1, piv.data(), 1, wd.data(), 1, y6.data(), 1));
        TDLS_CHECK(Dynamic::solve(Shape, M7.data(), 1, piv.data(), 1, wd.data(), 1, b.data(),
                                  x7.data(), 1, r7));
        TDLS_CHECK(Dynamic::solve(Shape, M8.data(), 1, piv.data(), 1, wd.data(), 1, b.data(),
                                  x8.data(), 1));
        TDLS_CHECK(r5 == r1 && r7 == r1);
        TDLS_CHECK(tdls_tests::same_bits(b, p.b));
        for (const auto* x : {&y2, &x3, &x4, &y5, &y6, &x7, &x8})
            TDLS_CHECK(tdls_tests::same_bits(*x, y1));
        for (const auto* M : {&M2, &M3, &M4, &M5, &M6, &M7, &M8})
            TDLS_CHECK(tdls_tests::same_bits(*M, M1));
    }
}

/// \brief A configuration from its coupling and layout, Packed storage.
/// \param[in] diag diagonal coupling
/// \param[in] col  column-major layout
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> entry_config(const bool diag, const bool col) {
    tdls::ArrowheadLUppConfig<double> c;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.storage  = tdls::ArrowheadStorage::Packed;
    return c;
}

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/entry_points/diagonal") {
    entry_case<tdls::ArrowheadShape{12, 6, 10, 6}, entry_config(true, false)>(9, 1100100);
    entry_case<tdls::ArrowheadShape{12, 6, 10, 6}, entry_config(true, true)>(9, 1100200);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/entry_points/dense") {
    entry_case<tdls::ArrowheadShape{8, 6, 10, 0}, entry_config(false, false)>(9, 1100300);
    entry_case<tdls::ArrowheadShape{6, 1, 24, 0}, entry_config(false, true)>(9, 1100400);
}

TDLS_TEST_MAIN
