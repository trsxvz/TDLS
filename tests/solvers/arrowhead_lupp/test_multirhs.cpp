/// \file
/// \brief Bridge suite: the multi-right-hand-side entry points of
/// ArrowheadLUpp against solve_inplace, column by column.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// solve_inplace_multirhs and solve_multirhs carry nrhs right-hand sides
/// through one condensation: column m of the solution must match
/// solve_inplace on column m alone bit for bit, and so must the matrix
/// they leave. Both solvers, internal and strided right-hand sides (the
/// gaps between the strided entries untouched), the standard batch
/// (refused blocks included), passes of rhs_pass_width columns, the
/// gather and the Schur complement in place. solve_border_canonical_multirhs
/// must return the border rows of solve_inplace on the unit columns of
/// the border, equal in value (a zero may differ in sign: the canonical
/// path starts from exact zeros where solve_inplace subtracts zero
/// products), and leave the matrix as it was, bit for bit: it stores no
/// recovery data, and only schur_in_place writes the coupled border rows.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Every multi-right-hand-side entry point of both solvers on a
/// standard batch, against solve_inplace on each column.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \tparam nrhs   number of right-hand sides
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config, int nrhs>
void multirhs_case(const int count, const std::uint64_t seed) {
    using Static        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Dynamic       = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    constexpr int n     = Static::dimension;
    constexpr int B     = Shape.border;
    // Strided layout of the right-hand sides: entry i of column m at
    // i * rs + m * xs, the gaps filled with a sentinel.
    constexpr int rs = 2, xs = rs * n + 3;
    constexpr double sentinel = -12345.0;
    std::vector<double> ws1(Static::workspace_size),
        wsm(Static::template multirhs_workspace_size<nrhs>),
        wdm(static_cast<std::size_t>(Dynamic::workspace_size(Shape, nrhs)));
    std::vector<int> piv(static_cast<std::size_t>(Dynamic::pivot_size(Shape)));
    tdls_tests::UniformGenerator g(seed, 1.0);
    for (int s = 0; s < count; ++s) {
        const auto p  = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const auto M0 = tdls_tests::store_arrow<typename Static::storage_map>(p);
        // The right-hand sides: b, then random columns.
        std::vector<double> Y(static_cast<std::size_t>(n) * nrhs);
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < n; ++i)
                Y[static_cast<std::size_t>(m) * n + i] =
                    m == 0 ? p.b[static_cast<std::size_t>(i)] : g.next();
        // The reference: solve_inplace on each column.
        std::vector<double> X(Y.size()), M1;
        int r1 = 0;
        for (int m = 0; m < nrhs; ++m) {
            auto M = M0;
            std::vector<double> y(Y.begin() + static_cast<std::ptrdiff_t>(m) * n,
                                  Y.begin() + static_cast<std::ptrdiff_t>(m + 1) * n);
            int r         = 0;
            const bool ok = Static::template solve_inplace<true, true, true>(
                M.data(), 1, ws1.data(), 1, y.data(), 1, r);
            TDLS_CHECK(ok);
            std::copy(y.begin(), y.end(), X.begin() + static_cast<std::ptrdiff_t>(m) * n);
            M1 = M;
            r1 = r;
        }
        // Static, internal right-hand sides.
        {
            auto M = M0;
            auto Z = Y;
            int r  = 0;
            TDLS_CHECK((Static::template solve_inplace_multirhs<nrhs, true, true, true>(
                M.data(), 1, wsm.data(), 1, Z.data(), 1, 0, r)));
            TDLS_CHECK(r == r1);
            TDLS_CHECK(tdls_tests::same_bits(Z, X));
            TDLS_CHECK(tdls_tests::same_bits(M, M1));
        }
        // Static and dynamic, strided right-hand sides; solve_multirhs.
        std::vector<double> S(static_cast<std::size_t>(xs) * nrhs, sentinel);
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < n; ++i)
                S[static_cast<std::size_t>(i) * rs + static_cast<std::size_t>(m) * xs] =
                    Y[static_cast<std::size_t>(m) * n + i];
        auto check_strided = [&](const std::vector<double>& Z) {
            for (int m = 0; m < nrhs; ++m)
                for (int i = 0; i < n; ++i)
                    TDLS_CHECK_BITWISE(
                        &Z[static_cast<std::size_t>(i) * rs + static_cast<std::size_t>(m) * xs],
                        &X[static_cast<std::size_t>(m) * n + i], 1);
            int touched = 0;
            for (std::size_t e = 0; e < Z.size(); ++e) {
                const bool entry = e % xs < static_cast<std::size_t>(rs) * n && e % xs % rs == 0;
                if (!entry && Z[e] != sentinel) ++touched;
            }
            TDLS_CHECK(touched == 0);
        };
        {
            auto M = M0;
            auto Z = S;
            int r  = 0;
            TDLS_CHECK((Static::template solve_inplace_multirhs<nrhs, false, false, false>(
                M.data(), 1, wsm.data(), 1, Z.data(), rs, xs, r)));
            TDLS_CHECK(r == r1);
            check_strided(Z);
            TDLS_CHECK(tdls_tests::same_bits(M, M1));
        }
        {
            auto M = M0;
            std::vector<double> Z(S.size(), sentinel);
            TDLS_CHECK((Static::template solve_multirhs<nrhs, false, false, false>(
                M.data(), 1, wsm.data(), 1, S.data(), Z.data(), rs, xs)));
            check_strided(Z);
        }
        {
            auto M = M0;
            auto Z = S;
            int r  = 0;
            TDLS_CHECK(Dynamic::solve_inplace_multirhs(Shape, nrhs, M.data(), 1, piv.data(), 1,
                                                       wdm.data(), 1, Z.data(), rs, xs, r));
            TDLS_CHECK(r == r1);
            check_strided(Z);
            TDLS_CHECK(tdls_tests::same_bits(M, M1));
        }
        {
            auto M = M0;
            std::vector<double> Z(S.size(), sentinel);
            TDLS_CHECK(Dynamic::solve_multirhs(Shape, nrhs, M.data(), 1, piv.data(), 1, wdm.data(),
                                               1, S.data(), Z.data(), rs, xs));
            check_strided(Z);
        }
        // The canonical path: the border rows of the unit columns of the
        // border, from the first one and up to the last one.
        if constexpr (nrhs <= B) {
            for (const int col0 : {0, B - nrhs}) {
                std::vector<double> E(static_cast<std::size_t>(B) * nrhs);
                for (int m = 0; m < nrhs; ++m) {
                    auto M = M0;
                    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
                    y[static_cast<std::size_t>(col0 + m)] = 1.0;
                    TDLS_CHECK((Static::template solve_inplace<true, true, true>(
                        M.data(), 1, ws1.data(), 1, y.data(), 1)));
                    for (int r = 0; r < B; ++r)
                        E[static_cast<std::size_t>(m) * B + r] = y[static_cast<std::size_t>(r)];
                }
                // Nothing recovered, nothing stored: the matrix as it was,
                // but for the Schur complement in the coupled border rows
                // under schur_in_place.
                auto unchanged = [&](const std::vector<double>& M) {
                    auto E = M0;
                    if constexpr (Config.schur_in_place) {
                        constexpr int R0 = diag ? Shape.coupling_row : 0;
                        constexpr int R  = diag ? Shape.block : B;
                        for (int r = R0; r < R0 + R; ++r)
                            for (int c = 0; c < B; ++c) {
                                const auto e =
                                    static_cast<std::size_t>(Static::storage_map::abb(Shape, r, c));
                                E[e] = M[e];
                            }
                    }
                    TDLS_CHECK(tdls_tests::same_bits(M, E));
                };
                auto equal = [&](const std::vector<double>& Z, const int zr, const int zx) {
                    int wrong = 0;
                    for (int m = 0; m < nrhs; ++m)
                        for (int r = 0; r < B; ++r)
                            if (!(Z[static_cast<std::size_t>(r) * zr +
                                    static_cast<std::size_t>(m) * zx] ==
                                  E[static_cast<std::size_t>(m) * B + r]))
                                ++wrong;
                    TDLS_CHECK(wrong == 0);
                };
                {
                    auto M = M0;
                    std::vector<double> Z(static_cast<std::size_t>(B) * nrhs);
                    int r = 0;
                    TDLS_CHECK(
                        (Static::template solve_border_canonical_multirhs<nrhs, true, true, true>(
                            M.data(), 1, wsm.data(), 1, col0, Z.data(), 1, 0, r)));
                    TDLS_CHECK(r == r1);
                    equal(Z, 1, B);
                    unchanged(M);
                }
                {
                    auto M = M0;
                    std::vector<double> Z(static_cast<std::size_t>(rs) * B * nrhs);
                    TDLS_CHECK((
                        Static::template solve_border_canonical_multirhs<nrhs, false, false, false>(
                            M.data(), 1, wsm.data(), 1, col0, Z.data(), rs, rs * B)));
                    equal(Z, rs, rs * B);
                    unchanged(M);
                }
                {
                    auto M = M0;
                    std::vector<double> Z(static_cast<std::size_t>(rs) * B * nrhs);
                    int r = 0;
                    TDLS_CHECK(Dynamic::solve_border_canonical_multirhs(
                        Shape, nrhs, M.data(), 1, piv.data(), 1, wdm.data(), 1, col0, Z.data(), rs,
                        rs * B, r));
                    TDLS_CHECK(r == r1);
                    equal(Z, rs, rs * B);
                    unchanged(M);
                }
            }
        }
    }
}

/// \brief A configuration from its coupling, its layout and the knobs
/// of the right-hand-side passes, Packed storage.
/// \param[in] diag   diagonal coupling
/// \param[in] col    column-major layout
/// \param[in] width  rhs_pass_width
/// \param[in] gather gather_rhs
/// \param[in] place  schur_in_place
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> mrhs_config(const bool diag, const bool col,
                                                        const int width   = 0,
                                                        const bool gather = false,
                                                        const bool place  = false) {
    tdls::ArrowheadLUppConfig<double> c;
    c.coupling       = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.layout         = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.storage        = tdls::ArrowheadStorage::Packed;
    c.rhs_pass_width = width;
    c.gather_rhs     = gather;
    c.schur_in_place = place;
    return c;
}

constexpr tdls::ArrowheadShape polycrystal{12, 6, 10, 6};
constexpr tdls::ArrowheadShape dense_shape{8, 6, 10, 0};
constexpr tdls::ArrowheadShape iwan{6, 1, 24, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/multirhs/diagonal") {
    multirhs_case<polycrystal, mrhs_config(true, false), 1>(9, 1300100);
    multirhs_case<polycrystal, mrhs_config(true, false), 6>(9, 1300200);
    multirhs_case<polycrystal, mrhs_config(true, true), 6>(9, 1300300);
    multirhs_case<polycrystal, mrhs_config(true, true), 3>(9, 1300400);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/multirhs/dense") {
    multirhs_case<dense_shape, mrhs_config(false, false), 1>(9, 1300500);
    multirhs_case<dense_shape, mrhs_config(false, false), 5>(9, 1300600);
    multirhs_case<dense_shape, mrhs_config(false, true), 5>(9, 1300700);
    multirhs_case<iwan, mrhs_config(false, false), 6>(9, 1300800);
    multirhs_case<iwan, mrhs_config(false, true), 6>(9, 1300900);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/multirhs/knobs") {
    multirhs_case<polycrystal, mrhs_config(true, false, 2), 6>(9, 1301000);
    multirhs_case<polycrystal, mrhs_config(true, true, 2), 6>(9, 1301100);
    multirhs_case<polycrystal, mrhs_config(true, false, 0, true), 6>(9, 1301200);
    multirhs_case<polycrystal, mrhs_config(true, false, 0, false, true), 6>(9, 1301300);
    multirhs_case<dense_shape, mrhs_config(false, false, 3, false, true), 5>(9, 1301400);
    multirhs_case<dense_shape, mrhs_config(false, true, 0, true), 5>(9, 1301500);
}

TDLS_TEST_MAIN
