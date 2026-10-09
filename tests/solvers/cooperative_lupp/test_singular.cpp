/// \file
/// \brief Suite of the singularity verdicts.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The solver declares a matrix singular only when the best pivot of a
/// column is zero or subnormal (below numeric_limits::min()), the
/// criterion of the reference solver and of the TiledLUpp solvers. On a
/// set of structurally singular constructions (zero column at several
/// positions, zero row, all-zero matrix, fully subnormal matrix) and on a
/// solvable extreme case (uniformly tiny but normal entries), the verdicts
/// of solve(), solve_inplace() and factorize() of both solvers must match
/// the reference and be uniform across the threads of the group, on the
/// sequential path, on a full group and on a group with a mixed slot. The verdict is
/// returned after a full sequence of barriers, never by an early exit, so
/// every path also runs to completion on a singular matrix.

#include <algorithm>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

constexpr int N = 12;

/// \brief Checks the verdicts of every path under one mapping.
/// \tparam rows_per_thread rows held by each thread
/// \param[in] A0       matrix, contiguous row-major
/// \param[in] b0       right-hand side
/// \param[in] solvable expected verdict
template<int rows_per_thread>
void check_mapping(const double* A0, const double* b0, const bool solvable) {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = rows_per_thread,
        .unroll_loops    = tdls_tests::test_unroll<N, rows_per_thread>};
    using Runner = tdls_tests::GroupRunner<double, N, config, false, false, false>;
    std::vector<double> A_out(N * N), x(N);
    std::vector<int> piv(N);
    tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::fused,
                          tdls_tests::Path::split>([&](auto tag) {
        constexpr auto path = decltype(tag)::value;
        bool uniform        = false;
        const bool ok =
            Runner::template run<path>(A0, b0, A_out.data(), piv.data(), x.data(), uniform);
        TDLS_CHECK(uniform);
        TDLS_CHECK(ok == solvable);
    });
}

/// \brief Checks the verdicts of every path of the dynamic solver under
/// one mapping.
/// \tparam rows_per_thread rows held by each thread
/// \param[in] A0       matrix, contiguous row-major
/// \param[in] b0       right-hand side
/// \param[in] solvable expected verdict
template<int rows_per_thread>
void check_mapping_dynamic(const double* A0, const double* b0, const bool solvable) {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = rows_per_thread};
    using Runner          = tdls_tests::DynamicGroupRunner<double, config>;
    std::vector<double> A_out(N * N), x(N);
    std::vector<int> piv(N);
    tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::fused,
                          tdls_tests::Path::split>([&](auto tag) {
        constexpr auto path = decltype(tag)::value;
        bool uniform        = false;
        const bool ok =
            Runner::template run<path>(N, A0, b0, A_out.data(), piv.data(), x.data(), uniform);
        TDLS_CHECK(uniform);
        TDLS_CHECK(ok == solvable);
    });
}

/// \brief Runs every mapping of both solvers and the reference on one
/// contiguous system and checks that every verdict matches the
/// expectation.
/// \param[in] A0       matrix, contiguous row-major
/// \param[in] b0       right-hand side
/// \param[in] solvable expected verdict
void check_verdicts(const double* A0, const double* b0, const bool solvable) {
    check_mapping<N>(A0, b0, solvable);
    check_mapping<3>(A0, b0, solvable);
    check_mapping<5>(A0, b0, solvable);
    check_mapping_dynamic<N>(A0, b0, solvable);
    check_mapping_dynamic<3>(A0, b0, solvable);
    check_mapping_dynamic<5>(A0, b0, solvable);

    std::vector<double> A(A0, A0 + N * N), xr(N);
    std::vector<int> piv(N);
    const bool ok_ref = tdls_tests::reference_solve(A.data(), piv.data(), b0, xr.data(), N);
    TDLS_CHECK(ok_ref == solvable);
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/singular/zero-column-first") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 660100, 0.5);
    tdls_tests::zero_column(batch, 0, 0);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("cooperativelupp/singular/zero-column-middle") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 660200, 0.5);
    tdls_tests::zero_column(batch, 0, 5);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("cooperativelupp/singular/zero-column-last") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 660300, 0.5);
    tdls_tests::zero_column(batch, 0, N - 1);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("cooperativelupp/singular/zero-row") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 660400, 0.5);
    for (int j = 0; j < N; ++j)
        batch.matrix(0)[3 * N + j] = 0.0;
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("cooperativelupp/singular/all-zero-matrix") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 660500, 0.5);
    std::fill(batch.matrix(0), batch.matrix(0) + N * N, 0.0);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("cooperativelupp/singular/subnormal-matrix") {
    // Every entry is subnormal, so every pivot candidate is below the
    // singularity floor (MAGMA's exact zero test would accept them).
    auto batch = tdls_tests::make_batch<double>(N, 1, 660600, 1e-310);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("cooperativelupp/singular/tiny-but-normal-matrix-is-solvable") {
    // Uniformly tiny normal entries stay well above the floor: a small
    // scale is not a singularity.
    auto batch = tdls_tests::make_batch<double>(N, 1, 660700, 1e-30);
    check_verdicts(batch.matrix(0), batch.rhs(0), true);
}

TDLS_TEST_MAIN
