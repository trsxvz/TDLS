/// \file
/// \brief Suite of the CooperativeLUppConfig knobs.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Each compile-time knob is checked through its observable contract:
/// unroll_loops never changes any value (bitwise equivalence of
/// both settings, on the sequential path of a small system and on a group
/// of threads, fused and split, under both row interchanges); a relative
/// pivot threshold keeps the row in place exactly when it reaches its
/// fraction of the column maximum and the singularity floor; a raised
/// singular_floor turns tiny-pivot batches into singular verdicts, which
/// the default floor solves; a long double configuration takes its
/// floating-point knobs as double literals, stored exactly; and the
/// default configuration carries the documented values. rows_per_thread,
/// row_interchange and layout have their own bridge suites.

#include <cstdint>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief Compares both unroll settings on one reproducible batch.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam interchange     row interchanges
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<int N, int rows_per_thread,
         tdls::RowInterchange interchange = tdls::RowInterchange::Logical>
void unroll_case(const int count, const std::uint64_t seed) {
    constexpr auto unrolled = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = rows_per_thread, .row_interchange = interchange, .unroll_loops = true};
    constexpr auto rolled = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = rows_per_thread, .row_interchange = interchange, .unroll_loops = false};
    using Unrolled   = tdls_tests::GroupRunner<double, N, unrolled, false, true, false>;
    using Rolled     = tdls_tests::GroupRunner<double, N, rolled, false, true, false>;
    const auto batch = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    std::vector<double> A_u(N * N), A_r(N * N);
    int piv_u[N], piv_r[N];
    double x_u[N], x_r[N];
    int compared = 0;
    for (int s = 0; s < count; ++s) {
        tdls_tests::for_paths<tdls_tests::Path::fused, tdls_tests::Path::split>([&](auto tag) {
            constexpr auto path = decltype(tag)::value;
            bool uniform_u = false, uniform_r = false;
            const bool ok_u = Unrolled::template run<path>(batch.matrix(s), batch.rhs(s),
                                                           A_u.data(), piv_u, x_u, uniform_u);
            const bool ok_r = Rolled::template run<path>(batch.matrix(s), batch.rhs(s), A_r.data(),
                                                         piv_r, x_r, uniform_r);
            TDLS_CHECK(ok_u && ok_r && uniform_u && uniform_r);
            TDLS_CHECK_BITWISE(A_u.data(), A_r.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_u, piv_r, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(x_u, x_r, static_cast<std::size_t>(N));
            ++compared;
        });
    }
    // Floor: both paths compared on every system.
    TDLS_CHECK(compared == 2 * count);
}

/// \brief Solves one 4 x 4 system on a group of two threads through the
/// fused path and returns the pivot entry of row 0: 0 when the row in
/// place kept the first pivot, 1 when row 1 took it. Both row
/// interchanges record the exchange of rows 0 and 1 the same way.
/// \tparam Config solver configuration
/// \param[in]  A0 matrix, contiguous row-major
/// \param[in]  b0 right-hand side
/// \param[out] ok verdict, uniform across the group
/// \return the pivot entry of row 0
template<tdls::CooperativeLUppConfig<double> Config>
int first_pivot(const double* A0, const double* b0, bool& ok) {
    using Runner = tdls_tests::GroupRunner<double, 4, Config, false, false, false>;
    std::vector<double> A_out(16), x(4);
    int piv[4];
    bool uniform = false;
    ok =
        Runner::template run<tdls_tests::Path::fused>(A0, b0, A_out.data(), piv, x.data(), uniform);
    ok = ok && uniform;
    return piv[0];
}

/// \brief Configuration of a group of two threads with the given
/// relative threshold and singularity floor, built by member assignment.
/// \param[in] threshold relative pivot threshold
/// \param[in] floor     singularity floor
/// \return the configuration, under logical row interchanges
constexpr tdls::CooperativeLUppConfig<double>
threshold_config(const double threshold, const double floor = std::numeric_limits<double>::min()) {
    tdls::CooperativeLUppConfig<double> config{};
    config.rows_per_thread          = 2;
    config.relative_pivot_threshold = threshold;
    config.singular_floor           = floor;
    config.unroll_loops             = false;
    return config;
}

/// \brief The given configuration under physical row interchanges.
/// \param[in] config configuration under logical row interchanges
/// \return the same configuration, with physical row interchanges
constexpr tdls::CooperativeLUppConfig<double> physical(tdls::CooperativeLUppConfig<double> config) {
    config.row_interchange = tdls::RowInterchange::Physical;
    return config;
}

/// \brief Checks the pivot choice of one configuration under both row
/// interchanges.
/// \tparam Config solver configuration, under logical row interchanges
/// \param[in] A0     matrix, contiguous row-major
/// \param[in] b0     right-hand side
/// \param[in] expect expected pivot entry of row 0
template<tdls::CooperativeLUppConfig<double> Config>
void check_first_pivot(const double* A0, const double* b0, const int expect) {
    bool ok_l = false, ok_p = false;
    TDLS_CHECK(first_pivot<Config>(A0, b0, ok_l) == expect);
    TDLS_CHECK(first_pivot<physical(Config)>(A0, b0, ok_p) == expect);
    TDLS_CHECK(ok_l && ok_p);
}

/// \brief Solves a batch of tiny-entry systems under the given floor and
/// counts the singular verdicts.
/// \tparam Config solver configuration
/// \param[in] batch the systems
/// \return the number of singular verdicts
template<tdls::CooperativeLUppConfig<double> Config>
int singular_count(const tdls_tests::SystemBatch<double>& batch) {
    using Runner = tdls_tests::GroupRunner<double, 12, Config, false, false, false>;
    std::vector<double> A_out(144), x(12);
    int piv[12];
    int singular = 0;
    for (int s = 0; s < batch.count; ++s) {
        bool uniform = false;
        if (!Runner::template run<tdls_tests::Path::fused>(batch.matrix(s), batch.rhs(s),
                                                           A_out.data(), piv, x.data(), uniform))
            ++singular;
        TDLS_CHECK(uniform);
    }
    return singular;
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/knobs/unroll/double/N=7,rows_per_thread=7") {
    unroll_case<7, 7>(100, 670707);
}
TDLS_TEST_CASE("cooperativelupp/knobs/unroll/double/N=8,rows_per_thread=3") {
    unroll_case<8, 3>(30, 670803);
}
TDLS_TEST_CASE("cooperativelupp/knobs/unroll/double/N=7,rows_per_thread=7,physical") {
    unroll_case<7, 7, tdls::RowInterchange::Physical>(100, 670717);
}
TDLS_TEST_CASE("cooperativelupp/knobs/unroll/double/N=8,rows_per_thread=3,physical") {
    unroll_case<8, 3, tdls::RowInterchange::Physical>(30, 670813);
}

TDLS_TEST_CASE("cooperativelupp/knobs/relative-pivot-threshold") {
    // Column 0: the row in place holds 0.5, row 1 the maximum 1. The other
    // columns are diagonally dominant, and keep their pivots in place.
    const double A0[16] = {0.5, 0.1, 0.0, 0.0, 1.0, 4.0, 0.1, 0.0,
                           0.1, 0.0, 4.0, 0.1, 0.0, 0.1, 0.0, 4.0};
    const double b0[4]  = {1.0, 2.0, 3.0, 4.0};
    check_first_pivot<threshold_config(1.0)>(A0, b0, 1);  // LAPACK: the maximum wins
    check_first_pivot<threshold_config(0.6)>(A0, b0, 1);  // 0.5 below 0.6 * 1
    check_first_pivot<threshold_config(0.5)>(A0, b0, 0);  // 0.5 reaches 0.5 * 1: kept
    check_first_pivot<threshold_config(0.25)>(A0, b0, 0); // kept a fortiori
}

TDLS_TEST_CASE("cooperativelupp/knobs/relative-pivot-threshold-below-floor") {
    // The row in place holds 5e-4, above 0.001 times the maximum 0.1 but
    // below the floor 1e-3: it must not keep the pivot, which goes to row 1.
    const double A0[16] = {5e-4, 0.1, 0.0, 0.0, 0.1, 4.0, 0.1, 0.0,
                           0.0,  0.1, 4.0, 0.1, 0.0, 0.0, 0.1, 4.0};
    const double b0[4]  = {1.0, 2.0, 3.0, 4.0};
    check_first_pivot<threshold_config(0.001, 1e-3)>(A0, b0, 1);
}

TDLS_TEST_CASE("cooperativelupp/knobs/singular-floor") {
    // Entries of magnitude at most 1e-6: every pivot falls below a 1e-3
    // floor, none below the default one.
    const auto batch      = tdls_tests::make_batch<double>(12, 50, 670400, 1e-6);
    constexpr auto raised = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = 3, .singular_floor = 1e-3, .unroll_loops = false};
    constexpr auto standard =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3, .unroll_loops = false};
    TDLS_CHECK(singular_count<raised>(batch) == batch.count);
    TDLS_CHECK(singular_count<standard>(batch) == 0);
}

TDLS_TEST_CASE("cooperativelupp/knobs/long-double-knobs-are-exact") {
    constexpr auto config = tdls::CooperativeLUppConfig<long double>{
        .relative_pivot_threshold = 0.1, .singular_floor = 1e-300};
    using Solver  = tdls::CooperativeLUppSolverStatic<long double, 12, config>;
    using Dynamic = tdls::CooperativeLUppSolverDynamic<long double, config>;
    TDLS_CHECK(Solver::relative_pivot_threshold == static_cast<long double>(0.1));
    TDLS_CHECK(Solver::singular_floor == static_cast<long double>(1e-300));
    TDLS_CHECK(Dynamic::relative_pivot_threshold == static_cast<long double>(0.1));
}

TDLS_TEST_CASE("cooperativelupp/knobs/defaults") {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{};
    TDLS_CHECK(config.rows_per_thread == 1);
    TDLS_CHECK(config.row_interchange == tdls::RowInterchange::Logical);
    TDLS_CHECK(static_cast<double>(config.relative_pivot_threshold) == 0.1);
    TDLS_CHECK(static_cast<double>(config.singular_floor) == std::numeric_limits<double>::min());
    TDLS_CHECK(config.unroll_loops);
    TDLS_CHECK(config.layout == tdls::MatrixLayout::RowMajor);
    using Solver = tdls::CooperativeLUppSolverStatic<double, 12>;
    TDLS_CHECK(Solver::rows_per_thread == 1 && Solver::threads_per_system == 12);
    TDLS_CHECK(Solver::relative_pivot_threshold == 0.1 && Solver::workspace_size == 3 * 12 + 2);
}

TDLS_TEST_MAIN
