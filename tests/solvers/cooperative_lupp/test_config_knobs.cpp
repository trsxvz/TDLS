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
/// of threads, fused and split); a raised singular_floor turns tiny-pivot batches into
/// singular verdicts, which the default floor solves; a long double
/// configuration takes its floor as a double literal, stored exactly; and
/// the default configuration carries the documented values.
/// rows_per_thread and layout have their own bridge suites.

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
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<int N, int rows_per_thread>
void unroll_case(const int count, const std::uint64_t seed) {
    constexpr auto unrolled = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = rows_per_thread, .unroll_loops = true};
    constexpr auto rolled = tdls::CooperativeLUppConfig<double>{.rows_per_thread = rows_per_thread,
                                                                .unroll_loops    = false};
    using Unrolled        = tdls_tests::GroupRunner<double, N, unrolled, false, true, false>;
    using Rolled          = tdls_tests::GroupRunner<double, N, rolled, false, true, false>;
    const auto batch      = tdls_tests::make_batch<double>(N, count, seed, 0.5);
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

TDLS_TEST_CASE("cooperativelupp/knobs/long-double-floor-is-exact") {
    constexpr auto config = tdls::CooperativeLUppConfig<long double>{.singular_floor = 1e-300};
    using Solver          = tdls::CooperativeLUppSolverStatic<long double, 12, config>;
    TDLS_CHECK(Solver::singular_floor == static_cast<long double>(1e-300));
}

TDLS_TEST_CASE("cooperativelupp/knobs/defaults") {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{};
    TDLS_CHECK(config.rows_per_thread == 1);
    TDLS_CHECK(static_cast<double>(config.singular_floor) == std::numeric_limits<double>::min());
    TDLS_CHECK(config.unroll_loops);
    TDLS_CHECK(config.layout == tdls::MatrixLayout::RowMajor);
    using Solver = tdls::CooperativeLUppSolverStatic<double, 12>;
    TDLS_CHECK(Solver::rows_per_thread == 1 && Solver::threads_per_system == 12);
}

TDLS_TEST_MAIN
