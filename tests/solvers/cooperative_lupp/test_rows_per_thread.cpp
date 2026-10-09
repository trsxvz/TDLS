/// \file
/// \brief Bridge suite: the mapping of rows to threads changes no value.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Each row receives the same operations in the same order whichever
/// thread holds it, so every value of rows_per_thread must reproduce the
/// sequential path (one thread per system, anchored by the oracle suite)
/// bit for bit: factored rows, row positions and solution. Every mapping
/// of N in {1, 5, 12, 13} is covered, from one row per thread (N threads)
/// to the sequential one, full and mixed slots alike, on real groups of
/// CPU threads, through the fused and the split paths. A rows_per_thread
/// above N must act as N, and the traits of the mapping are checked. The
/// mappings are bridged under the no-pragma branch, which keeps the many
/// instantiations cheap to build; the unroll bridge of the config_knobs
/// suite ties both branches, and the other suites run the unrolled
/// groups.

#include <cstdint>
#include <utility>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief Configuration with the given number of rows per thread, under
/// the no-pragma branch.
/// \tparam rows_per_thread rows held by each thread
template<int rows_per_thread>
constexpr auto rows_config =
    tdls::CooperativeLUppConfig<double>{.rows_per_thread = rows_per_thread, .unroll_loops = false};

/// \brief Solves one system with the given mapping, external storage.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam path            entry path
/// \param[in]  A0    original matrix, contiguous row-major
/// \param[in]  b0    right-hand side
/// \param[out] A_out factored rows
/// \param[out] piv   row positions
/// \param[out] x     solution
/// \return the verdict, false as well when the threads disagree on it
template<int N, int rows_per_thread, tdls_tests::Path path>
bool solve_with(const double* A0, const double* b0, double* A_out, int* piv, double* x) {
    using Runner =
        tdls_tests::GroupRunner<double, N, rows_config<rows_per_thread>, false, false, false>;
    bool uniform  = false;
    const bool ok = Runner::template run<path>(A0, b0, A_out, piv, x, uniform);
    return ok && uniform;
}

/// \brief Compares every mapping of dimension N to the sequential path
/// on one reproducible batch.
/// \tparam N system dimension
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<int N>
void mapping_case(const int count, const std::uint64_t seed) {
    const auto batch = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    std::vector<double> A_ref(N * N), A_other(N * N);
    int piv_ref[N], piv_other[N];
    double x_ref[N], x_other[N];
    int solved   = 0;
    int compared = 0;
    for (int s = 0; s < count; ++s) {
        tdls_tests::for_paths<tdls_tests::Path::fused, tdls_tests::Path::split>([&](auto path_tag) {
            constexpr auto path = decltype(path_tag)::value;
            if (!solve_with<N, N, path>(batch.matrix(s), batch.rhs(s), A_ref.data(), piv_ref,
                                        x_ref))
                return;
            ++solved;
            [&]<int... Ks>(std::integer_sequence<int, Ks...>) {
                (
                    [&] {
                        constexpr int rows_per_thread = Ks + 1;
                        const bool ok                 = solve_with<N, rows_per_thread, path>(
                            batch.matrix(s), batch.rhs(s), A_other.data(), piv_other, x_other);
                        TDLS_CHECK(ok);
                        TDLS_CHECK_BITWISE(A_ref.data(), A_other.data(),
                                           static_cast<std::size_t>(N) * N);
                        TDLS_CHECK_BITWISE(piv_ref, piv_other, static_cast<std::size_t>(N));
                        TDLS_CHECK_BITWISE(x_ref, x_other, static_cast<std::size_t>(N));
                        ++compared;
                    }(),
                    ...);
            }(std::make_integer_sequence<int, N + 2>{});
        });
    }
    // Floor: every system solved on both paths, and every mapping
    // compared (rows_per_thread = 1 .. N + 2, the last two above N).
    TDLS_CHECK(solved == 2 * count);
    TDLS_CHECK(compared == 2 * count * (N + 2));
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/bridge/rows-per-thread/double/N=1") {
    mapping_case<1>(50, 620100);
}
TDLS_TEST_CASE("cooperativelupp/bridge/rows-per-thread/double/N=5") {
    mapping_case<5>(40, 620500);
}
TDLS_TEST_CASE("cooperativelupp/bridge/rows-per-thread/double/N=12") {
    mapping_case<12>(20, 621200);
}
TDLS_TEST_CASE("cooperativelupp/bridge/rows-per-thread/double/N=13") {
    mapping_case<13>(20, 621300);
}

TDLS_TEST_CASE("cooperativelupp/bridge/rows-per-thread/traits") {
    using S12_1  = tdls::CooperativeLUppSolverStatic<double, 12, rows_config<1>>;
    using S12_5  = tdls::CooperativeLUppSolverStatic<double, 12, rows_config<5>>;
    using S12_12 = tdls::CooperativeLUppSolverStatic<double, 12, rows_config<12>>;
    using S12_20 = tdls::CooperativeLUppSolverStatic<double, 12, rows_config<20>>;
    using S13_5  = tdls::CooperativeLUppSolverStatic<double, 13, rows_config<5>>;
    TDLS_CHECK(S12_1::rows_per_thread == 1 && S12_1::threads_per_system == 12);
    TDLS_CHECK(S12_5::rows_per_thread == 5 && S12_5::threads_per_system == 3);
    TDLS_CHECK(S12_12::rows_per_thread == 12 && S12_12::threads_per_system == 1);
    TDLS_CHECK(S12_20::rows_per_thread == 12 && S12_20::threads_per_system == 1);
    TDLS_CHECK(S13_5::rows_per_thread == 5 && S13_5::threads_per_system == 3);
    TDLS_CHECK(S12_1::workspace_size == 36 && S13_5::workspace_size == 39);
    // Slot classification of N = 13 on 3 threads of 5 rows: slots 0 .. 3
    // are full (rows up to 11), slot 4 is mixed (row 12 for thread 0 only).
    TDLS_CHECK(S13_5::slot_is_full(3) && !S13_5::slot_is_full(4));
    TDLS_CHECK(S13_5::slot_has_rows(4));
    // N = 12 on 3 threads of 5 rows: slot 4 holds no real row at all.
    TDLS_CHECK(S12_5::slot_is_full(3) && !S12_5::slot_has_rows(4));
}

TDLS_TEST_MAIN
