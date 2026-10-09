/// \file
/// \brief Bridge suite: the dynamic CooperativeLUpp solver reproduces the
/// static CooperativeLUpp solver bitwise.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// At equal shape (n = N, same configuration) and on identical inputs,
/// CooperativeLUppSolverDynamic and CooperativeLUppSolverStatic execute
/// the same arithmetic sequence: the factored rows, the row positions and
/// the solution must be bitwise identical, through every entry path, and
/// the singularity verdicts must agree (one structurally singular system
/// is injected in every batch; its unspecified outputs are not compared).
/// The grid crosses the structural cases of the mapping: the scalar
/// corner, one thread per system, rows_per_thread above the dimension,
/// full groups, a mixed slot and one row per thread, on real groups of
/// CPU threads. The dynamic solver ignores unroll_loops: under
/// both settings it must reproduce the static solver, which honours it.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief Solves every system of a reproducible batch with both solvers
/// through every entry path and checks the bitwise equality of all
/// outputs.
/// \tparam T          scalar type
/// \tparam N          system dimension
/// \tparam Config     configuration of the static solver
/// \tparam DynConfig  configuration of the dynamic solver
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, tdls::CooperativeLUppConfig<T> Config,
         tdls::CooperativeLUppConfig<T> DynConfig = Config>
void bridge_case(const int count, const double bound, const std::uint64_t seed) {
    using Static  = tdls_tests::GroupRunner<T, N, Config, false, false, false>;
    using Dynamic = tdls_tests::DynamicGroupRunner<T, DynConfig>;
    auto batch    = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::zero_column(batch, 0, N - 1);

    std::vector<T> A_s(N * N), A_d(N * N);
    int piv_s[N], piv_d[N];
    T x_s[N], x_d[N];
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::split,
                              tdls_tests::Path::fused, tdls_tests::Path::split_inplace,
                              tdls_tests::Path::canonical>([&](auto tag) {
            constexpr auto path = decltype(tag)::value;
            const int col       = s % N;
            bool uniform_s      = false;
            bool uniform_d      = false;
            const bool ok_s = Static::template run<path>(batch.matrix(s), batch.rhs(s), A_s.data(),
                                                         piv_s, x_s, uniform_s, col);
            const bool ok_d = Dynamic::template run<path>(N, batch.matrix(s), batch.rhs(s),
                                                          A_d.data(), piv_d, x_d, uniform_d, col);
            TDLS_CHECK(uniform_s && uniform_d);
            TDLS_CHECK(ok_s == ok_d);
            if (!ok_s || !ok_d) return;
            ++solved;
            TDLS_CHECK_BITWISE(A_s.data(), A_d.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_s, piv_d, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(x_s, x_d, static_cast<std::size_t>(N));
        });
    }
    // Floor: every system but the singular one solved, on all five paths.
    TDLS_CHECK(solved == 5 * (count - 1));
}

/// \brief Configuration with the given number of rows per thread and
/// layout.
/// \tparam T               scalar type
/// \tparam rows_per_thread rows held by each thread
/// \tparam layout          matrix layout
/// \tparam unroll          unroll policy
template<typename T, int rows_per_thread, tdls::MatrixLayout layout = tdls::MatrixLayout::RowMajor,
         bool unroll = true>
constexpr auto config = tdls::CooperativeLUppConfig<T>{
    .rows_per_thread = rows_per_thread, .unroll_loops = unroll, .layout = layout};

/// \brief Configuration with the given number of rows per thread, under
/// the unroll policy of the test configurations (see
/// tdls_tests::test_unroll).
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
template<typename T, int N, int rows_per_thread>
constexpr auto test_config = config<T, rows_per_thread, tdls::MatrixLayout::RowMajor,
                                    tdls_tests::test_unroll<N, rows_per_thread>>;

} // namespace

/// Emits the default and tiny-entry bridge cases of one (type, N,
/// rows_per_thread) cell.
#define TDLS_BRIDGE_CASES(T, N, ROWS, COUNT, SEED)                                                 \
    TDLS_TEST_CASE("cooperativelupp/bridge/static-vs-dynamic/" #T "/N=" #N                         \
                   ",rows_per_thread=" #ROWS ",default") {                                         \
        bridge_case<T, N, test_config<T, N, ROWS>>(COUNT, 0.5, SEED);                              \
    }                                                                                              \
    TDLS_TEST_CASE("cooperativelupp/bridge/static-vs-dynamic/" #T "/N=" #N                         \
                   ",rows_per_thread=" #ROWS ",tiny") {                                            \
        bridge_case<T, N, test_config<T, N, ROWS>>(COUNT, 5e-10, SEED + 1);                        \
    }

// Scalar corner, one thread per system, rows_per_thread above N.
TDLS_BRIDGE_CASES(double, 1, 1, 100, 860101)
TDLS_BRIDGE_CASES(double, 12, 12, 100, 861212)
TDLS_BRIDGE_CASES(double, 5, 8, 100, 860508)
// Groups of CPU threads: full mappings, a mixed slot, one row per thread.
TDLS_BRIDGE_CASES(double, 12, 3, 30, 861203)
TDLS_BRIDGE_CASES(double, 12, 6, 30, 861206)
TDLS_BRIDGE_CASES(double, 13, 5, 30, 861305)
TDLS_BRIDGE_CASES(double, 7, 1, 30, 860701)
// Float.
TDLS_BRIDGE_CASES(float, 13, 5, 30, 961305)

TDLS_TEST_CASE("cooperativelupp/bridge/static-vs-dynamic/double/N=13,rows_per_thread=5,colmajor") {
    bridge_case<double, 13,
                config<double, 5, tdls::MatrixLayout::ColMajor, tdls_tests::test_unroll<13, 5>>>(
        30, 0.5, 861315);
}

TDLS_TEST_CASE("cooperativelupp/bridge/static-vs-dynamic/double/N=7,rows_per_thread=3,"
               "dynamic-unroll-ignored") {
    // The static solver unrolls; the dynamic one is configured not to,
    // and has no pragma under either setting.
    bridge_case<double, 7, config<double, 3>,
                config<double, 3, tdls::MatrixLayout::RowMajor, false>>(30, 0.5, 860723);
}

TDLS_TEST_MAIN
