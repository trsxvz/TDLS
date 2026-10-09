/// \file
/// \brief Bridge suite: physical row interchanges reproduce the logical
/// ones.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Both schemes choose the same pivots and run the same operations in the
/// same order: only the place of the rows differs. On identical inputs,
/// the solution of every entry path must be bitwise identical, and so
/// must the factored rows once placed. The logical scheme leaves physical
/// row r at position piv[r]; the physical scheme stores the row at
/// position k in row k and records its original index in piv[k]. The
/// singularity verdicts must agree (one structurally singular system is
/// injected in every batch; its unspecified outputs are not compared).
/// Every other system of a batch is made diagonally dominant, so that the
/// columns whose pivot is in place and the columns that exchange rows are
/// both exercised, and the suite asserts that both occur. The grid crosses
/// the sequential path, full groups, a mixed slot, one row per thread and
/// the unrolled branch on real groups of CPU threads, the float type, the
/// column-major layout and the internal residencies, under the default
/// threshold and under a relative threshold of 0.1.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief The given configuration under physical row interchanges.
/// \tparam T scalar type
/// \param[in] config configuration under logical row interchanges
/// \return the same configuration, with physical row interchanges
template<typename T>
constexpr tdls::CooperativeLUppConfig<T> physical(tdls::CooperativeLUppConfig<T> config) {
    config.row_interchange = tdls::RowInterchange::Physical;
    return config;
}

/// \brief Solves every system of a reproducible batch under both row
/// interchanges through every entry path and checks the bitwise equality
/// of the solutions and of the placed factored rows.
/// \tparam T        scalar type
/// \tparam N        system dimension
/// \tparam Config   configuration under logical row interchanges
/// \tparam internal residency of every operand
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<typename T, int N, tdls::CooperativeLUppConfig<T> Config, bool internal = false>
void interchange_case(const int count, const std::uint64_t seed) {
    using Logical  = tdls_tests::GroupRunner<T, N, Config, internal, internal, internal>;
    using Physical = tdls_tests::GroupRunner<T, N, physical(Config), internal, internal, internal>;
    auto batch     = tdls_tests::make_batch<T>(N, count, seed, 0.5);
    for (int s = 1; s < count; s += 2)
        for (int i = 0; i < N; ++i)
            batch.matrix(s)[i * N + i] += static_cast<T>(N);
    tdls_tests::zero_column(batch, 0, N - 1);

    std::vector<T> A_l(N * N), A_p(N * N), placed(N * N);
    int piv_l[N], piv_p[N], origin[N];
    T x_l[N], x_p[N];
    int solved    = 0;
    int exchanged = 0;
    int in_place  = 0;
    for (int s = 0; s < count; ++s) {
        tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::split,
                              tdls_tests::Path::fused, tdls_tests::Path::split_inplace,
                              tdls_tests::Path::canonical>([&](auto tag) {
            constexpr auto path = decltype(tag)::value;
            const int col       = s % N;
            bool uniform_l      = false;
            bool uniform_p      = false;
            const bool ok_l = Logical::template run<path>(batch.matrix(s), batch.rhs(s), A_l.data(),
                                                          piv_l, x_l, uniform_l, col);
            const bool ok_p = Physical::template run<path>(batch.matrix(s), batch.rhs(s),
                                                           A_p.data(), piv_p, x_p, uniform_p, col);
            TDLS_CHECK(uniform_l && uniform_p);
            TDLS_CHECK(ok_l == ok_p);
            if (!ok_l || !ok_p) return;
            ++solved;
            TDLS_CHECK_BITWISE(x_l, x_p, static_cast<std::size_t>(N));

            // Logical physical row r sits at position piv_l[r].
            bool moved = false;
            for (int r = 0; r < N; ++r) {
                const int position = piv_l[r];
                TDLS_CHECK(position >= 0 && position < N);
                if (position < 0 || position >= N) return;
                for (int c = 0; c < N; ++c)
                    placed[position * N + c] = A_l[r * N + c];
                origin[position] = r;
                moved            = moved || position != r;
            }
            TDLS_CHECK_BITWISE(placed.data(), A_p.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(origin, piv_p, static_cast<std::size_t>(N));
            if constexpr (path == tdls_tests::Path::split) {
                if (moved)
                    ++exchanged;
                else
                    ++in_place;
            }
        });
    }
    // Floor: every system but the singular one solved, on all five paths,
    // and both kinds of columns met (a 1 x 1 system never moves a row).
    TDLS_CHECK(solved == 5 * (count - 1));
    TDLS_CHECK(in_place > 0);
    TDLS_CHECK(N == 1 || exchanged > 0);
}

/// \brief Configuration with the given number of rows per thread and
/// relative threshold, under the unroll policy of the test
/// configurations (see tdls_tests::test_unroll).
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam layout          matrix layout
template<typename T, int N, int rows_per_thread,
         tdls::MatrixLayout layout = tdls::MatrixLayout::RowMajor>
constexpr auto test_config =
    tdls::CooperativeLUppConfig<T>{.rows_per_thread = rows_per_thread,
                                   .unroll_loops    = tdls_tests::test_unroll<N, rows_per_thread>,
                                   .layout          = layout};

/// \brief The same configuration under a relative threshold of 0.1.
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
template<typename T, int N, int rows_per_thread>
constexpr auto threshold_config =
    tdls::CooperativeLUppConfig<T>{.rows_per_thread          = rows_per_thread,
                                   .relative_pivot_threshold = 0.1,
                                   .unroll_loops = tdls_tests::test_unroll<N, rows_per_thread>};

} // namespace

/// Emits the bridge case of one (type, N, rows_per_thread) cell.
#define TDLS_INTERCHANGE_CASE(T, N, ROWS, COUNT, SEED)                                             \
    TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/" #T "/N=" #N                           \
                   ",rows_per_thread=" #ROWS) {                                                    \
        interchange_case<T, N, test_config<T, N, ROWS>>(COUNT, SEED);                              \
    }

// Sequential path: the scalar corner, odd and even dimensions.
TDLS_INTERCHANGE_CASE(double, 1, 1, 50, 870101)
TDLS_INTERCHANGE_CASE(double, 5, 5, 100, 870505)
TDLS_INTERCHANGE_CASE(double, 12, 12, 100, 871212)
// Groups of CPU threads: full mappings, a mixed slot, one row per thread,
// and a small mixed group under the forced unrolling.
TDLS_INTERCHANGE_CASE(double, 12, 3, 30, 871203)
TDLS_INTERCHANGE_CASE(double, 12, 6, 30, 871206)
TDLS_INTERCHANGE_CASE(double, 13, 5, 30, 871305)
TDLS_INTERCHANGE_CASE(double, 7, 1, 30, 870701)
TDLS_INTERCHANGE_CASE(double, 8, 3, 30, 870803)
// Float.
TDLS_INTERCHANGE_CASE(float, 13, 5, 30, 971305)

TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/double/N=13,rows_per_thread=5,colmajor") {
    interchange_case<double, 13, test_config<double, 13, 5, tdls::MatrixLayout::ColMajor>>(30,
                                                                                           871315);
}

// Internal residencies: the slices of the threads hold the moved rows.
TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/double/N=8,rows_per_thread=3,internal") {
    interchange_case<double, 8, test_config<double, 8, 3>, true>(30, 870813);
}
TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/double/N=12,rows_per_thread=12,internal") {
    interchange_case<double, 12, test_config<double, 12, 12>, true>(100, 871222);
}

// Relative threshold 0.1: both schemes keep the same pivots in place.
TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/double/N=7,rows_per_thread=7,threshold") {
    interchange_case<double, 7, threshold_config<double, 7, 7>>(100, 870777);
}
TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/double/N=12,rows_per_thread=3,threshold") {
    interchange_case<double, 12, threshold_config<double, 12, 3>>(30, 871233);
}
TDLS_TEST_CASE("cooperativelupp/bridge/row-interchange/double/N=13,rows_per_thread=5,threshold") {
    interchange_case<double, 13, threshold_config<double, 13, 5>>(30, 871355);
}

TDLS_TEST_MAIN
