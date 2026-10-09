/// \file
/// \brief Bridge suite: the deduced barrier reproduces an explicit one.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every case solves a reproducible batch through every entry path twice:
/// once with the GroupBarrier of the runner, once with tdls::AutoSync, the
/// solver then deducing its barrier. On the host, that is the CPU barrier
/// held in the first two elements of the workspace for a group of several
/// threads, and nothing for a group of one. The barrier changes no
/// arithmetic: the verdicts, the factored rows, the pivot entries and the
/// solutions must be bitwise identical. One structurally singular system
/// is injected in every batch; its unspecified outputs are not compared.
/// The grid crosses one thread per system, full groups, a mixed slot,
/// one row per thread, the unrolled branch, both row interchanges, the
/// float type, the column-major layout, the internal residencies, and
/// the runtime solver over a range of dimensions, hence of group sizes.
///
/// A reuse case solves systems of several dimensions, hence several
/// group sizes, in one workspace zeroed once: the barrier keeps its two
/// elements in place and in their state between calls.
///
/// A last case builds the barrier with make_sync, as a caller exchanging
/// data between the threads of its group: each thread writes a
/// right-hand side read by another thread, between two solves that use
/// the same barrier. The outputs must match those of the explicit
/// barrier, and the barrier elements of the workspace must be back to
/// their state between calls.

#include <cstdint>
#include <thread>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief Solves every system of a reproducible batch through every entry
/// path with an explicit and with the deduced barrier, and checks the
/// bitwise equality of every output.
/// \tparam T        scalar type
/// \tparam N        system dimension
/// \tparam Config   solver configuration
/// \tparam internal residency of every operand
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<typename T, int N, tdls::CooperativeLUppConfig<T> Config, bool internal = false>
void static_case(const int count, const std::uint64_t seed) {
    using Explicit = tdls_tests::GroupRunner<T, N, Config, internal, internal, internal, false>;
    using Deduced  = tdls_tests::GroupRunner<T, N, Config, internal, internal, internal, true>;
    auto batch     = tdls_tests::make_batch<T>(N, count, seed, 0.5);
    tdls_tests::zero_column(batch, 0, N - 1);

    std::vector<T> A_e(N * N), A_d(N * N);
    int piv_e[N], piv_d[N];
    T x_e[N], x_d[N];
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::split,
                              tdls_tests::Path::fused, tdls_tests::Path::split_inplace,
                              tdls_tests::Path::canonical>([&](auto tag) {
            constexpr auto path = decltype(tag)::value;
            const int col       = s % N;
            bool uniform_e      = false;
            bool uniform_d      = false;
            const bool ok_e = Explicit::template run<path>(batch.matrix(s), batch.rhs(s),
                                                           A_e.data(), piv_e, x_e, uniform_e, col);
            const bool ok_d = Deduced::template run<path>(batch.matrix(s), batch.rhs(s), A_d.data(),
                                                          piv_d, x_d, uniform_d, col);
            TDLS_CHECK(uniform_e && uniform_d);
            TDLS_CHECK(ok_e == ok_d);
            if (!ok_e || !ok_d) return;
            ++solved;
            TDLS_CHECK_BITWISE(A_e.data(), A_d.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_e, piv_d, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(x_e, x_d, static_cast<std::size_t>(N));
        });
    }
    // Floor: every system but the singular one solved, on all five paths.
    TDLS_CHECK(solved == 5 * (count - 1));
}

/// \brief The same bridge for the runtime solver, over the dimensions
/// 1 to n_max, so that the size of the group varies at run time.
/// \tparam T      scalar type
/// \tparam Config solver configuration
/// \param[in] n_max largest dimension
/// \param[in] count number of systems per dimension
/// \param[in] seed  generator seed
template<typename T, tdls::CooperativeLUppConfig<T> Config>
void dynamic_case(const int n_max, const int count, const std::uint64_t seed) {
    using Explicit = tdls_tests::DynamicGroupRunner<T, Config, false>;
    using Deduced  = tdls_tests::DynamicGroupRunner<T, Config, true>;
    int solved     = 0;
    for (int n = 1; n <= n_max; ++n) {
        auto batch = tdls_tests::make_batch<T>(n, count, seed + static_cast<std::uint64_t>(n), 0.5);
        tdls_tests::zero_column(batch, 0, n - 1);
        std::vector<T> A_e(n * n), A_d(n * n), x_e(n), x_d(n);
        std::vector<int> piv_e(n), piv_d(n);
        for (int s = 0; s < count; ++s) {
            tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::split,
                                  tdls_tests::Path::fused, tdls_tests::Path::split_inplace,
                                  tdls_tests::Path::canonical>([&](auto tag) {
                constexpr auto path = decltype(tag)::value;
                const int col       = s % n;
                bool uniform_e      = false;
                bool uniform_d      = false;
                const bool ok_e =
                    Explicit::template run<path>(n, batch.matrix(s), batch.rhs(s), A_e.data(),
                                                 piv_e.data(), x_e.data(), uniform_e, col);
                const bool ok_d =
                    Deduced::template run<path>(n, batch.matrix(s), batch.rhs(s), A_d.data(),
                                                piv_d.data(), x_d.data(), uniform_d, col);
                TDLS_CHECK(uniform_e && uniform_d);
                TDLS_CHECK(ok_e == ok_d);
                if (!ok_e || !ok_d) return;
                ++solved;
                TDLS_CHECK_BITWISE(A_e.data(), A_d.data(), static_cast<std::size_t>(n) * n);
                TDLS_CHECK_BITWISE(piv_e.data(), piv_d.data(), static_cast<std::size_t>(n));
                TDLS_CHECK_BITWISE(x_e.data(), x_d.data(), static_cast<std::size_t>(n));
            });
        }
    }
    // Floor: every system but the singular one of each dimension solved.
    TDLS_CHECK(solved == 5 * (count - 1) * n_max);
}

/// \brief Two solves of one system by a group of CPU threads, with an
/// exchange between them: each thread writes the second right-hand side
/// of the rows of the next thread. The barrier is the one of make_sync,
/// called by the threads themselves and passed to the entry points, or
/// a GroupBarrier.
/// \tparam N       system dimension
/// \tparam Config  solver configuration
/// \tparam deduced whether the barrier comes from make_sync
/// \param[in]  A0   matrix, contiguous row-major
/// \param[in]  b0   first right-hand side
/// \param[out] x1   first solution
/// \param[out] x2   second solution
/// \param[out] ok   whether every thread solved the system
/// \param[out] rest whether the barrier elements of the workspace end in
///             their state between calls
template<int N, tdls::CooperativeLUppConfig<double> Config, bool deduced>
void exchange(const double* A0, const double* b0, double* x1, double* x2, bool& ok, bool& rest) {
    using Solver    = tdls::CooperativeLUppSolverStatic<double, N, Config>;
    constexpr int T = Solver::threads_per_system;
    constexpr int R = Solver::rows_per_thread;
    std::vector<double> A(A0, A0 + N * N), b2(N), work(Solver::workspace_size, 0.0);
    std::vector<int> piv(N);
    std::vector<char> verdicts(T, 0);
    tdls_tests::GroupBarrier group_barrier(T);
    std::vector<std::thread> group;
    for (int tx = 0; tx < T; ++tx)
        group.emplace_back([&, tx] {
            auto run = [&](auto&& sync) {
                verdicts[tx] = Solver::template solve<false, false, false>(
                    tx, A.data(), 1, piv.data(), 1, b0, x1, 1, work.data(), sync);
                // Rows of the next thread: read by it after the barrier.
                const int next = (tx + 1) % T;
                for (int K = 0; K < R; ++K) {
                    const int r = next + K * T;
                    if (r < N) b2[r] = 2.0 * x1[r] + 1.0;
                }
                sync();
                Solver::template substitute<false, false, false>(
                    tx, A.data(), 1, piv.data(), 1, b2.data(), x2, 1, work.data(), sync);
            };
            if constexpr (deduced)
                run(Solver::make_sync(work.data()));
            else
                run([&] { group_barrier.arrive_and_wait(); });
        });
    for (auto& thread : group)
        thread.join();
    ok = true;
    for (const char verdict : verdicts)
        ok = ok && verdict != 0;
    const double count = work[0];
    const double flag  = work[1];
    rest               = !deduced || (count == 0.0 && (flag == 0.0 || flag == 1.0));
}

/// \brief Configuration with the given number of rows per thread, row
/// interchanges and layout, under the unroll policy of the test
/// configurations (see tdls_tests::test_unroll).
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam interchange     row interchanges
/// \tparam layout          matrix layout
template<typename T, int N, int rows_per_thread,
         tdls::RowInterchange interchange = tdls::RowInterchange::Logical,
         tdls::MatrixLayout layout        = tdls::MatrixLayout::RowMajor>
constexpr auto test_config =
    tdls::CooperativeLUppConfig<T>{.rows_per_thread = rows_per_thread,
                                   .row_interchange = interchange,
                                   .unroll_loops    = tdls_tests::test_unroll<N, rows_per_thread>,
                                   .layout          = layout};

/// \brief Physical row interchanges.
constexpr auto physical = tdls::RowInterchange::Physical;

} // namespace

/// Emits the bridge case of one (type, N, rows_per_thread) cell.
#define TDLS_AUTO_SYNC_CASE(T, N, ROWS, COUNT, SEED)                                               \
    TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/" #T "/N=" #N ",rows_per_thread=" #ROWS) {    \
        static_case<T, N, test_config<T, N, ROWS>>(COUNT, SEED);                                   \
    }

// One thread per system: the deduced barrier resolves to nothing.
TDLS_AUTO_SYNC_CASE(double, 1, 1, 30, 910101)
TDLS_AUTO_SYNC_CASE(double, 9, 9, 50, 910909)
// Groups of CPU threads: full mappings, a mixed slot, one row per thread,
// and a small mixed group under the forced unrolling.
TDLS_AUTO_SYNC_CASE(double, 12, 3, 20, 911203)
TDLS_AUTO_SYNC_CASE(double, 13, 5, 20, 911305)
TDLS_AUTO_SYNC_CASE(double, 7, 1, 20, 910701)
TDLS_AUTO_SYNC_CASE(double, 8, 3, 20, 910803)
// Float.
TDLS_AUTO_SYNC_CASE(float, 13, 5, 20, 921305)

TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/double/N=13,rows_per_thread=5,physical") {
    static_case<double, 13, test_config<double, 13, 5, physical>>(20, 911315);
}
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/double/N=8,rows_per_thread=3,physical") {
    static_case<double, 8, test_config<double, 8, 3, physical>>(20, 910813);
}
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/double/N=12,rows_per_thread=5,colmajor") {
    static_case<
        double, 12,
        test_config<double, 12, 5, tdls::RowInterchange::Logical, tdls::MatrixLayout::ColMajor>>(
        20, 911225);
}

// Internal residencies: the slices of the threads, the barrier in the
// shared workspace.
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/double/N=8,rows_per_thread=3,internal") {
    static_case<double, 8, test_config<double, 8, 3>, true>(20, 910833);
}

// Runtime solver: groups of 1 to 5 threads.
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/dynamic/double,rows_per_thread=3") {
    dynamic_case<double, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3}>(13, 6, 912000);
}
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/dynamic/double,rows_per_thread=2,physical") {
    dynamic_case<double, tdls::CooperativeLUppConfig<double>{
                             .rows_per_thread = 2, .row_interchange = physical}>(9, 6, 913000);
}

// One workspace, zeroed once, for dimensions 12, 6, 9, 1 and 13: groups
// of 4, 2, 3, 1 and 5 threads in turn.
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/dynamic/workspace-reuse") {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;
    using Explicit        = tdls_tests::DynamicGroupRunner<double, config, false>;
    std::vector<double> work(Solver::workspace_size(13), 0.0);
    int solved = 0;
    for (const int n : {12, 6, 9, 1, 13}) {
        const auto batch = tdls_tests::make_batch<double>(n, 1, 915000 + n, 0.5);
        std::vector<double> A(batch.matrix(0), batch.matrix(0) + n * n), x(n), A_e(n * n), x_e(n);
        std::vector<int> piv(n), piv_e(n);
        bool uniform = false;
        TDLS_CHECK(Explicit::template run<tdls_tests::Path::combined>(
            n, batch.matrix(0), batch.rhs(0), A_e.data(), piv_e.data(), x_e.data(), uniform));
        const int threads = Solver::threads_per_system(n);
        std::vector<char> verdicts(threads, 0);
        tdls_tests::run_group<true>(threads, [&](const int tx, auto&& sync) {
            verdicts[tx] = Solver::solve(n, tx, A.data(), 1, piv.data(), 1, batch.rhs(0), x.data(),
                                         1, work.data(), sync);
        });
        bool ok = true;
        for (const char verdict : verdicts)
            ok = ok && verdict != 0;
        TDLS_CHECK(ok);
        if (!ok) continue;
        ++solved;
        TDLS_CHECK_BITWISE(x.data(), x_e.data(), static_cast<std::size_t>(n));
        TDLS_CHECK(work[0] == 0.0 && (work[1] == 0.0 || work[1] == 1.0));
    }
    TDLS_CHECK(solved == 5);
}

// make_sync: the caller's exchange between two solves.
TDLS_TEST_CASE("cooperativelupp/bridge/auto-sync/make-sync/double/N=11,rows_per_thread=3") {
    constexpr int N    = 11;
    constexpr auto cfg = test_config<double, N, 3>;
    auto batch         = tdls_tests::make_batch<double>(N, 10, 914000, 0.5);
    for (int s = 0; s < 10; ++s) {
        double x1_e[N], x2_e[N], x1_d[N], x2_d[N];
        bool ok_e   = false;
        bool ok_d   = false;
        bool rest_e = false;
        bool rest_d = false;
        exchange<N, cfg, false>(batch.matrix(s), batch.rhs(s), x1_e, x2_e, ok_e, rest_e);
        exchange<N, cfg, true>(batch.matrix(s), batch.rhs(s), x1_d, x2_d, ok_d, rest_d);
        TDLS_CHECK(ok_e && ok_d);
        TDLS_CHECK(rest_e && rest_d);
        TDLS_CHECK_BITWISE(x1_e, x1_d, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x2_e, x2_d, static_cast<std::size_t>(N));
    }
}

TDLS_TEST_MAIN
