/// \file
/// \brief Bridge suite: the entry points are mutually consistent.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The equivalences documented by the CooperativeLUpp solver are checked
/// bitwise on identical inputs: solve() and solve_inplace(), the original
/// MAGMA kernel, against factorize() + substitute();
/// factorize() + substitute_inplace() against the same; substitute_canonical()
/// against substitute() on the canonical vector; and the reuse of one
/// factorization across several right-hand sides against fresh solves.
/// Every case runs on the sequential path and on groups of CPU threads.
/// The guarantees on return close the suite, on both solvers: every
/// thread reads the whole results right after the call, the entries of
/// the other threads included, then overwrites its share of the
/// workspace, and nothing may differ from the sequential solve. A last
/// case guards the workspace contract: no solver may declare the
/// workspace restrict, since the other threads write it between two
/// barriers.

#include <cstdint>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief Runs every entry-point equivalence on one reproducible batch.
/// The matrix is external, pivot and right-hand sides are slices of the
/// threads; the residency bridge covers the other combinations.
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<typename T, int N, int rows_per_thread>
void entry_points_case(const int count, const std::uint64_t seed) {
    constexpr auto config =
        tdls::CooperativeLUppConfig<T>{.rows_per_thread = rows_per_thread,
                                       .unroll_loops = tdls_tests::test_unroll<N, rows_per_thread>};
    using Runner     = tdls_tests::GroupRunner<T, N, config, true, true, false>;
    using Solver     = typename Runner::Solver;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, 0.5);

    std::vector<T> A_split(N * N), A_other(N * N);
    int piv_split[N], piv_other[N];
    T x_split[N], x_other[N];
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        const T* A0  = batch.matrix(s);
        const T* b0  = batch.rhs(s);
        bool uniform = false;

        // Baseline: factorize then substitute.
        if (!Runner::template run<tdls_tests::Path::split>(A0, b0, A_split.data(), piv_split,
                                                           x_split, uniform))
            continue;
        TDLS_CHECK(uniform);
        ++solved;

        // solve(), solve_inplace() and factorize() + substitute_inplace()
        // must reproduce the split path.
        tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::fused,
                              tdls_tests::Path::split_inplace>([&](auto path_tag) {
            constexpr auto path = decltype(path_tag)::value;
            const bool ok =
                Runner::template run<path>(A0, b0, A_other.data(), piv_other, x_other, uniform);
            TDLS_CHECK(ok && uniform);
            TDLS_CHECK_BITWISE(A_split.data(), A_other.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_split, piv_other, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(x_split, x_other, static_cast<std::size_t>(N));
        });

        // substitute_canonical(col) must reproduce substitute() on e_col.
        for (int col = 0; col < N; ++col) {
            T e[N], x_canonical[N];
            for (int i = 0; i < N; ++i)
                e[i] = i == col ? T(1) : T(0);
            (void)Runner::template run<tdls_tests::Path::split>(A0, e, A_other.data(), piv_other,
                                                                x_other, uniform);
            (void)Runner::template run<tdls_tests::Path::canonical>(
                A0, e, A_other.data(), piv_other, x_canonical, uniform, col);
            TDLS_CHECK_BITWISE(x_other, x_canonical, static_cast<std::size_t>(N));
        }

        // One factorization reused for a second right-hand side must match
        // a fresh solve of the same system.
        {
            const T* b2 = batch.rhs((s + 1) % count);
            T x_reuse[N];
            std::vector<T> A(A0, A0 + N * N);
            std::vector<int> piv(N);
            std::vector<T> work(Solver::workspace_size);
            tdls_tests::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
                T x_first[N];
                if (!Solver::template factorize<false, false>(tx, A.data(), 1, piv.data(), 1,
                                                              work.data(), sync))
                    return;
                Solver::template substitute<false, false, false>(tx, A.data(), 1, piv.data(), 1, b0,
                                                                 x_first, 1, work.data(), sync);
                Solver::template substitute<false, false, false>(tx, A.data(), 1, piv.data(), 1, b2,
                                                                 x_reuse, 1, work.data(), sync);
            });
            const bool ok = Runner::template run<tdls_tests::Path::split>(
                A0, b2, A_other.data(), piv_other, x_other, uniform);
            TDLS_CHECK(ok);
            TDLS_CHECK_BITWISE(x_reuse, x_other, static_cast<std::size_t>(N));
        }
    }
    // Floor: the generator produces well-conditioned systems, so every
    // one must have been solved (a suite green on zero solved systems
    // would assert nothing).
    TDLS_CHECK(solved == count);
}

/// \brief Checks the guarantees on return on one reproducible batch: N =
/// 13 on 3 threads of 5 rows, every operand external. Right after
/// solve_inplace, each thread copies the whole factored matrix, the row
/// positions and the solution, then overwrites its share of the
/// workspace with NaNs. Every copy must match the sequential solve
/// bitwise, which holds only if every result is visible to the whole
/// group on return and the workspace is no longer in use.
/// \tparam dynamic exercise the runtime solver instead of the
///         compile-time one
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<bool dynamic>
void return_guarantees_case(const int count, const std::uint64_t seed) {
    constexpr int N = 13;
    static constexpr auto config =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 5, .unroll_loops = false};
    static constexpr auto single =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = N, .unroll_loops = false};
    using Static          = tdls::CooperativeLUppSolverStatic<double, N, config>;
    using Dynamic         = tdls::CooperativeLUppSolverDynamic<double, config>;
    using Sequential      = tdls::CooperativeLUppSolverStatic<double, N, single>;
    constexpr int threads = Static::threads_per_system;
    const auto batch      = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    int checked           = 0;
    for (int s = 0; s < count; ++s) {
        std::vector<double> A_ref(batch.matrix(s), batch.matrix(s) + N * N);
        std::vector<double> y_ref(batch.rhs(s), batch.rhs(s) + N), w_ref(3 * N);
        std::vector<int> piv_ref(N);
        if (!Sequential::solve_inplace<false, false, false>(0, A_ref.data(), 1, piv_ref.data(), 1,
                                                            y_ref.data(), 1, w_ref.data()))
            continue;

        std::vector<double> A(batch.matrix(s), batch.matrix(s) + N * N);
        std::vector<double> y(batch.rhs(s), batch.rhs(s) + N), work(3 * N);
        std::vector<int> piv(N), verdicts(threads);
        std::vector<std::vector<double>> seen_A(threads), seen_y(threads);
        std::vector<std::vector<int>> seen_piv(threads);
        tdls_tests::run_group<threads>([&](const int tx, auto&& sync) {
            bool ok = false;
            if constexpr (dynamic)
                ok = Dynamic::solve_inplace(N, tx, A.data(), 1, piv.data(), 1, y.data(), 1,
                                            work.data(), sync);
            else
                ok = Static::solve_inplace<false, false, false>(tx, A.data(), 1, piv.data(), 1,
                                                                y.data(), 1, work.data(), sync);
            verdicts[tx] = ok ? 1 : 0;
            // Right after the call: every result, the entries written by
            // the other threads included...
            seen_A[tx].assign(A.begin(), A.end());
            seen_y[tx].assign(y.begin(), y.end());
            seen_piv[tx].assign(piv.begin(), piv.end());
            // ... and the workspace reused at once, each thread its share.
            for (int i = tx; i < 3 * N; i += threads)
                work[i] = std::numeric_limits<double>::quiet_NaN();
        });
        for (int tx = 0; tx < threads; ++tx) {
            TDLS_CHECK(verdicts[tx] == 1);
            TDLS_CHECK_BITWISE(A_ref.data(), seen_A[tx].data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(y_ref.data(), seen_y[tx].data(), static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(piv_ref.data(), seen_piv[tx].data(), static_cast<std::size_t>(N));
        }
        ++checked;
    }
    // Floor: every system solved and checked.
    TDLS_CHECK(checked == count);
}

/// \brief Regression of the workspace contract: the other threads of the
/// group write the workspace between two barriers, so no solver may
/// declare it restrict. With the qualifier, clang 20 at -O3 reused a
/// workspace value read before a barrier in this very pattern, the
/// runtime dimension snippet of substitute_inplace: factorize, then
/// substitute_inplace, n = 5 on 3 threads of 2 rows, operands in
/// vectors. The solution must match the sequential solve bitwise.
/// \tparam dynamic exercise the runtime solver instead of the
///         compile-time one
template<bool dynamic>
void workspace_contract_case() {
    constexpr int N              = 5;
    static constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    static constexpr auto single = tdls::CooperativeLUppConfig<double>{.rows_per_thread = N};
    using Static                 = tdls::CooperativeLUppSolverStatic<double, N, config>;
    using Dynamic                = tdls::CooperativeLUppSolverDynamic<double, config>;
    using Sequential             = tdls::CooperativeLUppSolverStatic<double, N, single>;
    const std::vector<double> A0 = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                    1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    const std::vector<double> b0 = {12, 16, 27, 40, 50};

    std::vector<double> A_ref(A0), x_ref(b0), w_ref(3 * N);
    std::vector<int> piv_ref(N);
    const bool factored =
        Sequential::factorize<false, false>(0, A_ref.data(), 1, piv_ref.data(), 1, w_ref.data());
    TDLS_CHECK(factored);
    Sequential::substitute_inplace<false, false, false>(0, A_ref.data(), 1, piv_ref.data(), 1,
                                                        x_ref.data(), 1, w_ref.data());

    std::vector<double> A(A0), x(b0), work(3 * N);
    std::vector<int> piv(N), verdicts(Static::threads_per_system, 0);
    // The runtime solver runs on the runtime form of run_group, as in the
    // snippet; the compile-time one needs the compile-time form, whose
    // group of one alone gets no barrier.
    if constexpr (dynamic) {
        tdls_tests::run_group(Dynamic::threads_per_system(N), [&](const int tx, auto&& sync) {
            verdicts[tx] =
                Dynamic::factorize(N, tx, A.data(), 1, piv.data(), 1, work.data(), sync) ? 1 : 0;
            Dynamic::substitute_inplace(N, tx, A.data(), 1, piv.data(), 1, x.data(), 1, work.data(),
                                        sync);
        });
    } else {
        tdls_tests::run_group<Static::threads_per_system>([&](const int tx, auto&& sync) {
            verdicts[tx] =
                Static::factorize<false, false>(tx, A.data(), 1, piv.data(), 1, work.data(), sync)
                    ? 1
                    : 0;
            Static::substitute_inplace<false, false, false>(tx, A.data(), 1, piv.data(), 1,
                                                            x.data(), 1, work.data(), sync);
        });
    }
    for (const int verdict : verdicts)
        TDLS_CHECK(verdict == 1);
    TDLS_CHECK_BITWISE(x_ref.data(), x.data(), static_cast<std::size_t>(N));
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/group/workspace-contract/static/double/N=5,rows_per_thread=2") {
    workspace_contract_case<false>();
}
TDLS_TEST_CASE("cooperativelupp/group/workspace-contract/dynamic/double/n=5,rows_per_thread=2") {
    workspace_contract_case<true>();
}
TDLS_TEST_CASE("cooperativelupp/group/return-guarantees/static/double/N=13,rows_per_thread=5") {
    return_guarantees_case<false>(300, 630513);
}
TDLS_TEST_CASE("cooperativelupp/group/return-guarantees/dynamic/double/n=13,rows_per_thread=5") {
    return_guarantees_case<true>(300, 630523);
}

TDLS_TEST_CASE("cooperativelupp/bridge/entry-points/double/N=12,rows_per_thread=12") {
    entry_points_case<double, 12, 12>(100, 630112);
}
TDLS_TEST_CASE("cooperativelupp/bridge/entry-points/double/N=7,rows_per_thread=3") {
    entry_points_case<double, 7, 3>(30, 630703);
}
TDLS_TEST_CASE("cooperativelupp/bridge/entry-points/double/N=12,rows_per_thread=3") {
    entry_points_case<double, 12, 3>(30, 630203);
}
TDLS_TEST_CASE("cooperativelupp/bridge/entry-points/double/N=13,rows_per_thread=5") {
    entry_points_case<double, 13, 5>(30, 630305);
}
TDLS_TEST_CASE("cooperativelupp/bridge/entry-points/float/N=12,rows_per_thread=4") {
    entry_points_case<float, 12, 4>(30, 630404);
}

TDLS_TEST_MAIN
