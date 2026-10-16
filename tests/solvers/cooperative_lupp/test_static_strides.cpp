/// \file
/// \brief Bridge suite: the compile-time strides of
/// CooperativeLUppSolverStatic against the runtime ones.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every entry point takes the stride of each operand as a template
/// argument too (tdls::dynamic_stride by default). Each one runs twice on
/// a batch of 3 systems interleaved as in the AoSoA layout, each operand
/// with its own stride, so that a stride given to the wrong operand shows:
/// the matrices by 4, the pivots by 5, the right-hand sides and solutions
/// by 6, the free lanes left as NaN (-1 for the pivots). Once with the
/// runtime strides, once with the strides as template arguments and
/// function arguments of 1, which the solver must ignore. The two give the
/// same buffers, bit for bit, the free lanes included, and the same
/// verdicts: one thread per system and groups of several threads, a single
/// stride given at compile time, the barrier deduced by default (no
/// barrier argument), the physical row interchanges and the column-major
/// layout.

#include <cstdint>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// Stride of the matrices.
constexpr int WA = 4;
/// Stride of the pivots.
constexpr int WP = 5;
/// Stride of the right-hand sides and solutions.
constexpr int WY = 6;
/// Systems of the batch: the other lanes stay free.
constexpr int count = 3;
/// The default of the stride template parameters.
constexpr int dyn = tdls::dynamic_stride;

/// \brief The buffers of one run, interleaved by WA, WP and WY.
struct Buffers {
    std::vector<double> A; ///< matrices
    std::vector<int> piv;  ///< pivots
    std::vector<double> b; ///< right-hand sides
    std::vector<double> x; ///< solutions
    std::vector<int> ok;   ///< verdicts
};

/// \brief The buffers of a batch, the free lanes as NaN (-1 for the
/// pivots).
/// \param[in] batch the systems
/// \param[in] N     dimension
/// \return the buffers, x a copy of b
Buffers fill(const tdls_tests::SystemBatch<double>& batch, const int N) {
    const double nan    = std::numeric_limits<double>::quiet_NaN();
    const std::size_t n = static_cast<std::size_t>(N);
    Buffers u{std::vector<double>(n * n * WA, nan), std::vector<int>(n * WP, -1),
              std::vector<double>(n * WY, nan), std::vector<double>(n * WY, nan),
              std::vector<int>(count, -1)};
    for (int s = 0; s < count; ++s) {
        for (int e = 0; e < N * N; ++e)
            u.A[static_cast<std::size_t>(e) * WA + s] = batch.matrix(s)[e];
        for (int i = 0; i < N; ++i) {
            u.b[static_cast<std::size_t>(i) * WY + s] = batch.rhs(s)[i];
            u.x[static_cast<std::size_t>(i) * WY + s] = batch.rhs(s)[i];
        }
    }
    return u;
}

/// \brief Bitwise equality of two runs.
/// \param[in] a first run
/// \param[in] b second run
void same(const Buffers& a, const Buffers& b) {
    TDLS_CHECK_BITWISE(a.A.data(), b.A.data(), a.A.size());
    TDLS_CHECK_BITWISE(a.piv.data(), b.piv.data(), a.piv.size());
    TDLS_CHECK_BITWISE(a.b.data(), b.b.data(), a.b.size());
    TDLS_CHECK_BITWISE(a.x.data(), b.x.data(), a.x.size());
    TDLS_CHECK_BITWISE(a.ok.data(), b.ok.data(), a.ok.size());
}

/// \brief Runs a call on every system of a copy of the buffers, each
/// system by its group of threads.
/// \tparam threads   threads per system
/// \tparam work_size entries of the workspace of a group
/// \tparam Fn        callable (tx, sync, A, piv, b, x, work) returning the
///                   verdict
/// \param[in] u  the buffers
/// \param[in] fn the call
/// \return the buffers after the calls
template<int threads, int work_size, typename Fn>
Buffers run(const Buffers& u, Fn&& fn) {
    Buffers v = u;
    for (int s = 0; s < count; ++s) {
        std::vector<double> work(work_size, 0.0);
        tdls_tests::run_group<threads>([&](const int tx, auto&& sync) {
            const bool ok = fn(tx, sync, v.A.data() + s, v.piv.data() + s, v.b.data() + s,
                               v.x.data() + s, work.data());
            if (tx == 0) v.ok[static_cast<std::size_t>(s)] = ok ? 1 : 0;
        });
    }
    return v;
}

/// \brief Compares every entry point with its compile-time strides against
/// its runtime strides.
/// \tparam N               dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam interchange     row interchanges
/// \tparam layout          matrix layout
/// \param[in] seed generator seed
template<int N, int rows_per_thread, tdls::RowInterchange interchange, tdls::MatrixLayout layout>
void strides_case(const std::uint64_t seed) {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = rows_per_thread,
        .row_interchange = interchange,
        .unroll_loops    = tdls_tests::test_unroll<N, rows_per_thread>,
        .layout          = layout};
    using S          = tdls::CooperativeLUppSolverStatic<double, N, config>;
    constexpr int th = S::threads_per_system;
    constexpr int ws = S::workspace_size;
    const auto batch = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    const Buffers u  = fill(batch, N);

    const auto solved = run<th, ws>(
        u, [](int tx, auto&& sync, double* A, int* p, double* b, double* x, double* work) {
            return S::template solve<false, false, false>(tx, A, WA, p, WP, b, x, WY, work, sync);
        });
    same(solved, run<th, ws>(u, [](int tx, auto&& sync, double* A, int* p, double* b, double* x,
                                   double* work) {
             return S::template solve<false, false, false, WA, WP, WY>(tx, A, 1, p, 1, b, x, 1,
                                                                       work, sync);
         }));
    // a single stride at compile time, the others at run time
    same(solved, run<th, ws>(u, [](int tx, auto&& sync, double* A, int* p, double* b, double* x,
                                   double* work) {
             return S::template solve<false, false, false, WA>(tx, A, 1, p, WP, b, x, WY, work,
                                                               sync);
         }));
    same(solved, run<th, ws>(u, [](int tx, auto&& sync, double* A, int* p, double* b, double* x,
                                   double* work) {
             return S::template solve<false, false, false, dyn, WP>(tx, A, WA, p, 1, b, x, WY, work,
                                                                    sync);
         }));
    same(solved, run<th, ws>(u, [](int tx, auto&& sync, double* A, int* p, double* b, double* x,
                                   double* work) {
             return S::template solve<false, false, false, dyn, dyn, WY>(tx, A, WA, p, WP, b, x, 1,
                                                                         work, sync);
         }));
    if constexpr (th == 1) {
        // the barrier deduced by default, no barrier argument
        same(solved, run<th, ws>(u, [](int tx, auto&&, double* A, int* p, double* b, double* x,
                                       double* work) {
                 return S::template solve<false, false, false, WA, WP, WY>(tx, A, 1, p, 1, b, x, 1,
                                                                           work);
             }));
    }
    same(run<th, ws>(u,
                     [](int tx, auto&& sync, double* A, int* p, double*, double* x, double* work) {
                         return S::template solve_inplace<false, false, false>(tx, A, WA, p, WP, x,
                                                                               WY, work, sync);
                     }),
         run<th, ws>(u,
                     [](int tx, auto&& sync, double* A, int* p, double*, double* x, double* work) {
                         return S::template solve_inplace<false, false, false, WA, WP, WY>(
                             tx, A, 1, p, 1, x, 1, work, sync);
                     }));
    const auto factored =
        run<th, ws>(u, [](int tx, auto&& sync, double* A, int* p, double*, double*, double* work) {
            return S::template factorize<false, false>(tx, A, WA, p, WP, work, sync);
        });
    same(factored,
         run<th, ws>(u, [](int tx, auto&& sync, double* A, int* p, double*, double*, double* work) {
             return S::template factorize<false, false, WA, WP>(tx, A, 1, p, 1, work, sync);
         }));
    same(
        run<th, ws>(factored,
                    [](int tx, auto&& sync, double* A, int* p, double* b, double* x, double* work) {
                        S::template substitute<false, false, false>(tx, A, WA, p, WP, b, x, WY,
                                                                    work, sync);
                        return true;
                    }),
        run<th, ws>(factored,
                    [](int tx, auto&& sync, double* A, int* p, double* b, double* x, double* work) {
                        S::template substitute<false, false, false, WA, WP, WY>(tx, A, 1, p, 1, b,
                                                                                x, 1, work, sync);
                        return true;
                    }));
    same(run<th, ws>(factored,
                     [](int tx, auto&& sync, double* A, int* p, double*, double* x, double* work) {
                         S::template substitute_inplace<false, false, false>(tx, A, WA, p, WP, x,
                                                                             WY, work, sync);
                         return true;
                     }),
         run<th, ws>(factored,
                     [](int tx, auto&& sync, double* A, int* p, double*, double* x, double* work) {
                         S::template substitute_inplace<false, false, false, WA, WP, WY>(
                             tx, A, 1, p, 1, x, 1, work, sync);
                         return true;
                     }));
    same(run<th, ws>(factored,
                     [](int tx, auto&& sync, double* A, int* p, double*, double* x, double* work) {
                         S::template substitute_canonical<false, false, false>(tx, A, WA, p, WP, 1,
                                                                               x, WY, work, sync);
                         return true;
                     }),
         run<th, ws>(factored,
                     [](int tx, auto&& sync, double* A, int* p, double*, double* x, double* work) {
                         S::template substitute_canonical<false, false, false, WA, WP, WY>(
                             tx, A, 1, p, 1, 1, x, 1, work, sync);
                         return true;
                     }));
    // Floor: the batch is invertible.
    int ok = 0;
    for (const int v : solved.ok)
        ok += v;
    TDLS_CHECK(ok == count);
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/bridge/static_strides/N=6,one thread") {
    strides_case<6, 6, tdls::RowInterchange::Logical, tdls::MatrixLayout::RowMajor>(160100);
}
TDLS_TEST_CASE("cooperativelupp/bridge/static_strides/N=5,rows_per_thread=2") {
    strides_case<5, 2, tdls::RowInterchange::Logical, tdls::MatrixLayout::RowMajor>(160200);
}
TDLS_TEST_CASE("cooperativelupp/bridge/static_strides/N=5,rows_per_thread=2,physical") {
    strides_case<5, 2, tdls::RowInterchange::Physical, tdls::MatrixLayout::RowMajor>(160300);
}
TDLS_TEST_CASE("cooperativelupp/bridge/static_strides/N=5,rows_per_thread=2,col") {
    strides_case<5, 2, tdls::RowInterchange::Logical, tdls::MatrixLayout::ColMajor>(160400);
}

TDLS_TEST_MAIN
