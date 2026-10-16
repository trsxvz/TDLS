/// \file
/// \brief Bridge suite: the compile-time strides of TiledLUppSolverStatic
/// against the runtime ones.
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
/// verdicts and out-of-tile counts. Every overload is covered, the
/// counting and diagnostics-free ones, the forward substitution folded in
/// factorize, a single stride given at compile time (the others at run
/// time), both schedules, a tile tail, the physical row interchanges and
/// the column-major layout.

#include <cstdint>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
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
/// Right-hand sides of the multi right-hand-side entry points.
constexpr int nrhs = 2;
/// The default of the stride template parameters.
constexpr int dyn = tdls::dynamic_stride;

/// \brief The buffers of one run: the matrices, the pivots, the
/// right-hand sides and the solutions, interleaved by WA, WP and WY, the
/// columns of b and x N * WY apart.
/// \tparam T scalar type
template<typename T>
struct Buffers {
    std::vector<T> A;     ///< matrices
    std::vector<int> piv; ///< pivots
    std::vector<T> b;     ///< right-hand sides
    std::vector<T> x;     ///< solutions
    std::vector<int> ok;  ///< verdicts
    std::vector<int> oot; ///< out-of-tile counts
};

/// \brief The buffers of a batch, the free lanes as NaN (-1 for the
/// pivots).
/// \tparam T scalar type
/// \param[in] batch the systems
/// \param[in] N     dimension
/// \return the buffers, x a copy of b
template<typename T>
Buffers<T> fill(const tdls_tests::SystemBatch<T>& batch, const int N) {
    const T nan          = std::numeric_limits<T>::quiet_NaN();
    const std::size_t n  = static_cast<std::size_t>(N);
    const std::size_t ny = n * WY * nrhs;
    Buffers<T> u{std::vector<T>(n * n * WA, nan), std::vector<int>(n * WP, -1),
                 std::vector<T>(ny, nan),         std::vector<T>(ny, nan),
                 std::vector<int>(count, -1),     std::vector<int>(count, -1)};
    for (int s = 0; s < count; ++s) {
        for (int e = 0; e < N * N; ++e)
            u.A[static_cast<std::size_t>(e) * WA + s] = batch.matrix(s)[e];
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < N; ++i) {
                const std::size_t e = (static_cast<std::size_t>(m) * n + i) * WY + s;
                u.b[e]              = batch.rhs(s)[i] * T(m + 1);
                u.x[e]              = u.b[e];
            }
    }
    return u;
}

/// \brief Bitwise equality of two runs.
/// \tparam T scalar type
/// \param[in] a first run
/// \param[in] b second run
template<typename T>
void same(const Buffers<T>& a, const Buffers<T>& b) {
    TDLS_CHECK_BITWISE(a.A.data(), b.A.data(), a.A.size());
    TDLS_CHECK_BITWISE(a.piv.data(), b.piv.data(), a.piv.size());
    TDLS_CHECK_BITWISE(a.b.data(), b.b.data(), a.b.size());
    TDLS_CHECK_BITWISE(a.x.data(), b.x.data(), a.x.size());
    TDLS_CHECK_BITWISE(a.ok.data(), b.ok.data(), a.ok.size());
    TDLS_CHECK_BITWISE(a.oot.data(), b.oot.data(), a.oot.size());
}

/// \brief Runs a call on every system of a copy of the buffers.
/// \tparam T  scalar type
/// \tparam Fn callable (A, piv, b, x, ok, oot)
/// \param[in] u  the buffers
/// \param[in] fn the call
/// \return the buffers after the calls
template<typename T, typename Fn>
Buffers<T> run(const Buffers<T>& u, Fn&& fn) {
    Buffers<T> v = u;
    for (int s = 0; s < count; ++s)
        fn(v.A.data() + s, v.piv.data() + s, v.b.data() + s, v.x.data() + s,
           v.ok[static_cast<std::size_t>(s)], v.oot[static_cast<std::size_t>(s)]);
    return v;
}

/// \brief Compares every entry point with its compile-time strides against
/// its runtime strides.
/// \tparam T      scalar type
/// \tparam N      dimension
/// \tparam Config configuration
/// \param[in] seed generator seed
template<typename T, int N, tdls::TiledLUppConfig<T> Config>
void strides_case(const std::uint64_t seed) {
    using S                  = tdls::TiledLUppSolverStatic<T, N, Config>;
    constexpr std::size_t xc = static_cast<std::size_t>(N) * WY;
    const auto batch         = tdls_tests::make_batch<T>(N, count, seed, 0.5);
    const Buffers<T> u       = fill(batch, N);
    int solved               = 0;

    // factorize, both overloads, and the forward substitution folded in
    same(run(u, [](T* A, int* p, T*, T*, int& ok,
                   int& oot) { ok = S::template factorize<false, false>(A, WA, p, WP, oot); }),
         run(u, [](T* A, int* p, T*, T*, int& ok, int& oot) {
             ok = S::template factorize<false, false, true, false, true, WA, WP>(A, 1, p, 1, oot);
         }));
    same(run(u, [](T* A, int* p, T*, T*, int& ok,
                   int&) { ok = S::template factorize<false, false>(A, WA, p, WP); }),
         run(u, [](T* A, int* p, T*, T*, int& ok, int&) {
             ok = S::template factorize<false, false, WA, WP>(A, 1, p, 1);
         }));
    same(run(u,
             [](T* A, int* p, T*, T* x, int& ok, int& oot) {
                 ok = S::template factorize<false, false, true, true, false>(A, WA, p, WP, oot, x,
                                                                             WY);
             }),
         run(u, [](T* A, int* p, T*, T* x, int& ok, int& oot) {
             ok = S::template factorize<false, false, true, true, false, WA, WP, WY>(A, 1, p, 1,
                                                                                     oot, x, 1);
         }));

    // solve, solve_inplace and their multi right-hand-side forms
    const auto r1 = run(u, [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
        ok = S::template solve<false, false, false>(A, WA, p, WP, b, x, WY, oot);
    });
    same(r1, run(u, [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
             ok =
                 S::template solve<false, false, false, true, WA, WP, WY>(A, 1, p, 1, b, x, 1, oot);
         }));
    same(run(u, [](T* A, int* p, T* b, T* x, int& ok,
                   int&) { ok = S::template solve<false, false, false>(A, WA, p, WP, b, x, WY); }),
         run(u, [](T* A, int* p, T* b, T* x, int& ok, int&) {
             ok = S::template solve<false, false, false, WA, WP, WY>(A, 1, p, 1, b, x, 1);
         }));
    same(run(u,
             [](T* A, int* p, T*, T* x, int& ok, int& oot) {
                 ok = S::template solve_inplace<false, false, false>(A, WA, p, WP, x, WY, oot);
             }),
         run(u, [](T* A, int* p, T*, T* x, int& ok, int& oot) {
             ok = S::template solve_inplace<false, false, false, true, WA, WP, WY>(A, 1, p, 1, x, 1,
                                                                                   oot);
         }));
    same(run(u,
             [](T* A, int* p, T*, T* x, int& ok, int&) {
                 ok = S::template solve_inplace<false, false, false>(A, WA, p, WP, x, WY);
             }),
         run(u, [](T* A, int* p, T*, T* x, int& ok, int&) {
             ok = S::template solve_inplace<false, false, false, WA, WP, WY>(A, 1, p, 1, x, 1);
         }));
    same(run(u,
             [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
                 ok = S::template solve_multirhs<nrhs, false, false, false>(A, WA, p, WP, b, x, WY,
                                                                            xc, oot);
             }),
         run(u, [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
             ok = S::template solve_multirhs<nrhs, false, false, false, 0, true, WA, WP, WY>(
                 A, 1, p, 1, b, x, 1, xc, oot);
         }));
    same(run(u,
             [](T* A, int* p, T* b, T* x, int& ok, int&) {
                 ok = S::template solve_multirhs<nrhs, false, false, false>(A, WA, p, WP, b, x, WY,
                                                                            xc);
             }),
         run(u, [](T* A, int* p, T* b, T* x, int& ok, int&) {
             ok = S::template solve_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(A, 1, p, 1,
                                                                                       b, x, 1, xc);
         }));
    same(run(u,
             [](T* A, int* p, T*, T* x, int& ok, int& oot) {
                 ok = S::template solve_inplace_multirhs<nrhs, false, false, false>(A, WA, p, WP, x,
                                                                                    WY, xc, oot);
             }),
         run(u, [](T* A, int* p, T*, T* x, int& ok, int& oot) {
             ok =
                 S::template solve_inplace_multirhs<nrhs, false, false, false, 0, true, WA, WP, WY>(
                     A, 1, p, 1, x, 1, xc, oot);
         }));
    same(run(u,
             [](T* A, int* p, T*, T* x, int& ok, int&) {
                 ok = S::template solve_inplace_multirhs<nrhs, false, false, false>(A, WA, p, WP, x,
                                                                                    WY, xc);
             }),
         run(u, [](T* A, int* p, T*, T* x, int& ok, int&) {
             ok = S::template solve_inplace_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(
                 A, 1, p, 1, x, 1, xc);
         }));
    // a single stride at compile time, the others at run time
    same(r1, run(u, [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
             ok = S::template solve<false, false, false, true, WA>(A, 1, p, WP, b, x, WY, oot);
         }));
    same(r1, run(u, [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
             ok = S::template solve<false, false, false, true, dyn, WP>(A, WA, p, 1, b, x, WY, oot);
         }));
    same(r1, run(u, [](T* A, int* p, T* b, T* x, int& ok, int& oot) {
             ok = S::template solve<false, false, false, true, dyn, dyn, WY>(A, WA, p, WP, b, x, 1,
                                                                             oot);
         }));

    // the substitutions, on the factors of the runtime factorization
    const Buffers<T> f = run(u, [](T* A, int* p, T*, T*, int& ok, int& oot) {
        ok = S::template factorize<false, false>(A, WA, p, WP, oot);
    });
    for (const int ok : f.ok)
        solved += ok;
    same(run(f, [](T* A, int* p, T* b, T* x, int&,
                   int&) { S::template substitute<false, false, false>(A, WA, p, WP, b, x, WY); }),
         run(f, [](T* A, int* p, T* b, T* x, int&, int&) {
             S::template substitute<false, false, false, WA, WP, WY>(A, 1, p, 1, b, x, 1);
         }));
    same(run(f,
             [](T* A, int* p, T*, T* x, int&, int&) {
                 S::template substitute_inplace<false, false, false>(A, WA, p, WP, x, WY);
             }),
         run(f, [](T* A, int* p, T*, T* x, int&, int&) {
             S::template substitute_inplace<false, false, false, WA, WP, WY>(A, 1, p, 1, x, 1);
         }));
    same(run(f,
             [](T* A, int* p, T*, T* x, int&, int&) {
                 S::template substitute_canonical<false, false, false>(A, WA, p, WP, 1, x, WY);
             }),
         run(f, [](T* A, int* p, T*, T* x, int&, int&) {
             S::template substitute_canonical<false, false, false, WA, WP, WY>(A, 1, p, 1, 1, x, 1);
         }));
    same(run(f,
             [](T* A, int* p, T*, T* x, int&, int&) {
                 S::template substitute_canonical_multirhs<nrhs, false, false, false>(A, WA, p, WP,
                                                                                      1, x, WY, xc);
             }),
         run(f, [](T* A, int* p, T*, T* x, int&, int&) {
             S::template substitute_canonical_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(
                 A, 1, p, 1, 1, x, 1, xc);
         }));
    same(run(f,
             [](T* A, int* p, T* b, T* x, int&, int&) {
                 S::template substitute_multirhs<nrhs, false, false, false>(A, WA, p, WP, b, x, WY,
                                                                            xc);
             }),
         run(f, [](T* A, int* p, T* b, T* x, int&, int&) {
             S::template substitute_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(A, 1, p, 1,
                                                                                       b, x, 1, xc);
         }));
    same(run(f,
             [](T* A, int* p, T*, T* x, int&, int&) {
                 S::template substitute_inplace_multirhs<nrhs, false, false, false>(A, WA, p, WP, x,
                                                                                    WY, xc);
             }),
         run(f, [](T* A, int* p, T*, T* x, int&, int&) {
             S::template substitute_inplace_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(
                 A, 1, p, 1, x, 1, xc);
         }));
    same(run(f,
             [](T* A, int* p, T*, T* x, int&, int&) {
                 S::template forward_inplace_multirhs<nrhs, false, false, false>(A, WA, p, WP, x,
                                                                                 WY, xc);
                 S::template backward_inplace_multirhs<nrhs, false, false, false>(A, WA, p, WP, x,
                                                                                  WY, xc);
             }),
         run(f, [](T* A, int* p, T*, T* x, int&, int&) {
             S::template forward_inplace_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(
                 A, 1, p, 1, x, 1, xc);
             S::template backward_inplace_multirhs<nrhs, false, false, false, 0, WA, WP, WY>(
                 A, 1, p, 1, x, 1, xc);
         }));
    // Floor: the batch is invertible.
    TDLS_CHECK(solved == count);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/bridge/static_strides/double/N=7,tile_size=3,RL") {
    strides_case<double, 7, tdls::TiledLUppConfig<double>{.tile_size = 3}>(150100);
}
TDLS_TEST_CASE("tiledlupp/bridge/static_strides/double/N=7,tile_size=3,LL") {
    strides_case<double, 7,
                 tdls::TiledLUppConfig<double>{.tile_size = 3,
                                               .schedule  = tdls::Schedule::LeftLooking}>(150200);
}
TDLS_TEST_CASE("tiledlupp/bridge/static_strides/double/N=7,tile_size=3,col") {
    strides_case<double, 7,
                 tdls::TiledLUppConfig<double>{.tile_size = 3,
                                               .layout    = tdls::MatrixLayout::ColMajor}>(150300);
}
TDLS_TEST_CASE("tiledlupp/bridge/static_strides/double/N=5,physical") {
    strides_case<double, 5,
                 tdls::TiledLUppConfig<double>{.row_interchange = tdls::RowInterchange::Physical}>(
        150400);
}
TDLS_TEST_CASE("tiledlupp/bridge/static_strides/float/N=7,tile_size=3,RL,unroll off") {
    strides_case<float, 7, tdls::TiledLUppConfig<float>{.tile_size = 3, .unroll_loops = false}>(
        150500);
}

TDLS_TEST_MAIN
