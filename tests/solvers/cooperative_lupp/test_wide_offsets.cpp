/// \file
/// \brief Bridge suite: element offsets beyond 2^32.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The offset of a strided element, index times stride, is computed in
/// 64-bit arithmetic: a batch may hold more than 2^32 elements. Each cell
/// solves one system, a group of one thread, spread over tens of
/// gigabytes of a lazily backed arena (huge_arena.hpp) by strides that
/// take the offsets of the matrix, the pivots and the right-hand side
/// past 2^32, and checks the results bitwise against the same system
/// stored contiguously (stride 1). A 32-bit product would wrap these
/// offsets around onto other elements. Both solvers, both layouts.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "huge_arena.hpp"

namespace {

constexpr int N = 4;
/// Matrix stride: the last element, index N * N - 1 = 15, lands at
/// 15 * (2^29 + 1) > 2^32.
constexpr int matrix_stride = (1 << 29) + 1;
/// Vector stride: the last entry, index 3, lands at 3 * (2^31 - 1) > 2^32.
constexpr int vector_stride = 2147483647;

/// \brief Element e of a strided arena.
template<typename T>
T& at(const tdls_tests::HugeArena<T>& arena, const int e, const int stride) {
    return arena.data()[static_cast<std::size_t>(e) * static_cast<std::size_t>(stride)];
}

/// \brief Solves one system twice, stored contiguously and spread by the
/// wide strides, and compares the factored matrix, the pivots and the
/// solution bitwise.
/// \tparam Layout  matrix layout
/// \tparam dynamic whether the runtime solver runs instead of the
///         compile-time one
/// \param[in] seed generator seed
template<tdls::MatrixLayout Layout, bool dynamic>
void wide_case(const std::uint64_t seed) {
    constexpr auto config =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = N, .layout = Layout};
    tdls_tests::HugeArena<double> A(static_cast<std::size_t>(N * N - 1) * matrix_stride + 1);
    tdls_tests::HugeArena<int> piv(static_cast<std::size_t>(N - 1) * vector_stride + 1);
    tdls_tests::HugeArena<double> y(static_cast<std::size_t>(N - 1) * vector_stride + 1);
    if (A.data() == nullptr || piv.data() == nullptr || y.data() == nullptr) {
        std::printf("       not run: no lazily backed arena on this system\n");
        return;
    }
    tdls_tests::UniformGenerator g(seed, 1.0);
    double A_ref[N * N], y_ref[N];
    int piv_ref[N];
    for (int e = 0; e < N * N; ++e)
        A_ref[e] = g.next();
    for (int i = 0; i < N; ++i)
        A_ref[i * N + i] += N;
    for (double& v : y_ref)
        v = g.next();
    for (int e = 0; e < N * N; ++e)
        at(A, e, matrix_stride) = A_ref[e];
    for (int i = 0; i < N; ++i)
        at(y, i, vector_stride) = y_ref[i];

    bool ok_ref = false, ok_wide = false;
    if constexpr (dynamic) {
        using Solver = tdls::CooperativeLUppSolverDynamic<double, config>;
        std::vector<double> work(Solver::workspace_size(N), 0.0);
        ok_ref  = Solver::solve_inplace(N, 0, A_ref, 1, piv_ref, 1, y_ref, 1, work.data(),
                                        tdls::NoSync{});
        ok_wide = Solver::solve_inplace(N, 0, A.data(), matrix_stride, piv.data(), vector_stride,
                                        y.data(), vector_stride, work.data(), tdls::NoSync{});
    } else {
        using Solver = tdls::CooperativeLUppSolverStatic<double, N, config>;
        std::vector<double> work(Solver::workspace_size, 0.0);
        ok_ref = Solver::template solve_inplace<false, false, false>(
            0, A_ref, 1, piv_ref, 1, y_ref, 1, work.data(), tdls::NoSync{});
        ok_wide = Solver::template solve_inplace<false, false, false>(
            0, A.data(), matrix_stride, piv.data(), vector_stride, y.data(), vector_stride,
            work.data(), tdls::NoSync{});
    }
    TDLS_CHECK(ok_ref);
    TDLS_CHECK(ok_wide);
    for (int e = 0; e < N * N; ++e)
        TDLS_CHECK_BITWISE(&A_ref[e], &at(A, e, matrix_stride), 1);
    for (int i = 0; i < N; ++i) {
        TDLS_CHECK_BITWISE(&piv_ref[i], &at(piv, i, vector_stride), 1);
        TDLS_CHECK_BITWISE(&y_ref[i], &at(y, i, vector_stride), 1);
    }
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/bridge/wide_offsets/static/rowmajor") {
    wide_case<tdls::MatrixLayout::RowMajor, false>(260100);
}
TDLS_TEST_CASE("cooperativelupp/bridge/wide_offsets/static/colmajor") {
    wide_case<tdls::MatrixLayout::ColMajor, false>(260200);
}
TDLS_TEST_CASE("cooperativelupp/bridge/wide_offsets/dynamic/rowmajor") {
    wide_case<tdls::MatrixLayout::RowMajor, true>(260300);
}
TDLS_TEST_CASE("cooperativelupp/bridge/wide_offsets/dynamic/colmajor") {
    wide_case<tdls::MatrixLayout::ColMajor, true>(260400);
}

TDLS_TEST_MAIN
