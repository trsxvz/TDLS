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
/// spreads one system over tens of gigabytes of a lazily backed arena
/// (huge_arena.hpp), with strides that take the offsets of the matrix, the
/// pivots, the right-hand side and the columns of a block past 2^32, and
/// checks the results bitwise against the same system held in contiguous
/// arrays. A 32-bit product would wrap these offsets around onto other
/// elements. Both solvers, both layouts.

#include <cstdint>
#include <cstdio>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "huge_arena.hpp"

namespace {

constexpr int N    = 4;
constexpr int nrhs = 4;
/// Matrix stride: the last element, index N * N - 1 = 15, lands at
/// 15 * (2^29 + 1) > 2^32.
constexpr int matrix_stride = (1 << 29) + 1;
/// Vector and column stride: the last entry, index 3, lands at
/// 3 * (2^31 - 1) > 2^32.
constexpr int vector_stride = 2147483647;

/// \brief One reproducible, well-conditioned system and its right-hand
/// sides, column after column.
struct System {
    double A[N * N];
    double B[nrhs * N];
    explicit System(const std::uint64_t seed) {
        tdls_tests::UniformGenerator g(seed, 1.0);
        for (double& a : A)
            a = g.next();
        for (int i = 0; i < N; ++i)
            A[i * N + i] += N;
        for (double& b : B)
            b = g.next();
    }
};

/// \brief Strided arenas of one system: matrix, pivots, right-hand side,
/// and a block of nrhs columns.
struct Arenas {
    tdls_tests::HugeArena<double> A{static_cast<std::size_t>(N * N - 1) * matrix_stride + 1};
    tdls_tests::HugeArena<int> piv{static_cast<std::size_t>(N - 1) * vector_stride + 1};
    tdls_tests::HugeArena<double> y{static_cast<std::size_t>(N - 1) * vector_stride + 1};
    tdls_tests::HugeArena<double> X{static_cast<std::size_t>(nrhs - 1) * vector_stride + N};
    [[nodiscard]] bool ready() const {
        return A.data() != nullptr && piv.data() != nullptr && y.data() != nullptr &&
               X.data() != nullptr;
    }
};

/// \brief Element e of a strided arena.
template<typename T>
T& at(const tdls_tests::HugeArena<T>& arena, const int e, const int stride) {
    return arena.data()[static_cast<std::size_t>(e) * static_cast<std::size_t>(stride)];
}

/// \brief Compile-time solver: solve_inplace with every operand strided,
/// then substitute_inplace_multirhs on a block whose columns are strided,
/// against the contiguous runs.
/// \tparam Layout matrix layout
template<tdls::MatrixLayout Layout>
void static_case() {
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2, .layout = Layout};
    using Solver          = tdls::TiledLUppSolverStatic<double, N, config>;
    Arenas arenas;
    if (!arenas.ready()) {
        std::printf("       not run: no lazily backed arena on this system\n");
        return;
    }
    const System s(250100);

    double A_ref[N * N], y_ref[N];
    int piv_ref[N];
    for (int e = 0; e < N * N; ++e)
        A_ref[e] = s.A[e];
    for (int i = 0; i < N; ++i)
        y_ref[i] = s.B[i];
    const bool ok_ref =
        Solver::template solve_inplace<true, true, true>(A_ref, 1, piv_ref, 1, y_ref, 1);
    TDLS_CHECK(ok_ref);

    for (int e = 0; e < N * N; ++e)
        at(arenas.A, e, matrix_stride) = s.A[e];
    for (int i = 0; i < N; ++i)
        at(arenas.y, i, vector_stride) = s.B[i];
    const bool ok_wide = Solver::template solve_inplace<false, false, false>(
        arenas.A.data(), matrix_stride, arenas.piv.data(), vector_stride, arenas.y.data(),
        vector_stride);
    TDLS_CHECK(ok_wide);
    for (int e = 0; e < N * N; ++e)
        TDLS_CHECK_BITWISE(&A_ref[e], &at(arenas.A, e, matrix_stride), 1);
    for (int i = 0; i < N; ++i) {
        TDLS_CHECK_BITWISE(&piv_ref[i], &at(arenas.piv, i, vector_stride), 1);
        TDLS_CHECK_BITWISE(&y_ref[i], &at(arenas.y, i, vector_stride), 1);
    }

    // The block: contiguous columns against columns vector_stride apart.
    double X_ref[nrhs * N];
    for (int e = 0; e < nrhs * N; ++e)
        X_ref[e] = s.B[e];
    Solver::template substitute_inplace_multirhs<nrhs, true, true, true>(A_ref, 1, piv_ref, 1,
                                                                         X_ref, 1, 0);
    for (int w = 0; w < nrhs; ++w)
        for (int i = 0; i < N; ++i)
            arenas.X.data()[static_cast<std::size_t>(w) * vector_stride + i] = s.B[w * N + i];
    Solver::template substitute_inplace_multirhs<nrhs, false, true, true>(
        A_ref, 1, piv_ref, 1, arenas.X.data(), 1, vector_stride);
    for (int w = 0; w < nrhs; ++w)
        TDLS_CHECK_BITWISE(&X_ref[w * N],
                           &arenas.X.data()[static_cast<std::size_t>(w) * vector_stride], N);
}

/// \brief Runtime solver: the same cell as static_case.
/// \tparam Layout matrix layout
template<tdls::MatrixLayout Layout>
void dynamic_case() {
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2, .layout = Layout};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;
    Arenas arenas;
    if (!arenas.ready()) {
        std::printf("       not run: no lazily backed arena on this system\n");
        return;
    }
    const System s(250200);

    double A_ref[N * N], y_ref[N];
    int piv_ref[N];
    for (int e = 0; e < N * N; ++e)
        A_ref[e] = s.A[e];
    for (int i = 0; i < N; ++i)
        y_ref[i] = s.B[i];
    const bool ok_ref = Solver::solve_inplace(N, A_ref, 1, piv_ref, 1, y_ref, 1);
    TDLS_CHECK(ok_ref);

    for (int e = 0; e < N * N; ++e)
        at(arenas.A, e, matrix_stride) = s.A[e];
    for (int i = 0; i < N; ++i)
        at(arenas.y, i, vector_stride) = s.B[i];
    const bool ok_wide = Solver::solve_inplace(N, arenas.A.data(), matrix_stride, arenas.piv.data(),
                                               vector_stride, arenas.y.data(), vector_stride);
    TDLS_CHECK(ok_wide);
    for (int e = 0; e < N * N; ++e)
        TDLS_CHECK_BITWISE(&A_ref[e], &at(arenas.A, e, matrix_stride), 1);
    for (int i = 0; i < N; ++i) {
        TDLS_CHECK_BITWISE(&piv_ref[i], &at(arenas.piv, i, vector_stride), 1);
        TDLS_CHECK_BITWISE(&y_ref[i], &at(arenas.y, i, vector_stride), 1);
    }

    double X_ref[nrhs * N];
    for (int e = 0; e < nrhs * N; ++e)
        X_ref[e] = s.B[e];
    Solver::substitute_inplace_multirhs(N, nrhs, A_ref, 1, piv_ref, 1, X_ref, 1, N);
    for (int w = 0; w < nrhs; ++w)
        for (int i = 0; i < N; ++i)
            arenas.X.data()[static_cast<std::size_t>(w) * vector_stride + i] = s.B[w * N + i];
    Solver::substitute_inplace_multirhs(N, nrhs, A_ref, 1, piv_ref, 1, arenas.X.data(), 1,
                                        vector_stride);
    for (int w = 0; w < nrhs; ++w)
        TDLS_CHECK_BITWISE(&X_ref[w * N],
                           &arenas.X.data()[static_cast<std::size_t>(w) * vector_stride], N);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/bridge/wide_offsets/static/rowmajor") {
    static_case<tdls::MatrixLayout::RowMajor>();
}
TDLS_TEST_CASE("tiledlupp/bridge/wide_offsets/static/colmajor") {
    static_case<tdls::MatrixLayout::ColMajor>();
}
TDLS_TEST_CASE("tiledlupp/bridge/wide_offsets/dynamic/rowmajor") {
    dynamic_case<tdls::MatrixLayout::RowMajor>();
}
TDLS_TEST_CASE("tiledlupp/bridge/wide_offsets/dynamic/colmajor") {
    dynamic_case<tdls::MatrixLayout::ColMajor>();
}

TDLS_TEST_MAIN
