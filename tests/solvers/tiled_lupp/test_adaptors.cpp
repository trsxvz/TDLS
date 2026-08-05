/// \file
/// \brief Suite of the generic dense adaptors.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. Released under the
/// BSD 3-Clause License (see the LICENSE file).
///
/// The adaptors are structural: they accept any type exposing the dense
/// contract, without naming any external library. The suite exercises the
/// three recognized data() shapes through minimal mock types (contiguous
/// object, strided view returning a (pointer, stride) pair, strided view
/// with a separate getStride()), checks the residency classification at
/// compile time, rejects a gather-like type, and requires every adaptor
/// call to reproduce the raw pointer API bitwise.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

constexpr int N = 12;

/// \brief Indexing policy of the matrix mocks, row-major contiguous.
struct MockMatrixPolicy {
    using size_type            = int;
    static constexpr int arity = 2;
    constexpr int size(const int d) const {
        return d >= 0 ? N : N;
    }
    constexpr int getIndex(const int i, const int j) const {
        return i * N + j;
    }
};

/// \brief Indexing policy of the vector mocks, contiguous.
struct MockVectorPolicy {
    using size_type            = int;
    static constexpr int arity = 1;
    constexpr int size(const int) const {
        return N;
    }
    constexpr int getIndex(const int i) const {
        return i;
    }
};

/// \brief Contiguous matrix mock: data() returns a plain pointer.
struct MockMatrix {
    using indexing_policy = MockMatrixPolicy;
    double v[N * N];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

/// \brief Contiguous vector mock.
struct MockVector {
    using indexing_policy = MockVectorPolicy;
    double v[N];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

/// \brief Contiguous int vector mock, used as a dense pivot.
struct MockIntVector {
    using indexing_policy = MockVectorPolicy;
    int v[N];
    int* data() {
        return v;
    }
    const int* data() const {
        return v;
    }
};

/// \brief Strided matrix view mock: data() returns a (pointer, stride)
/// pair, the shape adopted by tfel::math::StridedCoalescedView.
struct MockPairMatrixView {
    using indexing_policy = MockMatrixPolicy;
    double* pointer;
    std::size_t stride_value;
    std::pair<double*, std::size_t> data() const {
        return {pointer, stride_value};
    }
};

/// \brief Strided vector view mock: plain data() plus a separate
/// getStride().
struct MockGetStrideVectorView {
    using indexing_policy = MockVectorPolicy;
    double* pointer;
    int stride_value;
    double* data() const {
        return pointer;
    }
    int getStride() const {
        return stride_value;
    }
};

/// \brief Gather-like mock: one pointer per element and no data() at
/// all, hence outside the dense contract.
struct MockGatherView {
    using indexing_policy = MockMatrixPolicy;
    double* pointers[N * N];
};

//! \brief column count of the fixed-size right-hand-side blocks
constexpr int M = 5;

/// \brief Indexing policy of the fixed-size right-hand-side block mock:
/// N x M, row-major contiguous: the shape of a tmatrix<N, M>.
struct MockRhsMatrixPolicy {
    using size_type            = int;
    static constexpr int arity = 2;
    constexpr int size(const int d) const {
        return d == 0 ? N : M;
    }
    constexpr int getIndex(const int i, const int j) const {
        return i * M + j;
    }
};

/// \brief Fixed-size right-hand-side block mock, one system per column.
struct MockRhsMatrix {
    using indexing_policy = MockRhsMatrixPolicy;
    double v[N * M];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

/// \brief Indexing policy of the runtime matrix mock: extents as data
/// members, zero by default: the shape of tfel::math::matrix.
struct MockRuntimeMatrixPolicy {
    using size_type            = std::size_t;
    static constexpr int arity = 2;
    std::size_t rows           = 0;
    std::size_t cols           = 0;
    constexpr std::size_t size(const int d) const {
        return d == 0 ? rows : cols;
    }
    constexpr std::size_t getIndex(const std::size_t i, const std::size_t j) const {
        return i * cols + j;
    }
};

/// \brief Indexing policy of the runtime vector mock.
struct MockRuntimeVectorPolicy {
    using size_type            = std::size_t;
    static constexpr int arity = 1;
    std::size_t len            = 0;
    constexpr std::size_t size(const int) const {
        return len;
    }
    constexpr std::size_t getIndex(const std::size_t i) const {
        return i;
    }
};

/// \brief Runtime-sized matrix mock: heap storage, policy instance.
struct MockRuntimeMatrix {
    using indexing_policy = MockRuntimeMatrixPolicy;
    std::vector<double> v;
    MockRuntimeMatrixPolicy policy;
    explicit MockRuntimeMatrix(const std::size_t n) : v(n * n), policy{n, n} {
    }
    MockRuntimeMatrix(const std::size_t r, const std::size_t c) : v(r * c), policy{r, c} {
    }
    double* data() {
        return v.data();
    }
    const double* data() const {
        return v.data();
    }
    MockRuntimeMatrixPolicy getIndexingPolicy() const {
        return policy;
    }
};

/// \brief Runtime-sized vector mock.
struct MockRuntimeVector {
    using indexing_policy = MockRuntimeVectorPolicy;
    std::vector<double> v;
    MockRuntimeVectorPolicy policy;
    explicit MockRuntimeVector(const std::size_t n) : v(n), policy{n} {
    }
    double* data() {
        return v.data();
    }
    const double* data() const {
        return v.data();
    }
    MockRuntimeVectorPolicy getIndexingPolicy() const {
        return policy;
    }
};

// Compile-time classification contract.
static_assert(!tdls::detail::is_dense_v<MockGatherView>);
static_assert(tdls::storage_traits<MockMatrix>::is_internal);
static_assert(tdls::storage_traits<MockVector>::is_internal);
static_assert(!tdls::storage_traits<MockPairMatrixView>::is_internal);
static_assert(!tdls::storage_traits<MockGetStrideVectorView>::is_internal);
static_assert(!tdls::storage_traits<MockMatrix>::has_runtime_extents);
static_assert(!tdls::storage_traits<MockVector>::has_runtime_extents);
static_assert(tdls::storage_traits<MockRuntimeMatrix>::has_runtime_extents);
static_assert(tdls::storage_traits<MockRuntimeVector>::has_runtime_extents);

using RawSolver = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;

} // namespace

TDLS_TEST_CASE("tiledlupp/adaptors/contiguous-mocks-reproduce-raw-internal") {
    tdls_tests::UniformGenerator gen(210100, 0.5);
    for (int repeat = 0; repeat < 100; ++repeat) {
        MockMatrix A;
        MockVector b, x;
        double A_raw[N * N], b_raw[N], x_raw[N];
        int piv[N], piv_raw[N];
        for (int e = 0; e < N * N; ++e)
            A.v[e] = A_raw[e] = gen.next();
        for (int i = 0; i < N; ++i)
            b.v[i] = b_raw[i] = gen.next();
        const bool ok = tdls::solve(A, piv, b, x);
        const bool ok_raw =
            RawSolver::solve<true, true, true>(A_raw, 1, piv_raw, 1, b_raw, x_raw, 1);
        TDLS_CHECK(ok == ok_raw);
        TDLS_CHECK_BITWISE(A.v, A_raw, static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(piv, piv_raw, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x.v, x_raw, static_cast<std::size_t>(N));
    }
}

TDLS_TEST_CASE("tiledlupp/adaptors/strided-mocks-reproduce-raw-external") {
    // Matrix through the pair shape, vectors through the getStride shape,
    // all mapped on one SoA batch.
    constexpr int count = 64;
    tdls_tests::UniformGenerator gen(210200, 0.5);
    std::vector<double> gA(static_cast<std::size_t>(N) * N * count);
    std::vector<double> gb(static_cast<std::size_t>(N) * count);
    std::vector<double> gx(static_cast<std::size_t>(N) * count, 0.0);
    for (auto& v : gA)
        v = gen.next();
    for (auto& v : gb)
        v = gen.next();
    std::vector<double> gA_raw(gA), gx_raw(static_cast<std::size_t>(N) * count, 0.0);

    for (int s = 0; s < count; ++s) {
        MockPairMatrixView A{gA.data() + s, static_cast<std::size_t>(count)};
        MockGetStrideVectorView b{gb.data() + s, count}, x{gx.data() + s, count};
        int piv[N], piv_raw[N];
        const bool ok = tdls::solve(A, piv, b, x);
        const bool ok_raw =
            RawSolver::factorize<true, false>(gA_raw.data() + s, count, piv_raw, 1) &&
            (RawSolver::substitute<false, true, false>(gA_raw.data() + s, count, piv_raw, 1,
                                                       gb.data() + s, gx_raw.data() + s, count),
             true);
        TDLS_CHECK(ok == ok_raw);
        TDLS_CHECK_BITWISE(piv, piv_raw, static_cast<std::size_t>(N));
    }
    TDLS_CHECK_BITWISE(gA.data(), gA_raw.data(), gA.size());
    TDLS_CHECK_BITWISE(gx.data(), gx_raw.data(), gx.size());
}

TDLS_TEST_CASE("tiledlupp/adaptors/dense-int-pivot") {
    tdls_tests::UniformGenerator gen(210300, 0.5);
    MockMatrix A;
    double A_raw[N * N];
    MockIntVector piv;
    int piv_raw[N];
    for (int e = 0; e < N * N; ++e)
        A.v[e] = A_raw[e] = gen.next();
    const bool ok     = tdls::factorize(A, piv);
    const bool ok_raw = RawSolver::factorize<true, true>(A_raw, 1, piv_raw, 1);
    TDLS_CHECK(ok == ok_raw);
    TDLS_CHECK_BITWISE(A.v, A_raw, static_cast<std::size_t>(N) * N);
    TDLS_CHECK_BITWISE(piv.v, piv_raw, static_cast<std::size_t>(N));
}

TDLS_TEST_CASE("tiledlupp/adaptors/runtime-sized-mocks-route-to-the-dynamic-solver") {
    // A runtime-sized matrix resolves TiledLUppSolverDynamic: every call
    // must reproduce the raw runtime API bitwise. The pivot mixes both
    // accepted shapes (raw int pointer, then dense int storage through
    // the same raw pointer path).
    using RawDynamic = tdls::TiledLUppSolverDynamic<double>;
    const int n      = 12;
    tdls_tests::UniformGenerator gen(210500, 0.5);
    for (int repeat = 0; repeat < 50; ++repeat) {
        MockRuntimeMatrix A(n);
        MockRuntimeVector b(n), x(n);
        std::vector<double> A_raw(static_cast<std::size_t>(n) * n), b_raw(n), x_raw(n);
        std::vector<int> piv(n), piv_raw(n);
        for (int e = 0; e < n * n; ++e)
            A.v[e] = A_raw[e] = gen.next();
        for (int i = 0; i < n; ++i)
            b.v[i] = b_raw[i] = gen.next();

        int* piv_p    = piv.data();
        const bool ok = tdls::solve(A, piv_p, b, x);
        const bool ok_raw =
            RawDynamic::solve(n, A_raw.data(), 1, piv_raw.data(), 1, b_raw.data(), x_raw.data(), 1);
        TDLS_CHECK(ok == ok_raw);
        TDLS_CHECK_BITWISE(A.v.data(), A_raw.data(), static_cast<std::size_t>(n) * n);
        TDLS_CHECK_BITWISE(piv.data(), piv_raw.data(), static_cast<std::size_t>(n));
        TDLS_CHECK_BITWISE(x.v.data(), x_raw.data(), static_cast<std::size_t>(n));

        // The factored matrix is reusable: canonical columns and the
        // in-place substitution must reproduce the raw calls too.
        MockRuntimeVector z(n);
        std::vector<double> z_raw(n);
        for (int c = 0; c < n; ++c) {
            tdls::substitute_canonical(A, piv_p, c, z);
            RawDynamic::substitute_canonical(n, A_raw.data(), 1, piv_raw.data(), 1, c, z_raw.data(),
                                             1);
            TDLS_CHECK_BITWISE(z.v.data(), z_raw.data(), static_cast<std::size_t>(n));
        }
        for (int i = 0; i < n; ++i) {
            z.v[i]   = b.v[i];
            z_raw[i] = b_raw[i];
        }
        tdls::substitute_inplace(A, piv_p, z);
        RawDynamic::substitute_inplace(n, A_raw.data(), 1, piv_raw.data(), 1, z_raw.data(), 1);
        TDLS_CHECK_BITWISE(z.v.data(), z_raw.data(), static_cast<std::size_t>(n));
    }

    // The fused path on fresh systems.
    MockRuntimeMatrix A2(n);
    MockRuntimeVector y2(n);
    std::vector<double> A2_raw(static_cast<std::size_t>(n) * n), y2_raw(n);
    std::vector<int> piv2(n), piv2_raw(n);
    for (int e = 0; e < n * n; ++e)
        A2.v[e] = A2_raw[e] = gen.next();
    for (int i = 0; i < n; ++i)
        y2.v[i] = y2_raw[i] = gen.next();
    int* piv2_p         = piv2.data();
    const bool ok_fused = tdls::solve_inplace(A2, piv2_p, y2);
    const bool ok_fused_raw =
        RawDynamic::solve_inplace(n, A2_raw.data(), 1, piv2_raw.data(), 1, y2_raw.data(), 1);
    TDLS_CHECK(ok_fused == ok_fused_raw);
    TDLS_CHECK_BITWISE(y2.v.data(), y2_raw.data(), static_cast<std::size_t>(n));
}

TDLS_TEST_CASE("tiledlupp/adaptors/matrix-rhs-routes-to-the-multirhs-entry-points") {
    // A matrix-like right-hand side (arity 2) is solved column by column
    // against one factorization, through the _multirhs raw entry points in
    // external addressing (row stride M, column stride 1 for the
    // row-major mocks). Every call must reproduce the raw block API
    // bitwise, whatever the W cutting.
    tdls_tests::UniformGenerator gen(210600, 0.5);
    MockMatrix A;
    MockRhsMatrix B, X;
    double A_raw[N * N], X_raw[N * M];
    int piv[N], piv_raw[N];
    for (int e = 0; e < N * N; ++e)
        A.v[e] = A_raw[e] = gen.next();
    for (int e = 0; e < N * M; ++e)
        B.v[e] = gen.next();

    // One-call solve, all columns in one pass.
    TDLS_CHECK(tdls::solve(A, piv, B, X));
    TDLS_CHECK(
        (RawSolver::solve_multirhs<M, false, true, true>(A_raw, 1, piv_raw, 1, B.v, X_raw, M, 1)));
    TDLS_CHECK_BITWISE(A.v, A_raw, static_cast<std::size_t>(N) * N);
    TDLS_CHECK_BITWISE(piv, piv_raw, static_cast<std::size_t>(N));
    TDLS_CHECK_BITWISE(X.v, X_raw, static_cast<std::size_t>(N) * M);

    // W cutting: passes of 2 columns plus a remainder of 1 must land on
    // the same solutions bitwise (per-column arithmetic is identical).
    MockRhsMatrix Xw;
    tdls::substitute<void, 2>(A, piv, B, Xw);
    TDLS_CHECK_BITWISE(Xw.v, X_raw, static_cast<std::size_t>(N) * M);

    // In-place block on the factored matrix.
    MockRhsMatrix Y;
    double Y_raw[N * M];
    for (int e = 0; e < N * M; ++e)
        Y.v[e] = Y_raw[e] = B.v[e];
    tdls::substitute_inplace(A, piv, Y);
    RawSolver::substitute_inplace_multirhs<M, false, true, true>(A_raw, 1, piv_raw, 1, Y_raw, M, 1);
    TDLS_CHECK_BITWISE(Y.v, Y_raw, static_cast<std::size_t>(N) * M);

    // One-call in-place solve on a fresh system, with a W cutting.
    MockMatrix A2;
    double A2_raw[N * N];
    for (int e = 0; e < N * N; ++e)
        A2.v[e] = A2_raw[e] = A.v[e] * 0.5 + 0.25;
    for (int e = 0; e < N * M; ++e)
        Y.v[e] = Y_raw[e] = B.v[e];
    TDLS_CHECK((tdls::solve_inplace<void, 3>(A2, piv, Y)));
    TDLS_CHECK((RawSolver::solve_inplace_multirhs<M, false, true, true, 3>(A2_raw, 1, piv_raw, 1,
                                                                           Y_raw, M, 1)));
    TDLS_CHECK_BITWISE(Y.v, Y_raw, static_cast<std::size_t>(N) * M);
}

TDLS_TEST_CASE("tiledlupp/adaptors/runtime-matrix-rhs-routes-to-the-dynamic-multirhs") {
    // Runtime-sized matrix right-hand sides: the column count is read on
    // the object and the dynamic _multirhs entry points are resolved. Every
    // call must reproduce the raw runtime block API bitwise.
    using RawDynamic = tdls::TiledLUppSolverDynamic<double>;
    const int n = 12, m = 5;
    tdls_tests::UniformGenerator gen(210700, 0.5);
    MockRuntimeMatrix A(n), B(n, m), X(n, m);
    std::vector<double> A_raw(static_cast<std::size_t>(n) * n),
        X_raw(static_cast<std::size_t>(n) * m);
    std::vector<int> piv(n), piv_raw(n);
    for (int e = 0; e < n * n; ++e)
        A.v[e] = A_raw[e] = gen.next();
    for (int e = 0; e < n * m; ++e)
        B.v[e] = gen.next();

    int* piv_p = piv.data();
    TDLS_CHECK(tdls::solve(A, piv_p, B, X));
    TDLS_CHECK(RawDynamic::solve_multirhs(n, m, 0, A_raw.data(), 1, piv_raw.data(), 1, B.v.data(),
                                          X_raw.data(), m, 1));
    TDLS_CHECK_BITWISE(A.v.data(), A_raw.data(), static_cast<std::size_t>(n) * n);
    TDLS_CHECK_BITWISE(piv.data(), piv_raw.data(), static_cast<std::size_t>(n));
    TDLS_CHECK_BITWISE(X.v.data(), X_raw.data(), static_cast<std::size_t>(n) * m);

    // W cutting on the runtime path.
    MockRuntimeMatrix Xw(n, m);
    tdls::substitute<void, 2>(A, piv_p, B, Xw);
    TDLS_CHECK_BITWISE(Xw.v.data(), X_raw.data(), static_cast<std::size_t>(n) * m);

    // In-place block on the factored matrix.
    MockRuntimeMatrix Y(n, m);
    std::vector<double> Y_raw(B.v);
    Y.v = B.v;
    tdls::substitute_inplace(A, piv_p, Y);
    RawDynamic::substitute_inplace_multirhs(n, m, A_raw.data(), 1, piv_raw.data(), 1, Y_raw.data(),
                                            m, 1);
    TDLS_CHECK_BITWISE(Y.v.data(), Y_raw.data(), static_cast<std::size_t>(n) * m);
}

TDLS_TEST_CASE("tiledlupp/adaptors/substitution-entry-points-reproduce-raw") {
    tdls_tests::UniformGenerator gen(210400, 0.5);
    MockMatrix A;
    MockVector b, x;
    double A_raw[N * N], b_raw[N], x_raw[N];
    int piv[N], piv_raw[N];
    for (int e = 0; e < N * N; ++e)
        A.v[e] = A_raw[e] = gen.next();
    for (int i = 0; i < N; ++i)
        b.v[i] = b_raw[i] = gen.next();

    TDLS_CHECK(tdls::factorize(A, piv));
    TDLS_CHECK((RawSolver::factorize<true, true>(A_raw, 1, piv_raw, 1)));

    tdls::substitute(A, piv, b, x);
    RawSolver::substitute<true, true, true>(A_raw, 1, piv_raw, 1, b_raw, x_raw, 1);
    TDLS_CHECK_BITWISE(x.v, x_raw, static_cast<std::size_t>(N));

    MockVector y;
    double y_raw[N];
    for (int i = 0; i < N; ++i) {
        y.v[i]   = b.v[i];
        y_raw[i] = b_raw[i];
    }
    tdls::substitute_inplace(A, piv, y);
    RawSolver::substitute_inplace<true, true, true>(A_raw, 1, piv_raw, 1, y_raw, 1);
    TDLS_CHECK_BITWISE(y.v, y_raw, static_cast<std::size_t>(N));

    for (int c = 0; c < N; ++c) {
        tdls::substitute_canonical(A, piv, c, x);
        RawSolver::substitute_canonical<true, true, true>(A_raw, 1, piv_raw, 1, c, x_raw, 1);
        TDLS_CHECK_BITWISE(x.v, x_raw, static_cast<std::size_t>(N));
    }

    MockMatrix A2;
    double A2_raw[N * N];
    for (int e = 0; e < N * N; ++e)
        A2.v[e] = A2_raw[e] = A.v[e] * 0.5 + 0.25;
    MockVector z;
    double z_raw[N];
    for (int i = 0; i < N; ++i) {
        z.v[i]   = b.v[i];
        z_raw[i] = b_raw[i];
    }
    const bool ok_fused = tdls::solve_inplace(A2, piv, z);
    const bool ok_fused_raw =
        RawSolver::solve_inplace<true, true, true>(A2_raw, 1, piv_raw, 1, z_raw, 1);
    TDLS_CHECK(ok_fused == ok_fused_raw);
    TDLS_CHECK_BITWISE(z.v, z_raw, static_cast<std::size_t>(N));
}

TDLS_TEST_MAIN
