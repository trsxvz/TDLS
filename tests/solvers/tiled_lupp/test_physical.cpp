/// \file
/// \brief Physical row interchanges (TiledLUppConfig::row_interchange): the
/// logical scheme over a single tile, and the runtime solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// RowInterchange::Physical chooses the pivots of RowInterchange::Logical
/// over a single tile (a tile_size of at least N) and runs the same
/// factorization: the verdicts and the factors agree bitwise, the rows of
/// the physical factors in the pivoted order. Every solution has a small
/// backward error; the substitutions sum in another order than the logical
/// ones, so that the solutions agree within rounding errors. The runtime
/// solver reproduces the static one bitwise: factors, interchanges, the
/// count of weak pivots, every substitution and every solve, the strided
/// residency. Matrices that keep their rows, that pivot at every column,
/// and random ones; both layouts; float and double.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tdls/tdls.hpp>

#include "harness.hpp"

namespace {

/// \brief Backward error of x for A x = b: |A x - b|_inf / (|A|_inf |x|_inf
/// + |b|_inf), in double.
/// \tparam T scalar type
/// \param[in] n   dimension
/// \param[in] A   matrix, row-major (column-major when col)
/// \param[in] col whether A is stored by columns
/// \param[in] x   solution
/// \param[in] b   right-hand side
/// \return the backward error
template<typename T>
double backward_error(const int n, const std::vector<T>& A, const bool col, const T* x,
                      const T* b) {
    double r = 0, an = 0, xn = 0, bn = 0;
    for (int i = 0; i < n; ++i) {
        double s = -double(b[i]), row = 0;
        for (int j = 0; j < n; ++j) {
            const double a = col ? A[j * n + i] : A[i * n + j];
            s += a * double(x[j]);
            row += std::fabs(a);
        }
        r  = std::max(r, std::fabs(s));
        an = std::max(an, row);
        xn = std::max(xn, std::fabs(double(x[i])));
        bn = std::max(bn, std::fabs(double(b[i])));
    }
    return r / (an * xn + bn);
}

/// \brief One matrix of a family: 0 keeps its rows (dominant diagonal), 1
/// pivots at every column (dominant exchange matrix), 2 random.
/// \tparam T scalar type
/// \param[in]     n      dimension
/// \param[in]     family family
/// \param[in,out] g      generator
/// \return the matrix, row-major
template<typename T>
std::vector<T> family_matrix(const int n, const int family, std::mt19937_64& g) {
    std::uniform_real_distribution<double> u(-0.5, 0.5);
    std::vector<T> A(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double v = u(g);
            if (family == 0 && i == j) v += 2;
            if (family == 1 && j == n - 1 - i) v += 2;
            A[static_cast<std::size_t>(i) * n + j] = T(v);
        }
    return A;
}

/// \brief Checks one matrix against the logical scheme and the runtime
/// solver.
/// \tparam T scalar type
/// \tparam N dimension
/// \tparam L layout
/// \param[in]     family family of the matrix
/// \param[in,out] g      generator
template<typename T, int N, tdls::MatrixLayout L>
void physical_case(const int family, std::mt19937_64& g) {
    constexpr bool col = L == tdls::MatrixLayout::ColMajor;
    constexpr tdls::TiledLUppConfig<T> lc{.tile_size = N, .layout = L};
    constexpr tdls::TiledLUppConfig<T> pc{
        .tile_size = N, .row_interchange = tdls::RowInterchange::Physical, .layout = L};
    using Log              = tdls::TiledLUppSolverStatic<T, N, lc>;
    using Phy              = tdls::TiledLUppSolverStatic<T, N, pc>;
    using Dyn              = tdls::TiledLUppSolverDynamic<T, pc>;
    const double tol       = std::is_same_v<T, float> ? 1e-5 : 1e-13;
    const std::vector<T> A = family_matrix<T>(N, family, g);
    std::uniform_real_distribution<double> u(-0.5, 0.5);
    std::vector<T> b(N), B3(3 * N);
    for (auto& v : b)
        v = T(u(g));
    for (auto& v : B3)
        v = T(u(g));

    // The factors: the logical row i lives at row pl[i] of the logical
    // factors, at row i of the physical ones.
    std::vector<T> Fl = A, Fp = A, Fd = A;
    int pl[N], pp[N], pd[N];
    int cl = -1, cp = -1, cd = -1;
    const bool okl = Log::template factorize<true, true>(Fl.data(), 1, pl, 1, cl);
    const bool okp = Phy::template factorize<true, true>(Fp.data(), 1, pp, 1, cp);
    const bool okd = Dyn::factorize(N, Fd.data(), 1, pd, 1, cd);
    TDLS_CHECK(okl == okp);
    TDLS_CHECK(okp == okd);
    if (!okl) return;
    // the weak pivots, counted as by the logical scheme over one tile
    TDLS_CHECK(cp == cl);
    TDLS_CHECK(cd == cl);
    bool same = true;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            same = same && Fl[pl[i] * N + j] == Fp[i * N + j];
    TDLS_CHECK(same);
    TDLS_CHECK_BITWISE(Fp.data(), Fd.data(), static_cast<std::size_t>(N) * N);
    TDLS_CHECK_BITWISE(pp, pd, static_cast<std::size_t>(N));

    // substitute and substitute_inplace
    std::vector<T> xl(N), xp(N), xd(N);
    Log::template substitute<true, true, true>(Fl.data(), 1, pl, 1, b.data(), xl.data(), 1);
    Phy::template substitute<true, true, true>(Fp.data(), 1, pp, 1, b.data(), xp.data(), 1);
    Dyn::substitute(N, Fd.data(), 1, pd, 1, b.data(), xd.data(), 1);
    TDLS_CHECK_LE(backward_error(N, A, col, xp.data(), b.data()), tol);
    TDLS_CHECK_BITWISE(xp.data(), xd.data(), static_cast<std::size_t>(N));
    // The substitutions sum in another order than the logical ones: the
    // solutions agree within rounding errors.
    TDLS_CHECK_LE(backward_error(N, A, col, xl.data(), b.data()), tol);
    std::vector<T> yp = b;
    Phy::template substitute_inplace<true, true, true>(Fp.data(), 1, pp, 1, yp.data(), 1);
    TDLS_CHECK_BITWISE(yp.data(), xp.data(), static_cast<std::size_t>(N));

    // solve_inplace on strided storage (stride 3)
    std::vector<T> As(3 * N * N), ys(3 * N), Ad(3 * N * N), yd(3 * N);
    int ps[3 * N], pds[3 * N];
    for (int e = 0; e < N * N; ++e)
        As[3 * e] = Ad[3 * e] = A[e];
    for (int e = 0; e < N; ++e)
        ys[3 * e] = yd[3 * e] = b[e];
    TDLS_CHECK(
        (Phy::template solve_inplace<false, false, false>(As.data(), 3, ps, 3, ys.data(), 3)));
    TDLS_CHECK(Dyn::solve_inplace(N, Ad.data(), 3, pds, 3, yd.data(), 3));
    TDLS_CHECK_BITWISE(ys.data(), yd.data(), static_cast<std::size_t>(3 * N));
    for (int e = 0; e < N; ++e)
        TDLS_CHECK(ys[3 * e] == xp[e]);

    // Three right-hand sides in passes of 2; the two stages; canonical
    // columns.
    std::vector<T> Xp = B3, Xd = B3, Yp = B3;
    Phy::template substitute_inplace_multirhs<3, true, true, true, 2>(Fp.data(), 1, pp, 1,
                                                                      Xp.data(), 1, N);
    Dyn::template substitute_inplace_multirhs<2>(N, 3, Fd.data(), 1, pd, 1, Xd.data(), 1, N);
    TDLS_CHECK_BITWISE(Xp.data(), Xd.data(), static_cast<std::size_t>(3 * N));
    Phy::template forward_inplace_multirhs<3, true, true, true>(Fp.data(), 1, pp, 1, Yp.data(), 1,
                                                                N);
    Phy::template backward_inplace_multirhs<3, true, true, true>(Fp.data(), 1, pp, 1, Yp.data(), 1,
                                                                 N);
    TDLS_CHECK_BITWISE(Xp.data(), Yp.data(), static_cast<std::size_t>(3 * N));
    for (int c = 0; c < 3; ++c)
        TDLS_CHECK_LE(backward_error(N, A, col, Xp.data() + c * N, B3.data() + c * N), tol);
    std::vector<T> Cp(2 * N), Cd(2 * N);
    Phy::template substitute_canonical_multirhs<2, true, true, true>(Fp.data(), 1, pp, 1, 0,
                                                                     Cp.data(), 1, N);
    Dyn::substitute_canonical_multirhs(N, 2, Fd.data(), 1, pd, 1, 0, Cd.data(), 1, N);
    TDLS_CHECK_BITWISE(Cp.data(), Cd.data(), static_cast<std::size_t>(2 * N));

    // One canonical column, the last of the two above.
    constexpr int last = N > 1 ? 1 : 0;
    std::vector<T> cs(N), cds(N);
    Phy::template substitute_canonical<true, true, true>(Fp.data(), 1, pp, 1, last, cs.data(), 1);
    Dyn::substitute_canonical(N, Fd.data(), 1, pd, 1, last, cds.data(), 1);
    TDLS_CHECK_BITWISE(cs.data(), cds.data(), static_cast<std::size_t>(N));
    TDLS_CHECK_BITWISE(cs.data(), Cp.data() + last * N, static_cast<std::size_t>(N));

    // The other substitutions of the runtime solver and the out-of-place
    // multi-column ones, against the results above.
    std::vector<T> yds = b, Yd = B3, Mp(3 * N), Md(3 * N);
    Dyn::substitute_inplace(N, Fd.data(), 1, pd, 1, yds.data(), 1);
    TDLS_CHECK_BITWISE(yds.data(), xp.data(), static_cast<std::size_t>(N));
    Dyn::forward_inplace_multirhs(N, 3, Fd.data(), 1, pd, 1, Yd.data(), 1, N);
    Dyn::backward_inplace_multirhs(N, 3, Fd.data(), 1, pd, 1, Yd.data(), 1, N);
    TDLS_CHECK_BITWISE(Yd.data(), Xp.data(), static_cast<std::size_t>(3 * N));
    Phy::template substitute_multirhs<3, true, true, true, 2>(Fp.data(), 1, pp, 1, B3.data(),
                                                              Mp.data(), 1, N);
    Dyn::template substitute_multirhs<2>(N, 3, Fd.data(), 1, pd, 1, B3.data(), Md.data(), 1, N);
    TDLS_CHECK_BITWISE(Mp.data(), Xp.data(), static_cast<std::size_t>(3 * N));
    TDLS_CHECK_BITWISE(Md.data(), Xp.data(), static_cast<std::size_t>(3 * N));

    // The solves: the factorization and the substitutions above.
    std::vector<T> F1 = A, F2 = A, F3 = A, F4 = A, F5 = A, F6 = A;
    std::vector<T> s1(N), s2(N), S3(3 * N), S4(3 * N), S5 = B3, S6 = B3;
    int q1[N], q2[N], q3[N], q4[N], q5[N], q6[N];
    TDLS_CHECK(
        (Phy::template solve<true, true, true>(F1.data(), 1, q1, 1, b.data(), s1.data(), 1)));
    TDLS_CHECK(Dyn::solve(N, F2.data(), 1, q2, 1, b.data(), s2.data(), 1));
    TDLS_CHECK((Phy::template solve_multirhs<3, true, true, true, 2>(F3.data(), 1, q3, 1, B3.data(),
                                                                     S3.data(), 1, N)));
    TDLS_CHECK(
        Dyn::template solve_multirhs<2>(N, 3, F4.data(), 1, q4, 1, B3.data(), S4.data(), 1, N));
    TDLS_CHECK((Phy::template solve_inplace_multirhs<3, true, true, true, 2>(F5.data(), 1, q5, 1,
                                                                             S5.data(), 1, N)));
    TDLS_CHECK(Dyn::template solve_inplace_multirhs<2>(N, 3, F6.data(), 1, q6, 1, S6.data(), 1, N));
    TDLS_CHECK_BITWISE(s1.data(), xp.data(), static_cast<std::size_t>(N));
    TDLS_CHECK_BITWISE(s2.data(), xp.data(), static_cast<std::size_t>(N));
    for (const std::vector<T>* X : {&S3, &S4, &S5, &S6})
        TDLS_CHECK_BITWISE(X->data(), Xp.data(), static_cast<std::size_t>(3 * N));
}

/// \brief A column whose best candidate falls below oot_pivot_threshold,
/// above singular_floor: the physical factorizations count it, as the
/// logical scheme over one tile does, and still solve.
/// \tparam T scalar type
/// \tparam N dimension
template<typename T, int N>
void weak_pivot_case() {
    constexpr tdls::TiledLUppConfig<T> lc{.tile_size = N};
    constexpr tdls::TiledLUppConfig<T> pc{.row_interchange = tdls::RowInterchange::Physical};
    using Log    = tdls::TiledLUppSolverStatic<T, N, lc>;
    using Phy    = tdls::TiledLUppSolverStatic<T, N, pc>;
    using Dyn    = tdls::TiledLUppSolverDynamic<T, pc>;
    const T weak = Phy::oot_pivot_threshold / T(100);
    TDLS_CHECK(weak > Phy::singular_floor);
    std::vector<T> A(static_cast<std::size_t>(N) * N, T(0));
    for (int i = 0; i < N; ++i)
        A[static_cast<std::size_t>(i) * N + i] = i == N - 1 ? weak : T(1);
    std::vector<T> Fl = A, Fp = A, Fd = A;
    int pl[N], pp[N], pd[N];
    int cl = -1, cp = -1, cd = -1;
    TDLS_CHECK((Log::template factorize<true, true>(Fl.data(), 1, pl, 1, cl)));
    TDLS_CHECK((Phy::template factorize<true, true>(Fp.data(), 1, pp, 1, cp)));
    TDLS_CHECK(Dyn::factorize(N, Fd.data(), 1, pd, 1, cd));
    TDLS_CHECK(cl == 1);
    TDLS_CHECK(cp == 1);
    TDLS_CHECK(cd == 1);
}

/// \brief Every family, 8 matrices each, for the dimensions Ns.
/// \tparam T  scalar type
/// \tparam L  layout
/// \tparam Ns dimensions
/// \param[in] seed generator seed
template<typename T, tdls::MatrixLayout L, int... Ns>
void physical_grid(const std::uint64_t seed) {
    std::mt19937_64 g(seed);
    for (int family = 0; family < 3; ++family)
        for (int rep = 0; rep < 8; ++rep)
            (physical_case<T, Ns, L>(family, g), ...);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/physical/weak_pivot") {
    weak_pivot_case<double, 4>();
    weak_pivot_case<float, 4>();
    weak_pivot_case<double, 1>();
}

TDLS_TEST_CASE("tiledlupp/physical/double") {
    physical_grid<double, tdls::MatrixLayout::RowMajor, 1, 2, 3, 5, 6, 8, 10, 13>(601);
    physical_grid<double, tdls::MatrixLayout::ColMajor, 1, 2, 3, 5, 6, 8, 10, 13>(602);
}

TDLS_TEST_CASE("tiledlupp/physical/float") {
    physical_grid<float, tdls::MatrixLayout::RowMajor, 1, 3, 6, 9, 14>(603);
    physical_grid<float, tdls::MatrixLayout::ColMajor, 1, 3, 6, 9, 14>(604);
}

TDLS_TEST_MAIN
