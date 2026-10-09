/// \file
/// \brief Compile-time certificate suite of the CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every static_assert below constant-evaluates whole solves, and the
/// standard requires constant evaluation to reject undefined behaviour:
/// an out-of-bounds index, a read of an uninitialized value or a signed
/// overflow fails the compilation. Each assertion that compiles is
/// therefore a machine-checked certificate that the exercised path is
/// free of undefined behaviour. Constant evaluation runs one thread, so
/// the certificates cover the sequential path (one thread per system,
/// tdls::NoSync), whose slot logic is the one of every mapping: the 1 x 1
/// corner, even and odd dimensions, rows_per_thread above N, internal and
/// external residencies, the column-major layout, every entry point and
/// their equivalences, the no-pragma branches and the singular verdict of
/// the factorization. It covers both the compile-time and the runtime
/// CooperativeLUpp solvers, plus the static/dynamic bitwise bridge, itself
/// evaluated at compile time. The certificates are also re-run at run
/// time, so the suite reports like any other.

#include <tdls/tdls.hpp>

#include "harness.hpp"

namespace {

/// \brief Deterministic congruential step mapped to [-0.5, 0.5).
/// \param[in,out] state generator state
/// \return the next value of the stream
constexpr double lcg(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<double>(state >> 16) / 65536.0 - 0.5;
}

/// \brief Configuration of the sequential path, with the given knobs.
/// \tparam T      scalar type
/// \tparam N      system dimension
/// \tparam unroll unroll policy
/// \tparam layout matrix layout
template<typename T, int N, bool unroll = true,
         tdls::MatrixLayout layout = tdls::MatrixLayout::RowMajor>
constexpr auto sequential_config =
    tdls::CooperativeLUppConfig<T>{.rows_per_thread = N, .unroll_loops = unroll, .layout = layout};

/// \brief Normwise backward error |A0 x - b| / (|A0| |x| + |b|),
/// accumulated in double and computable during constant evaluation.
/// \tparam T  scalar type
/// \tparam N system dimension
/// \param[in] A0 original matrix, contiguous row-major
/// \param[in] x  computed solution
/// \param[in] b  right-hand side
/// \return the backward error
template<typename T, int N>
constexpr double backward_error(const T* A0, const T* x, const T* b) {
    double rmax = 0.0;
    double amax = 0.0;
    double xmax = 0.0;
    double bmax = 0.0;
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        for (int j = 0; j < N; ++j) {
            acc            = acc + static_cast<double>(A0[i * N + j]) * static_cast<double>(x[j]);
            const double a = tdls::detail::abs(static_cast<double>(A0[i * N + j]));
            if (a > amax) amax = a;
        }
        const double r = tdls::detail::abs(acc - static_cast<double>(b[i]));
        if (r > rmax) rmax = r;
        const double xa = tdls::detail::abs(static_cast<double>(x[i]));
        if (xa > xmax) xmax = xa;
        const double ba = tdls::detail::abs(static_cast<double>(b[i]));
        if (ba > bmax) bmax = ba;
    }
    return rmax / (amax * xmax + bmax);
}

/// \brief Fills a reproducible system.
/// \tparam T scalar type
/// \tparam N system dimension
/// \param[in]  seed generator seed
/// \param[out] A    matrix, contiguous row-major
/// \param[out] b    right-hand side
template<typename T, int N>
constexpr void fill_system(const unsigned seed, T* A, T* b) {
    unsigned s = seed;
    for (int e = 0; e < N * N; ++e)
        A[e] = static_cast<T>(lcg(s));
    for (int i = 0; i < N; ++i)
        b[i] = static_cast<T>(lcg(s));
}

/// \brief Certificate: solve_inplace on internal storage, verdict on the
/// backward error.
/// \tparam T      scalar type
/// \tparam N      system dimension
/// \tparam Config solver configuration (one thread per system)
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<typename T, int N, tdls::CooperativeLUppConfig<T> Config = sequential_config<T, N>>
constexpr bool solve_inplace_internal_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::CooperativeLUppSolverStatic<T, N, Config>;
    static_assert(Solver::threads_per_system == 1);
    T A[N * N]                     = {};
    T A0[N * N]                    = {};
    T b[N]                         = {};
    T y[N]                         = {};
    int piv[N]                     = {};
    T work[Solver::workspace_size] = {};
    fill_system<T, N>(seed, A0, b);
    for (int e = 0; e < N * N; ++e)
        A[e] = A0[e];
    for (int i = 0; i < N; ++i)
        y[i] = b[i];
    if (!Solver::template solve_inplace<true, true, true>(0, A, 1, piv, 1, y, 1, work))
        return false;
    return backward_error<T, N>(A0, y, b) <= tolerance;
}

/// \brief Certificate: solve with every operand external, mapped on a
/// stride-2 arena.
/// \tparam N system dimension
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<int N>
constexpr bool solve_external_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::CooperativeLUppSolverStatic<double, N, sequential_config<double, N>>;
    double A[2 * N * N]                 = {};
    double A0[N * N]                    = {};
    double b[2 * N]                     = {};
    double bc[N]                        = {};
    double x[2 * N]                     = {};
    int piv[2 * N]                      = {};
    double work[Solver::workspace_size] = {};
    fill_system<double, N>(seed, A0, bc);
    for (int e = 0; e < N * N; ++e)
        A[2 * e] = A0[e];
    for (int i = 0; i < N; ++i)
        b[2 * i] = bc[i];
    if (!Solver::template solve<false, false, false>(0, A, 2, piv, 2, b, x, 2, work)) return false;
    double xc[N] = {};
    for (int i = 0; i < N; ++i)
        xc[i] = x[2 * i];
    return backward_error<double, N>(A0, xc, bc) <= tolerance;
}

/// \brief Certificate: solve, solve_inplace, factorize + substitute and
/// factorize + substitute_inplace agree exactly on the factored rows, the
/// row positions and the solution.
/// \tparam N system dimension
/// \param[in] seed generator seed
/// \return true when every path agrees
template<int N>
constexpr bool entry_points_certificate(const unsigned seed) {
    using Solver     = tdls::CooperativeLUppSolverStatic<double, N, sequential_config<double, N>>;
    double A0[N * N] = {};
    double b[N]      = {};
    fill_system<double, N>(seed, A0, b);

    double work[Solver::workspace_size] = {};
    double A_s[N * N] = {}, A_f[N * N] = {}, A_p[N * N] = {};
    double x_s[N] = {}, y_f[N] = {}, x_p[N] = {}, x_i[N] = {};
    int piv_s[N] = {}, piv_f[N] = {}, piv_p[N] = {};
    for (int e = 0; e < N * N; ++e) {
        A_s[e] = A0[e];
        A_f[e] = A0[e];
        A_p[e] = A0[e];
    }
    for (int i = 0; i < N; ++i) {
        y_f[i] = b[i];
        x_i[i] = b[i];
    }
    if (!Solver::template solve<true, true, true>(0, A_s, 1, piv_s, 1, b, x_s, 1, work))
        return false;
    if (!Solver::template solve_inplace<true, true, true>(0, A_f, 1, piv_f, 1, y_f, 1, work))
        return false;
    if (!Solver::template factorize<true, true>(0, A_p, 1, piv_p, 1, work)) return false;
    Solver::template substitute<true, true, true>(0, A_p, 1, piv_p, 1, b, x_p, 1, work);
    Solver::template substitute_inplace<true, true, true>(0, A_p, 1, piv_p, 1, x_i, 1, work);
    for (int e = 0; e < N * N; ++e)
        if (A_s[e] != A_f[e] || A_s[e] != A_p[e]) return false;
    for (int i = 0; i < N; ++i) {
        if (piv_s[i] != piv_f[i] || piv_s[i] != piv_p[i]) return false;
        if (x_s[i] != y_f[i] || x_s[i] != x_p[i] || x_s[i] != x_i[i]) return false;
    }
    return true;
}

/// \brief Certificate: substitute_canonical(col) reproduces substitute()
/// on e_col for every column, and the column solves A x = e_col.
/// \tparam N system dimension
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when every column agrees within the tolerance
template<int N>
constexpr bool canonical_certificate(const unsigned seed, const double tolerance) {
    using Solver     = tdls::CooperativeLUppSolverStatic<double, N, sequential_config<double, N>>;
    double A0[N * N] = {};
    double b[N]      = {};
    double A[N * N]  = {};
    int piv[N]       = {};
    double work[Solver::workspace_size] = {};
    fill_system<double, N>(seed, A0, b);
    for (int e = 0; e < N * N; ++e)
        A[e] = A0[e];
    if (!Solver::template factorize<true, true>(0, A, 1, piv, 1, work)) return false;
    for (int col = 0; col < N; ++col) {
        double e[N]  = {};
        double x1[N] = {};
        double x2[N] = {};
        e[col]       = 1.0;
        Solver::template substitute<true, true, true>(0, A, 1, piv, 1, e, x1, 1, work);
        Solver::template substitute_canonical<true, true, true>(0, A, 1, piv, 1, col, x2, 1, work);
        for (int i = 0; i < N; ++i)
            if (x1[i] != x2[i]) return false;
        if (backward_error<double, N>(A0, x2, e) > tolerance) return false;
    }
    return true;
}

/// \brief Certificate: the column-major layout on transposed storage, and
/// a rows_per_thread above N, reproduce the row-major sequential solve
/// exactly.
/// \tparam N system dimension
/// \param[in] seed generator seed
/// \return true when the three solves agree
template<int N>
constexpr bool layout_and_clamp_certificate(const unsigned seed) {
    using Row = tdls::CooperativeLUppSolverStatic<double, N, sequential_config<double, N>>;
    using Col = tdls::CooperativeLUppSolverStatic<
        double, N, sequential_config<double, N, true, tdls::MatrixLayout::ColMajor>>;
    using Clamped = tdls::CooperativeLUppSolverStatic<
        double, N, tdls::CooperativeLUppConfig<double>{.rows_per_thread = N + 3}>;
    static_assert(Clamped::rows_per_thread == N && Clamped::threads_per_system == 1);
    double A0[N * N] = {};
    double b[N]      = {};
    fill_system<double, N>(seed, A0, b);

    double work[3 * N] = {};
    double A_r[N * N] = {}, A_c[N * N] = {}, A_k[N * N] = {};
    double y_r[N] = {}, y_c[N] = {}, y_k[N] = {};
    int piv_r[N] = {}, piv_c[N] = {}, piv_k[N] = {};
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            A_r[r * N + c] = A0[r * N + c];
            A_c[c * N + r] = A0[r * N + c];
            A_k[r * N + c] = A0[r * N + c];
        }
        y_r[r] = b[r];
        y_c[r] = b[r];
        y_k[r] = b[r];
    }
    if (!Row::template solve_inplace<false, false, false>(0, A_r, 1, piv_r, 1, y_r, 1, work))
        return false;
    if (!Col::template solve_inplace<false, false, false>(0, A_c, 1, piv_c, 1, y_c, 1, work))
        return false;
    if (!Clamped::template solve_inplace<false, false, false>(0, A_k, 1, piv_k, 1, y_k, 1, work))
        return false;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c)
            if (A_r[r * N + c] != A_c[c * N + r] || A_r[r * N + c] != A_k[r * N + c]) return false;
        if (piv_r[r] != piv_c[r] || piv_r[r] != piv_k[r]) return false;
        if (y_r[r] != y_c[r] || y_r[r] != y_k[r]) return false;
    }
    return true;
}

/// \brief Certificate: a zero column makes factorize return false. The
/// factorization alone divides by no zero pivot, unlike the substitution
/// of a singular matrix, so the singular verdict is constant-evaluable.
/// \tparam N system dimension
/// \return true when the singular matrix is rejected
template<int N>
constexpr bool singular_rejected_certificate() {
    using Solver    = tdls::CooperativeLUppSolverStatic<double, N, sequential_config<double, N>>;
    double A[N * N] = {};
    double b[N]     = {};
    int piv[N]      = {};
    double work[Solver::workspace_size] = {};
    fill_system<double, N>(7, A, b);
    for (int r = 0; r < N; ++r)
        A[r * N + 1] = 0.0;
    return !Solver::template factorize<true, true>(0, A, 1, piv, 1, work);
}

/// \brief Certificate: one solve_inplace of the runtime solver, verdict
/// on the backward error.
/// \tparam N               system dimension, passed at run time to the
///         solver and sizing the local arrays
/// \tparam rows_per_thread rows per thread of the configuration (at
///         least N: one thread per system)
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<int N, int rows_per_thread>
constexpr bool dynamic_solve_certificate(const unsigned seed, const double tolerance) {
    using Solver =
        tdls::CooperativeLUppSolverDynamic<double, tdls::CooperativeLUppConfig<double>{
                                                       .rows_per_thread = rows_per_thread}>;
    static_assert(rows_per_thread >= N);
    double A[N * N]    = {};
    double A0[N * N]   = {};
    double b[N]        = {};
    double y[N]        = {};
    int piv[N]         = {};
    double work[3 * N] = {};
    fill_system<double, N>(seed, A0, b);
    for (int e = 0; e < N * N; ++e)
        A[e] = A0[e];
    for (int i = 0; i < N; ++i)
        y[i] = b[i];
    if (Solver::threads_per_system(N) != 1 || Solver::workspace_size(N) != 3 * N) return false;
    if (!Solver::solve_inplace(N, 0, A, 1, piv, 1, y, 1, work)) return false;
    return backward_error<double, N>(A0, y, b) <= tolerance;
}

/// \brief Certificate: the runtime solver reproduces the compile-time one
/// exactly, through solve_inplace, solve, factorize + substitute and
/// substitute_canonical, on a stride-2 arena.
/// \tparam N system dimension
/// \param[in] seed generator seed
/// \return true when every output agrees
template<int N>
constexpr bool bridge_certificate(const unsigned seed) {
    using Static     = tdls::CooperativeLUppSolverStatic<double, N, sequential_config<double, N>>;
    using Dynamic    = tdls::CooperativeLUppSolverDynamic<double, sequential_config<double, N>>;
    double A0[N * N] = {};
    double b0[N]     = {};
    fill_system<double, N>(seed, A0, b0);

    double work[3 * N]    = {};
    double A_s[2 * N * N] = {}, A_d[2 * N * N] = {};
    double b[2 * N] = {}, y_s[2 * N] = {}, y_d[2 * N] = {}, x_s[2 * N] = {}, x_d[2 * N] = {};
    int piv_s[2 * N] = {}, piv_d[2 * N] = {};
    const auto reset = [&] {
        for (int e = 0; e < N * N; ++e) {
            A_s[2 * e] = A0[e];
            A_d[2 * e] = A0[e];
        }
        for (int i = 0; i < N; ++i) {
            b[2 * i]   = b0[i];
            y_s[2 * i] = b0[i];
            y_d[2 * i] = b0[i];
        }
    };
    const auto agree = [&] {
        for (int e = 0; e < 2 * N * N; ++e)
            if (A_s[e] != A_d[e]) return false;
        for (int i = 0; i < 2 * N; ++i)
            if (piv_s[i] != piv_d[i] || y_s[i] != y_d[i] || x_s[i] != x_d[i]) return false;
        return true;
    };

    reset();
    if (!Static::template solve_inplace<false, false, false>(0, A_s, 2, piv_s, 2, y_s, 2, work))
        return false;
    if (!Dynamic::solve_inplace(N, 0, A_d, 2, piv_d, 2, y_d, 2, work)) return false;
    if (!agree()) return false;

    reset();
    if (!Static::template solve<false, false, false>(0, A_s, 2, piv_s, 2, b, x_s, 2, work))
        return false;
    if (!Dynamic::solve(N, 0, A_d, 2, piv_d, 2, b, x_d, 2, work)) return false;
    if (!agree()) return false;

    reset();
    if (!Static::template factorize<false, false>(0, A_s, 2, piv_s, 2, work)) return false;
    if (!Dynamic::factorize(N, 0, A_d, 2, piv_d, 2, work)) return false;
    Static::template substitute<false, false, false>(0, A_s, 2, piv_s, 2, b, x_s, 2, work);
    Dynamic::substitute(N, 0, A_d, 2, piv_d, 2, b, x_d, 2, work);
    if (!agree()) return false;
    for (int col = 0; col < N; ++col) {
        Static::template substitute_canonical<false, false, false>(0, A_s, 2, piv_s, 2, col, x_s, 2,
                                                                   work);
        Dynamic::substitute_canonical(N, 0, A_d, 2, piv_d, 2, col, x_d, 2, work);
        if (!agree()) return false;
    }
    return true;
}

/// \brief Certificate: a zero column makes the factorization of the
/// runtime solver return false.
/// \tparam N system dimension
/// \return true when the singular matrix is rejected
template<int N>
constexpr bool dynamic_singular_rejected_certificate() {
    using Solver       = tdls::CooperativeLUppSolverDynamic<double, sequential_config<double, N>>;
    double A[N * N]    = {};
    double b[N]        = {};
    int piv[N]         = {};
    double work[3 * N] = {};
    fill_system<double, N>(7, A, b);
    for (int r = 0; r < N; ++r)
        A[r * N + 1] = 0.0;
    return !Solver::factorize(N, 0, A, 1, piv, 1, work);
}

// Internal storage: the scalar corner, even and odd dimensions.
static_assert(solve_inplace_internal_certificate<double, 1>(201, 1e-9));
static_assert(solve_inplace_internal_certificate<double, 4>(202, 1e-9));
static_assert(solve_inplace_internal_certificate<double, 5>(203, 1e-9));
static_assert(solve_inplace_internal_certificate<double, 6>(204, 1e-9));
// Float cell.
static_assert(solve_inplace_internal_certificate<float, 5>(205, 1e-5));
// The no-pragma loop branches (unroll_loops = false).
static_assert(
    solve_inplace_internal_certificate<double, 5, sequential_config<double, 5, false>>(206, 1e-9));
// External residency on a strided arena.
static_assert(solve_external_certificate<5>(207, 1e-9));
// Entry-point equivalences and the tangent-operator path.
static_assert(entry_points_certificate<5>(208));
static_assert(canonical_certificate<4>(209, 1e-9));
// Column-major layout and rows_per_thread above N.
static_assert(layout_and_clamp_certificate<5>(210));
// Singular verdict of the factorization.
static_assert(singular_rejected_certificate<4>());
// The runtime solver: the scalar corner, odd and even dimensions, and
// rows_per_thread above n (phantom slots).
static_assert(dynamic_solve_certificate<1, 1>(220, 1e-9));
static_assert(dynamic_solve_certificate<5, 5>(221, 1e-9));
static_assert(dynamic_solve_certificate<6, 6>(222, 1e-9));
static_assert(dynamic_solve_certificate<4, 7>(223, 1e-9));
static_assert(dynamic_singular_rejected_certificate<4>());
// The static/dynamic bitwise bridge, at compile time.
static_assert(bridge_certificate<5>(224));
static_assert(bridge_certificate<4>(225));

} // namespace

TDLS_TEST_CASE("cooperativelupp/constexpr/certificates-also-hold-at-run-time") {
    TDLS_CHECK((solve_inplace_internal_certificate<double, 1>(201, 1e-9)));
    TDLS_CHECK((solve_inplace_internal_certificate<double, 4>(202, 1e-9)));
    TDLS_CHECK((solve_inplace_internal_certificate<double, 5>(203, 1e-9)));
    TDLS_CHECK((solve_inplace_internal_certificate<double, 6>(204, 1e-9)));
    TDLS_CHECK((solve_inplace_internal_certificate<float, 5>(205, 1e-5)));
    TDLS_CHECK((solve_inplace_internal_certificate<double, 5, sequential_config<double, 5, false>>(
        206, 1e-9)));
    TDLS_CHECK((solve_external_certificate<5>(207, 1e-9)));
    TDLS_CHECK((entry_points_certificate<5>(208)));
    TDLS_CHECK((canonical_certificate<4>(209, 1e-9)));
    TDLS_CHECK((layout_and_clamp_certificate<5>(210)));
    TDLS_CHECK((singular_rejected_certificate<4>()));
    TDLS_CHECK((dynamic_solve_certificate<1, 1>(220, 1e-9)));
    TDLS_CHECK((dynamic_solve_certificate<5, 5>(221, 1e-9)));
    TDLS_CHECK((dynamic_solve_certificate<6, 6>(222, 1e-9)));
    TDLS_CHECK((dynamic_solve_certificate<4, 7>(223, 1e-9)));
    TDLS_CHECK((dynamic_singular_rejected_certificate<4>()));
    TDLS_CHECK((bridge_certificate<5>(224)));
    TDLS_CHECK((bridge_certificate<4>(225)));
}

TDLS_TEST_MAIN
