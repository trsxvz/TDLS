/// \file
/// \brief Compile-time certificate suite of the TiledLUpp solvers.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. Released under the
/// BSD 3-Clause License (see the LICENSE file).
///
/// Every static_assert below constant-evaluates a whole solve, and the
/// standard requires constant evaluation to reject undefined behaviour:
/// an out-of-bounds index, a read of an uninitialized value or a signed
/// overflow fails the compilation. Each assertion that compiles is
/// therefore a machine-checked certificate that the exercised path is
/// free of undefined behaviour. The grid crosses the structural
/// boundaries of the solver: divisible grids, trailing tiles, TS = N,
/// TS > N (single partial tile), the 1 x 1 corner, both schedules,
/// internal and external residencies, the out-of-tile recovery, the
/// entry-point equivalences and the singular verdict. It covers both the
/// compile-time and the runtime TiledLUpp solvers, plus the
/// static/dynamic bitwise bridge, itself evaluated at compile time.
/// The certificates are also re-run at run time, so the suite reports
/// like any other. The default configuration certifies the
/// unroll_inner branches; one no-unroll cell certifies the others.

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

/// \brief Configuration flipping only unroll_inner, so the no-pragma
/// branches of the two-branch unroll trick get their certificate too.
/// \tparam T  scalar type
/// \tparam TS tile size
template<typename T, int TS>
struct NoUnrollConfig : tdls::TiledLUppConfig<T, TS> {
    static constexpr bool unroll_inner = false; ///< no pragma anywhere
};

/// \brief Normwise backward error |A0 x - b| / (|A0| |x| + |b|),
/// accumulated in double and computable during constant evaluation.
/// \tparam T scalar type
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

/// \brief Certificate: one solve through the no-pragma loop branches.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<typename T, int N, int TS>
constexpr bool no_unroll_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, NoUnrollConfig<T, TS>>;
    T A[N * N]   = {};
    T A0[N * N]  = {};
    T b[N]       = {};
    T x[N]       = {};
    int piv[N]   = {};
    unsigned s   = seed;
    for (int e = 0; e < N * N; ++e) {
        A[e]  = static_cast<T>(lcg(s));
        A0[e] = A[e];
    }
    for (int i = 0; i < N; ++i)
        b[i] = static_cast<T>(lcg(s));
    if (!Solver::template solve<true, true, true>(A, 1, piv, 1, b, x, 1)) return false;
    return backward_error<T, N>(A0, x, b) <= tolerance;
}

/// \brief Certificate: one solve on internal (caller-local) storage,
/// verdict on the backward error.
/// \tparam T        scalar type
/// \tparam N        system dimension
/// \tparam TS       tile size
/// \tparam Schedule elimination schedule
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule>
constexpr bool solve_internal_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Schedule>>;
    T A[N * N]   = {};
    T A0[N * N]  = {};
    T b[N]       = {};
    T x[N]       = {};
    int piv[N]   = {};
    unsigned s   = seed;
    for (int e = 0; e < N * N; ++e) {
        A[e]  = static_cast<T>(lcg(s));
        A0[e] = A[e];
    }
    for (int i = 0; i < N; ++i)
        b[i] = static_cast<T>(lcg(s));
    if (!Solver::template solve<true, true, true>(A, 1, piv, 1, b, x, 1)) return false;
    return backward_error<T, N>(A0, x, b) <= tolerance;
}

/// \brief Certificate: one solve with every operand external, mapped on
/// a stride-2 arena.
/// \tparam T        scalar type
/// \tparam N        system dimension
/// \tparam TS       tile size
/// \tparam Schedule elimination schedule
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule>
constexpr bool solve_external_certificate(const unsigned seed, const double tolerance) {
    using Solver   = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Schedule>>;
    T A[2 * N * N] = {};
    T A0[N * N]    = {};
    T b[2 * N]     = {};
    T x[2 * N]     = {};
    int piv[2 * N] = {};
    unsigned s     = seed;
    for (int e = 0; e < N * N; ++e) {
        const T v = static_cast<T>(lcg(s));
        A[2 * e]  = v;
        A0[e]     = v;
    }
    for (int i = 0; i < N; ++i)
        b[2 * i] = static_cast<T>(lcg(s));
    if (!Solver::template solve<false, false, false>(A, 2, piv, 2, b, x, 2)) return false;
    T xc[N] = {};
    T bc[N] = {};
    for (int i = 0; i < N; ++i) {
        xc[i] = x[2 * i];
        bc[i] = b[2 * i];
    }
    return backward_error<T, N>(A0, xc, bc) <= tolerance;
}

/// \brief Configuration whose threshold no entry can reach: the
/// out-of-tile recovery fires on every full-tile column.
/// \tparam T  scalar type
/// \tparam TS tile size
template<typename T, int TS>
struct AlwaysOotConfig : tdls::TiledLUppConfig<T, TS> {
    static constexpr T oot_threshold = T(1e30); ///< unreachable threshold
};

/// \brief Certificate: the out-of-tile recovery path (candidate replay
/// included) is exercised on every column.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<typename T, int N, int TS>
constexpr bool oot_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, AlwaysOotConfig<T, TS>>;
    T A[N * N]   = {};
    T A0[N * N]  = {};
    T b[N]       = {};
    T x[N]       = {};
    int piv[N]   = {};
    unsigned s   = seed;
    for (int e = 0; e < N * N; ++e) {
        A[e]  = static_cast<T>(lcg(s));
        A0[e] = A[e];
    }
    for (int i = 0; i < N; ++i)
        b[i] = static_cast<T>(lcg(s));
    if (!Solver::template solve<true, true, true>(A, 1, piv, 1, b, x, 1)) return false;
    return backward_error<T, N>(A0, x, b) <= tolerance;
}

/// \brief Certificate: solve_inplace reproduces solve bitwise, factored
/// matrix and pivots included.
/// \tparam T        scalar type
/// \tparam N        system dimension
/// \tparam TS       tile size
/// \tparam Schedule elimination schedule
/// \param[in] seed generator seed
/// \return true when every output matches exactly
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule>
constexpr bool fused_matches_solve_certificate(const unsigned seed) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Schedule>>;
    T As[N * N]  = {};
    T Af[N * N]  = {};
    T b[N]       = {};
    T x[N]       = {};
    T y[N]       = {};
    int pivs[N]  = {};
    int pivf[N]  = {};
    unsigned s   = seed;
    for (int e = 0; e < N * N; ++e) {
        As[e] = static_cast<T>(lcg(s));
        Af[e] = As[e];
    }
    for (int i = 0; i < N; ++i) {
        b[i] = static_cast<T>(lcg(s));
        y[i] = b[i];
    }
    const bool oks = Solver::template solve<true, true, true>(As, 1, pivs, 1, b, x, 1);
    const bool okf = Solver::template solve_inplace<true, true, true>(Af, 1, pivf, 1, y, 1);
    if (!oks || !okf) return false;
    for (int e = 0; e < N * N; ++e)
        if (As[e] != Af[e]) return false;
    for (int i = 0; i < N; ++i)
        if (x[i] != y[i] || pivs[i] != pivf[i]) return false;
    return true;
}

/// \brief Certificate: the canonical columns solve A x = e_col from one
/// factorization: the tangent-operator path.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when every column stayed within the tolerance
template<typename T, int N, int TS>
constexpr bool canonical_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS>>;
    T A[N * N]   = {};
    T A0[N * N]  = {};
    T x[N]       = {};
    T e[N]       = {};
    int piv[N]   = {};
    unsigned s   = seed;
    for (int el = 0; el < N * N; ++el) {
        A[el]  = static_cast<T>(lcg(s));
        A0[el] = A[el];
    }
    if (!Solver::template factorize<true, true>(A, 1, piv, 1)) return false;
    for (int col = 0; col < N; ++col) {
        Solver::template substitute_canonical<true, true, true>(A, 1, piv, 1, col, x, 1);
        for (int i = 0; i < N; ++i)
            e[i] = (i == col) ? T(1) : T(0);
        if (backward_error<T, N>(A0, x, e) > tolerance) return false;
    }
    return true;
}

/// \brief Certificate: the runtime TiledLUpp solver solves at a dimension
/// only known through a function parameter (a constant here, so the
/// whole call constant-evaluates).
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] seed      generator seed
/// \param[in] tolerance backward-error bound
/// \return true when the solve succeeded within the tolerance
template<typename T, int N, int TS>
constexpr bool dynamic_solve_certificate(const unsigned seed, const double tolerance) {
    using Solver = tdls::TiledLUppSolverDynamic<T, tdls::TiledLUppConfig<T, TS>>;
    T A[N * N]   = {};
    T A0[N * N]  = {};
    T b[N]       = {};
    T x[N]       = {};
    int piv[N]   = {};
    unsigned s   = seed;
    for (int e = 0; e < N * N; ++e) {
        A[e]  = static_cast<T>(lcg(s));
        A0[e] = A[e];
    }
    for (int i = 0; i < N; ++i)
        b[i] = static_cast<T>(lcg(s));
    if (!Solver::solve(N, A, 1, piv, 1, b, x, 1)) return false;
    return backward_error<T, N>(A0, x, b) <= tolerance;
}

/// \brief Certificate: the static and the dynamic TiledLUpp solvers produce
/// bitwise-identical outputs at equal shape: the bridge invariant,
/// checked here during constant evaluation.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] seed generator seed
/// \return true when every output matches exactly
template<typename T, int N, int TS>
constexpr bool bridge_certificate(const unsigned seed) {
    using Config  = tdls::TiledLUppConfig<T, TS>;
    using Static  = tdls::TiledLUppSolverStatic<T, N, Config>;
    using Dynamic = tdls::TiledLUppSolverDynamic<T, Config>;
    T As[N * N]   = {};
    T Ad[N * N]   = {};
    T b[N]        = {};
    T xs[N]       = {};
    T xd[N]       = {};
    int ps[N]     = {};
    int pd[N]     = {};
    unsigned s    = seed;
    for (int e = 0; e < N * N; ++e) {
        As[e] = static_cast<T>(lcg(s));
        Ad[e] = As[e];
    }
    for (int i = 0; i < N; ++i)
        b[i] = static_cast<T>(lcg(s));
    const bool oks = Static::template solve<true, true, true>(As, 1, ps, 1, b, xs, 1);
    const bool okd = Dynamic::solve(N, Ad, 1, pd, 1, b, xd, 1);
    if (!oks || !okd) return false;
    for (int e = 0; e < N * N; ++e)
        if (As[e] != Ad[e]) return false;
    for (int i = 0; i < N; ++i)
        if (xs[i] != xd[i] || ps[i] != pd[i]) return false;
    return true;
}

/// \brief Certificate: the multi right-hand-side blocks hold their
/// documented equivalences during constant evaluation, whatever the
/// pass_width cutting: substitute_multirhs
/// matches W separate substitutes bitwise, substitute_inplace_multirhs
/// matches the two-buffer block, and the runtime solver's blocks match
/// the compile-time solver's.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \tparam W  number of right-hand-side columns per block
/// \param[in] seed generator seed
/// \return true when every output matches exactly
template<typename T, int N, int TS, int W>
constexpr bool multirhs_certificate(const unsigned seed) {
    using Config  = tdls::TiledLUppConfig<T, TS>;
    using Static  = tdls::TiledLUppSolverStatic<T, N, Config>;
    using Dynamic = tdls::TiledLUppSolverDynamic<T, Config>;
    T A[N * N];
    T B[W * N];
    T X_ref[W * N];
    T X[W * N];
    int piv[N];
    unsigned s = seed;
    for (int e = 0; e < N * N; ++e)
        A[e] = static_cast<T>(lcg(s));
    for (int e = 0; e < W * N; ++e)
        B[e] = static_cast<T>(lcg(s));
    if (!Static::template factorize<true, true>(A, 1, piv, 1)) return false;
    for (int w = 0; w < W; ++w)
        Static::template substitute<true, true, true>(A, 1, piv, 1, B + w * N, X_ref + w * N, 1);
    Static::template substitute_multirhs<W, true, true, true>(A, 1, piv, 1, B, X, 1, 0);
    for (int e = 0; e < W * N; ++e)
        if (X[e] != X_ref[e]) return false;
    for (int e = 0; e < W * N; ++e)
        X[e] = B[e];
    Static::template substitute_inplace_multirhs<W, true, true, true>(A, 1, piv, 1, X, 1, 0);
    for (int e = 0; e < W * N; ++e)
        if (X[e] != X_ref[e]) return false;
    Dynamic::substitute_multirhs(N, W, A, 1, piv, 1, B, X, 1, N);
    for (int e = 0; e < W * N; ++e)
        if (X[e] != X_ref[e]) return false;
    for (int e = 0; e < W * N; ++e)
        X[e] = B[e];
    Dynamic::substitute_inplace_multirhs(N, W, A, 1, piv, 1, X, 1, N);
    for (int e = 0; e < W * N; ++e)
        if (X[e] != X_ref[e]) return false;
    // The pass cutting must not change a single bit, on either solver.
    Static::template substitute_multirhs<W, true, true, true, 2>(A, 1, piv, 1, B, X, 1, 0);
    for (int e = 0; e < W * N; ++e)
        if (X[e] != X_ref[e]) return false;
    Dynamic::substitute_multirhs(N, W, A, 1, piv, 1, B, X, 1, N, 2);
    for (int e = 0; e < W * N; ++e)
        if (X[e] != X_ref[e]) return false;
    return true;
}

/// \brief Certificate: a structurally singular matrix (all zeros) is
/// rejected, with no division ever reached.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \return true when the solver reports the singularity
template<typename T, int N, int TS>
constexpr bool singular_rejected_certificate() {
    using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS>>;
    T A[N * N]   = {};
    T b[N]       = {};
    T x[N]       = {};
    int piv[N]   = {};
    return !Solver::template solve<true, true, true>(A, 1, piv, 1, b, x, 1);
}

constexpr auto RightLooking = tdls::TiledLUppSchedule::RightLooking;
constexpr auto LeftLooking  = tdls::TiledLUppSchedule::LeftLooking;

// Divisible grids, both schedules.
static_assert(solve_internal_certificate<double, 4, 2, RightLooking>(101, 1e-9));
static_assert(solve_internal_certificate<double, 4, 2, LeftLooking>(102, 1e-9));
static_assert(solve_internal_certificate<double, 6, 3, RightLooking>(103, 1e-9));
// Trailing tiles.
static_assert(solve_internal_certificate<double, 5, 3, RightLooking>(104, 1e-9));
static_assert(solve_internal_certificate<double, 5, 3, LeftLooking>(105, 1e-9));
// TS = N, TS > N (single partial tile) and the scalar corner.
static_assert(solve_internal_certificate<double, 4, 4, RightLooking>(106, 1e-9));
static_assert(solve_internal_certificate<double, 3, 8, RightLooking>(107, 1e-9));
static_assert(solve_internal_certificate<double, 1, 1, RightLooking>(108, 1e-9));
// Float cell (tolerance calibrated as in the oracle suites).
static_assert(solve_internal_certificate<float, 4, 2, RightLooking>(109, 1e-2));
// External residency on a strided arena.
static_assert(solve_external_certificate<double, 5, 3, RightLooking>(110, 1e-9));
// Out-of-tile recovery on every full-tile column.
static_assert(oot_certificate<double, 6, 3>(111, 1e-9));
// Entry-point equivalence and the tangent-operator path.
static_assert(fused_matches_solve_certificate<double, 5, 3, RightLooking>(112));
static_assert(canonical_certificate<double, 4, 2>(113, 1e-9));
// The no-pragma loop branches (unroll_inner = false).
static_assert(no_unroll_certificate<double, 5, 3>(114, 1e-9));
// Singular verdict.
static_assert(singular_rejected_certificate<double, 4, 2>());
// The runtime TiledLUpp solver: divisible, trailing, TS > n, scalar corner.
static_assert(dynamic_solve_certificate<double, 6, 3>(120, 1e-9));
static_assert(dynamic_solve_certificate<double, 5, 3>(121, 1e-9));
static_assert(dynamic_solve_certificate<double, 3, 8>(122, 1e-9));
static_assert(dynamic_solve_certificate<double, 1, 1>(123, 1e-9));
// The static/dynamic bitwise bridge, at compile time.
static_assert(bridge_certificate<double, 5, 3>(124));
static_assert(bridge_certificate<double, 4, 2>(125));
// The multi right-hand-side blocks, on both solvers.
static_assert(multirhs_certificate<double, 5, 3, 2>(130));
static_assert(multirhs_certificate<double, 4, 2, 4>(131));

} // namespace

TDLS_TEST_CASE("tiledlupp/constexpr/certificates-also-hold-at-run-time") {
    TDLS_CHECK((solve_internal_certificate<double, 4, 2, RightLooking>(101, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 4, 2, LeftLooking>(102, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 6, 3, RightLooking>(103, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 5, 3, RightLooking>(104, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 5, 3, LeftLooking>(105, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 4, 4, RightLooking>(106, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 3, 8, RightLooking>(107, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<double, 1, 1, RightLooking>(108, 1e-9)));
    TDLS_CHECK((solve_internal_certificate<float, 4, 2, RightLooking>(109, 1e-2)));
    TDLS_CHECK((solve_external_certificate<double, 5, 3, RightLooking>(110, 1e-9)));
    TDLS_CHECK((oot_certificate<double, 6, 3>(111, 1e-9)));
    TDLS_CHECK((fused_matches_solve_certificate<double, 5, 3, RightLooking>(112)));
    TDLS_CHECK((canonical_certificate<double, 4, 2>(113, 1e-9)));
    TDLS_CHECK((no_unroll_certificate<double, 5, 3>(114, 1e-9)));
    TDLS_CHECK((singular_rejected_certificate<double, 4, 2>()));
    TDLS_CHECK((dynamic_solve_certificate<double, 6, 3>(120, 1e-9)));
    TDLS_CHECK((dynamic_solve_certificate<double, 5, 3>(121, 1e-9)));
    TDLS_CHECK((dynamic_solve_certificate<double, 3, 8>(122, 1e-9)));
    TDLS_CHECK((dynamic_solve_certificate<double, 1, 1>(123, 1e-9)));
    TDLS_CHECK((bridge_certificate<double, 5, 3>(124)));
    TDLS_CHECK((bridge_certificate<double, 4, 2>(125)));
    TDLS_CHECK((multirhs_certificate<double, 5, 3, 2>(130)));
    TDLS_CHECK((multirhs_certificate<double, 4, 2, 4>(131)));
}

TDLS_TEST_MAIN
