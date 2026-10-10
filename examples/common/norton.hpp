/// \file
/// \brief Norton viscoplasticity integrated the MFront way: the physics
/// of the norton_law examples.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Small strain, isotropic elasticity, Norton creep:
///
///   eto = eel + evp,   sig = D : eel,   devp/dt = dp/dt n,
///   dp/dt = A (seq / s0)^m,   n = 3/2 dev(sig) / seq
///
/// with seq the von Mises stress. A time step is integrated by the
/// implicit Euler scheme, as the MFront `Implicit` DSL does: the
/// unknowns are the increments of the elastic strain (6 components) and
/// of the viscoplastic multiplier p, the residual is
///
///   f_eel = deel + dp n - deto,   f_p = dp - dt A (seq / s0)^m
///
/// and Newton iterations solve J c = f with the analytic jacobian of the
/// MFront tutorial. Every iteration is one solve_inplace on a fresh 7 x
/// 7 jacobian. At convergence the consistent tangent operator
/// dsig/deto = D . (dDeel/dDeto) needs the six columns of J^-1 that
/// belong to the elastic strain: one factorize and one
/// substitute_canonical_multirhs, where MFront runs six substitutions
/// on unit vectors.
///
/// Symmetric tensors are stored as 6-vectors in the TFEL convention
/// (xx, yy, zz, sqrt(2) xy, sqrt(2) xz, sqrt(2) yz), so that the dot
/// product of two vectors is the double contraction of the tensors and
/// the isotropic stiffness D is lambda Id (x) Id + 2 mu I. No tensor
/// library is needed: everything the law manipulates is dense, which
/// is exactly what the solver expects.

#ifndef TDLS_EXAMPLES_NORTON_HPP
#define TDLS_EXAMPLES_NORTON_HPP

#include <cmath>
#include <cstddef>
#include <limits>

#include <tdls/tdls.hpp>

#include "hash01.hpp"

namespace norton {

constexpr int stensor_size = 6; ///< components of a symmetric tensor in 3D
constexpr int N            = 7; ///< unknowns: the elastic strain increment and dp

/// \brief Configuration of the solvers of the CooperativeLUpp examples.
///
/// Built by a function rather than by designated initializers in the
/// alias below: MSVC does not evaluate designated initializers that
/// depend on the parameters of an alias template.
/// \param[in] rows_per_thread rows held by each thread of the group
/// \param[in] unroll          unroll policy: true on GPU, false on CPU
/// \return the configuration
constexpr tdls::CooperativeLUppConfig<double> group_config(const int rows_per_thread,
                                                           const bool unroll) {
    tdls::CooperativeLUppConfig<double> config;
    config.rows_per_thread = rows_per_thread;
    config.unroll_loops    = unroll;
    return config;
}

/// \brief LU solver of the 7 x 7 Newton systems: N is fixed by the
/// modelling hypothesis (3D), not by the data.
using Solver =
    tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double>{.tile_size = 3}>;

// Material of the Norton behaviour of tfelGPU, in SI units. The creep
// rate is written A (seq / s0)^m, so that A is a rate and the power
// applies to a dimensionless ratio.
constexpr double young  = 150e9;                                  ///< Young modulus (Pa)
constexpr double nu     = 0.3;                                    ///< Poisson ratio
constexpr double A      = 1e-4;                                   ///< Norton coefficient (1/s)
constexpr double s0     = 50e6;                                   ///< Norton reference stress (Pa)
constexpr double m      = 8.2;                                    ///< Norton exponent
constexpr double lambda = young * nu / ((1 + nu) * (1 - 2 * nu)); ///< first Lame coefficient
constexpr double mu     = young / (2 * (1 + nu));                 ///< shear modulus

constexpr double newton_epsilon = 1e-14; ///< convergence criterion on |f| / N, as in MFront
constexpr int newton_max_iter   = 30;    ///< Newton iterations before giving up

/// \brief Strain increment imposed on an integration point at every
/// time step: a tension along x with the Poisson contraction, plus a
/// shear, both scaled per point. The amplitudes span the elastic and
/// the creep regimes across a batch.
/// \param[in]  i    point index
/// \param[out] deto strain increment of the step
TDLS_HOST_DEVICE inline void strain_increment(const unsigned long long i, double* deto) {
    const double a = 1e-4 * (0.5 + 1.5 * hash01(i));
    const double g = 0.4 * hash01(i + 1);
    deto[0]        = a;
    deto[1]        = -nu * a;
    deto[2]        = -nu * a;
    deto[3]        = a * g;
    deto[4]        = 0;
    deto[5]        = 0;
}

/// \brief Trace of a symmetric tensor.
/// \param[in] s tensor
/// \return the trace
TDLS_HOST_DEVICE inline double trace(const double* s) {
    return s[0] + s[1] + s[2];
}

/// \brief Von Mises stress, sqrt(3/2 dev(s) : dev(s)).
/// \param[in] s stress tensor
/// \return the equivalent stress
TDLS_HOST_DEVICE inline double von_mises(const double* s) {
    const double t = trace(s) / 3;
    double d2      = 0;
    for (int i = 0; i < 3; ++i)
        d2 += (s[i] - t) * (s[i] - t);
    for (int i = 3; i < stensor_size; ++i)
        d2 += s[i] * s[i];
    return std::sqrt(1.5 * d2);
}

/// \brief Isotropic Hooke law, sig = lambda tr(eel) Id + 2 mu eel.
/// \param[in]  eel elastic strain
/// \param[out] sig stress
TDLS_HOST_DEVICE inline void hooke(const double* eel, double* sig) {
    const double t = lambda * trace(eel);
    for (int i = 0; i < stensor_size; ++i)
        sig[i] = 2 * mu * eel[i] + (i < 3 ? t : 0);
}

/// \brief Flow direction n = 3/2 dev(sig) / seq and the creep function,
/// evaluated at a stress state. The stress is clamped away from zero
/// as in MFront, so that a stress-free point has a zero direction and
/// no NaN.
/// \param[in]  sig  stress
/// \param[out] n    flow direction
/// \param[out] f    creep rate A (seq / s0)^m
/// \param[out] df   derivative of the creep rate with respect to seq
/// \param[out] iseq 1 / seq, clamped
TDLS_HOST_DEVICE inline void flow(const double* sig, double* n, double& f, double& df,
                                  double& iseq) {
    const double seq = von_mises(sig);
    iseq             = 1 / (seq > 1e-12 * young ? seq : 1e-12 * young);
    const double t   = trace(sig) / 3;
    for (int i = 0; i < stensor_size; ++i)
        n[i] = 1.5 * (sig[i] - (i < 3 ? t : 0)) * iseq;
    f  = A * std::pow(seq / s0, m);
    df = m * f * iseq;
}

/// \brief Entry (i, j) of the jacobian of the residual, the one of the
/// MFront tutorial. The deviatoric projector K and the dyad n (x) n are
/// written out in the 6-vector basis.
/// \param[in] i    row
/// \param[in] j    column
/// \param[in] n    flow direction
/// \param[in] dp   increment of the viscoplastic multiplier
/// \param[in] iseq 1 / seq, clamped
/// \param[in] df   derivative of the creep rate with respect to seq
/// \param[in] dt   time step
/// \return the entry
TDLS_HOST_DEVICE inline double jacobian_entry(const int i, const int j, const double* n,
                                              const double dp, const double iseq, const double df,
                                              const double dt) {
    if (i < stensor_size && j < stensor_size) {
        const double K = (i == j ? 1 : 0) - (i < 3 && j < 3 ? 1.0 / 3 : 0);
        return (i == j ? 1 : 0) + 2 * mu * dp * iseq * (1.5 * K - n[i] * n[j]);
    }
    if (i < stensor_size) return n[i];                     // column of dp
    if (j < stensor_size) return -2 * mu * dt * df * n[j]; // row of f_p
    return 1;
}

/// \brief Jacobian of the residual, every entry from jacobian_entry.
/// \tparam internal_matrix residency of J
/// \param[out] J        jacobian storage, N x N elements
/// \param[in]  J_stride element stride of J (external mode)
/// \param[in]  n        flow direction
/// \param[in]  dp       increment of the viscoplastic multiplier
/// \param[in]  iseq     1 / seq, clamped
/// \param[in]  df       derivative of the creep rate with respect to seq
/// \param[in]  dt       time step
template<bool internal_matrix>
TDLS_HOST_DEVICE inline void jacobian(double* J, const int J_stride, const double* n,
                                      const double dp, const double iseq, const double df,
                                      const double dt) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            J[internal_matrix ? i * N + j : (i * N + j) * J_stride] =
                jacobian_entry(i, j, n, dp, iseq, df, dt);
}

/// \brief Integrate one time step at one integration point.
///
/// The solver operands (jacobian, pivot, right-hand side) are the
/// caller's: local arrays under the internal residencies, slices of a
/// batch in remote memory under the external ones. The law itself only
/// ever manipulates local tensors.
/// \tparam internal_rhs    residency of the right-hand side r
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of the jacobian J
/// \param[in]     deto       strain increment of the step
/// \param[in]     dt         time step
/// \param[in,out] eel        elastic strain, advanced on success
/// \param[in,out] p          viscoplastic multiplier, advanced on success
/// \param[out]    sig        stress at the end of the step
/// \param[out]    Dt         consistent tangent operator, 6 x 6 row-major
/// \param[in,out] J          jacobian storage, N x N elements
/// \param[in]     J_stride   element stride of J (external mode)
/// \param[in,out] piv        pivot storage, N elements
/// \param[in]     piv_stride element stride of piv (external mode)
/// \param[in,out] r          right-hand side storage, N elements
/// \param[in]     rhs_stride element stride of r (external mode)
/// \return false when Newton did not converge or a jacobian was singular
template<bool internal_rhs, bool internal_piv, bool internal_matrix>
TDLS_HOST_DEVICE inline bool integrate(const double* deto, const double dt, double* eel, double& p,
                                       double* sig, double* Dt, double* J, const int J_stride,
                                       int* piv, const int piv_stride, double* r,
                                       const int rhs_stride) {
    // Addressing of the right-hand side: the ternary on a template bool
    // folds, exactly as in the solver.
    auto r_at = [=](const int i) -> double& { return r[internal_rhs ? i : i * rhs_stride]; };

    double deel[stensor_size] = {};
    double dp                 = 0;
    double n[stensor_size];
    double f, df, iseq;

    bool converged = false;
    for (int iter = 0; iter < newton_max_iter && !converged; ++iter) {
        // Residual at the current iterate. eel_n is the elastic strain
        // the stress is evaluated at (implicit Euler, theta = 1).
        double eel_n[stensor_size];
        for (int i = 0; i < stensor_size; ++i)
            eel_n[i] = eel[i] + deel[i];
        hooke(eel_n, sig);
        flow(sig, n, f, df, iseq);

        double norm = 0;
        for (int i = 0; i < stensor_size; ++i) {
            r_at(i) = deel[i] + dp * n[i] - deto[i];
            norm += r_at(i) * r_at(i);
        }
        r_at(6) = dp - dt * f;
        norm += r_at(6) * r_at(6);
        converged = std::sqrt(norm) / N < newton_epsilon;
        if (converged) break;

        // Newton correction: the jacobian is fresh at every iteration, so
        // the fused solve_inplace is the natural entry point.
        jacobian<internal_matrix>(J, J_stride, n, dp, iseq, df, dt);
        if (!Solver::solve_inplace<internal_rhs, internal_piv, internal_matrix>(
                J, J_stride, piv, piv_stride, r, rhs_stride))
            return false;
        for (int i = 0; i < stensor_size; ++i)
            deel[i] -= r_at(i);
        dp -= r_at(6);
    }
    if (!converged) return false;

    for (int i = 0; i < stensor_size; ++i)
        eel[i] += deel[i];
    p += dp;

    // Consistent tangent operator. dDeel/dDeto is the upper-left 6 x 6
    // block of J^-1 at the converged state: the six canonical columns
    // solved together on one factorization, every tile loaded once.
    jacobian<internal_matrix>(J, J_stride, n, dp, iseq, df, dt);
    if (!Solver::factorize<internal_piv, internal_matrix>(J, J_stride, piv, piv_stride))
        return false;
    double X[stensor_size * N]; // column w of J^-1 at X + w * N
    Solver::substitute_canonical_multirhs<stensor_size, true, internal_piv, internal_matrix>(
        J, J_stride, piv, piv_stride, 0, X, 1, 0);
    // Dt = D . iJ, with D = lambda Id (x) Id + 2 mu I.
    for (int j = 0; j < stensor_size; ++j) {
        const double t = lambda * (X[j * N + 0] + X[j * N + 1] + X[j * N + 2]);
        for (int i = 0; i < stensor_size; ++i)
            Dt[i * stensor_size + j] = 2 * mu * X[j * N + i] + (i < 3 ? t : 0);
    }
    return true;
}

/// \brief LU solver of the 7 x 7 Newton systems for a group of
/// threads, the solver of the CooperativeLUpp examples.
/// \tparam rows_per_thread rows held by each thread of the group
/// \tparam unroll          unroll policy: true on GPU, false on CPU
template<int rows_per_thread, bool unroll>
using GroupSolver =
    tdls::CooperativeLUppSolverStatic<double, N, group_config(rows_per_thread, unroll)>;

/// \brief Makes a vector of the group visible to every thread: copies
/// its N entries into x, read from v when v is external, through the
/// workspace when v is the slice of the thread. A barrier closes it, so
/// that v and the workspace can be overwritten right after.
/// \tparam internal_rhs residency of v: the slice of the thread or the
///         whole vector
/// \tparam Solver       a GroupSolver
/// \tparam Sync         callable type of the barrier
/// \param[in]     tx     rank of the thread in its group
/// \param[in]     sync   barrier of the group
/// \param[in]     v      the vector: the entries of the thread, or the
///                whole vector
/// \param[in]     stride element stride of v (external mode)
/// \param[in,out] work   workspace of the group, at least N elements
/// \param[out]    x      the N entries of the vector
template<bool internal_rhs, typename Solver, typename Sync>
TDLS_HOST_DEVICE inline void share(const int tx, Sync& sync, const double* v, const int stride,
                                   double* work, double* x) {
    constexpr int rows    = Solver::rows_per_thread;
    constexpr int threads = Solver::threads_per_system;
    if constexpr (internal_rhs) {
        for (int K = 0; K < rows && tx + K * threads < N; ++K)
            work[tx + K * threads] = v[K];
        sync();
        for (int i = 0; i < N; ++i)
            x[i] = work[i];
    } else {
        for (int i = 0; i < N; ++i)
            x[i] = v[i * stride];
    }
    sync();
}

/// \brief Integrate one time step at one integration point with a
/// CooperativeLUpp solver, called by every thread of the group that
/// solves the point.
///
/// Every thread keeps its own copy of the state and computes the whole
/// residual, which costs little at N = 7. It builds only the vectors of
/// the jacobian (columns under the row-major layout, rows under the
/// column-major one) and the entries of the residual that it holds in
/// the solver, so the solver reads them without a barrier. On return, the Newton
/// step is visible to the whole group: in r when it is external, in the
/// workspace otherwise, where each thread copies its entries. A barrier
/// then keeps the next iteration from overwriting the step before every
/// thread has read it.
///
/// The Newton loop and the tangent operator make the same calls in every
/// group that shares a barrier: a group that has converged, or failed,
/// keeps pace without updating its state while `any` reports a group
/// still iterating. `any` reduces a flag over the scope of the barrier:
/// the flag itself when the barrier covers exactly the group, the
/// logical or of the work-group under a SYCL work-group barrier.
/// \tparam internal_rhs    residency of r: the slice of the thread or the
///         whole vector
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of J
/// \tparam Solver          a GroupSolver
/// \tparam Sync            callable type of the barrier
/// \tparam Any             callable type of the reduction
/// \param[in]     tx         rank of the thread in its group
/// \param[in]     sync       barrier of the group
/// \param[in]     any        reduction of a flag over the scope of the
///                barrier
/// \param[in]     deto       strain increment of the step
/// \param[in]     dt         time step
/// \param[in,out] eel        elastic strain, advanced on success
/// \param[in,out] p          viscoplastic multiplier, advanced on success
/// \param[out]    sig        stress at the end of the step
/// \param[out]    Dt         consistent tangent operator, 6 x 6 row-major
/// \param[in,out] J          jacobian storage: the vectors of the thread,
///                or the whole matrix
/// \param[in]     J_stride   element stride of J (external mode)
/// \param[in,out] piv        pivot storage
/// \param[in]     piv_stride element stride of piv (external mode)
/// \param[in,out] r          residual storage
/// \param[in]     rhs_stride element stride of r (external mode)
/// \param[in,out] work       workspace of the group, Solver::workspace_size
///                elements
/// \return false when Newton did not converge or a jacobian was singular,
///         the same verdict in every thread of the group
template<bool internal_rhs, bool internal_piv, bool internal_matrix, typename Solver, typename Sync,
         typename Any>
TDLS_HOST_DEVICE inline bool
integrate_group(const int tx, Sync& sync, Any& any, const double* deto, const double dt,
                double* eel, double& p, double* sig, double* Dt, double* J, const int J_stride,
                int* piv, const int piv_stride, double* r, const int rhs_stride, double* work) {
    constexpr int rows    = Solver::rows_per_thread;
    constexpr int threads = Solver::threads_per_system;
    // Slot K of the thread holds vector tx + K * threads of the jacobian:
    // its column under the row-major layout of the solver, its row under
    // the column-major one, the vectors that sit at consecutive addresses
    // for consecutive threads. Element c of the vector sits at
    // c * N + tx + K * threads in the whole matrix, at c * rows + K in the
    // slice of the thread.
    auto J_at = [=](const int K, const int c) -> double& {
        return J[internal_matrix ? c * rows + K : (c * N + tx + K * threads) * J_stride];
    };
    // Entry of the jacobian at element c of vector v.
    auto entry = [&](const int v, const int c, const double* n, const double dp, const double iseq,
                     const double df) {
        return Solver::transposed ? jacobian_entry(c, v, n, dp, iseq, df, dt)
                                  : jacobian_entry(v, c, n, dp, iseq, df, dt);
    };
    auto r_at = [=](const int K) -> double& {
        return r[internal_rhs ? K : (tx + K * threads) * rhs_stride];
    };

    double deel[stensor_size] = {};
    double dp                 = 0;
    double n[stensor_size];
    double f, df, iseq;

    bool converged = false, failed = false;
    for (int iter = 0; iter < newton_max_iter; ++iter) {
        double eel_n[stensor_size], res[N];
        for (int i = 0; i < stensor_size; ++i)
            eel_n[i] = eel[i] + deel[i];
        hooke(eel_n, sig);
        flow(sig, n, f, df, iseq);
        double norm = 0;
        for (int i = 0; i < stensor_size; ++i) {
            res[i] = deel[i] + dp * n[i] - deto[i];
            norm += res[i] * res[i];
        }
        res[6] = dp - dt * f;
        norm += res[6] * res[6];
        if (!converged && !failed) converged = std::sqrt(norm) / N < newton_epsilon;
        if (!any(!converged && !failed)) break;

        // The vectors of the thread, then the Newton step of the group.
        for (int K = 0; K < rows && tx + K * threads < N; ++K) {
            const int v = tx + K * threads;
            for (int c = 0; c < N; ++c)
                J_at(K, c) = entry(v, c, n, dp, iseq, df);
            r_at(K) = res[v];
        }
        const bool ok = Solver::template solve_inplace<internal_rhs, internal_piv, internal_matrix>(
            tx, J, J_stride, piv, piv_stride, r, rhs_stride, work, sync);
        double c[N];
        share<internal_rhs, Solver>(tx, sync, r, rhs_stride, work, c);
        if (converged || failed) continue;
        if (!ok) {
            failed = true;
            continue;
        }
        for (int i = 0; i < stensor_size; ++i)
            deel[i] -= c[i];
        dp -= c[6];
    }
    const bool success = converged && !failed;
    if (success) {
        for (int i = 0; i < stensor_size; ++i)
            eel[i] += deel[i];
        p += dp;
    }

    // Consistent tangent operator: one factorization, then the six
    // canonical columns of the elastic strain, solved one by one as
    // MFront does.
    for (int K = 0; K < rows && tx + K * threads < N; ++K)
        for (int c = 0; c < N; ++c)
            J_at(K, c) = entry(tx + K * threads, c, n, dp, iseq, df);
    const bool factored = Solver::template factorize<internal_piv, internal_matrix>(
        tx, J, J_stride, piv, piv_stride, work, sync);
    for (int j = 0; j < stensor_size; ++j) {
        double X[N];
        Solver::template substitute_canonical<internal_rhs, internal_piv, internal_matrix>(
            tx, J, J_stride, piv, piv_stride, j, r, rhs_stride, work, sync);
        share<internal_rhs, Solver>(tx, sync, r, rhs_stride, work, X);
        const double t = lambda * (X[0] + X[1] + X[2]);
        for (int i = 0; i < stensor_size; ++i)
            Dt[i * stensor_size + j] = 2 * mu * X[i] + (i < 3 ? t : 0);
    }
    return success && factored;
}

/// \brief Reference solution of one step by the radial return.
///
/// With isotropic elasticity and a von Mises flow, the flow direction
/// is the one of the elastic predictor and the whole step reduces to a
/// scalar equation in dp: seq = seq_trial - 3 mu dp together with
/// dp = dt A (seq / s0)^m. A scalar Newton started at zero converges
/// monotonically. The 7 x 7 system of integrate() must reproduce this
/// solution: it is written for the general case, the isotropic case
/// checks it.
/// \param[in]  eel  elastic strain at the beginning of the step
/// \param[in]  deto strain increment of the step
/// \param[in]  dt   time step
/// \param[out] sig  stress at the end of the step
/// \return the increment of the viscoplastic multiplier
inline double radial_return(const double* eel, const double* deto, const double dt, double* sig) {
    double eel_tr[stensor_size];
    for (int i = 0; i < stensor_size; ++i)
        eel_tr[i] = eel[i] + deto[i];
    hooke(eel_tr, sig);
    double n[stensor_size];
    double f, df, iseq;
    flow(sig, n, f, df, iseq);
    const double seq_tr = von_mises(sig);

    double dp = 0;
    for (int iter = 0; iter < 100; ++iter) {
        const double seq = seq_tr - 3 * mu * dp;
        const double g   = dp - dt * A * std::pow(seq / s0, m);
        const double dg  = 1 + dt * A * m * std::pow(seq / s0, m - 1) * 3 * mu / s0;
        const double c   = g / dg;
        dp -= c;
        if (std::fabs(c) <= 1e-16 * (1 + dp)) break;
    }
    for (int i = 0; i < stensor_size; ++i)
        sig[i] -= 2 * mu * dp * n[i];
    return dp;
}

/// \brief Largest relative deviation between a consistent tangent
/// operator and central differences of the integration, column by
/// column, for one step taken from a given state.
/// \param[in] eel0 elastic strain at the beginning of the step
/// \param[in] p0   viscoplastic multiplier at the beginning of the step
/// \param[in] deto strain increment of the step
/// \param[in] dt   time step
/// \param[in] Dt   tangent operator to check, 6 x 6 row-major
/// \return the deviation, relative to the largest entry of Dt
inline double tangent_deviation(const double* eel0, const double p0, const double* deto,
                                const double dt, const double* Dt) {
    constexpr double h = 1e-8;
    double dmax        = 0;
    for (int i = 0; i < stensor_size * stensor_size; ++i) {
        if (!std::isfinite(Dt[i])) return std::numeric_limits<double>::infinity();
        dmax = std::fmax(dmax, std::fabs(Dt[i]));
    }

    double deviation = 0;
    for (int j = 0; j < stensor_size; ++j) {
        double sig_plus[stensor_size], sig_minus[stensor_size];
        for (int side = 0; side < 2; ++side) {
            double eel[stensor_size], deto_h[stensor_size], Dt_h[stensor_size * stensor_size];
            double J[N * N], r[N];
            int piv[N];
            double p = p0;
            for (int k = 0; k < stensor_size; ++k) {
                eel[k]    = eel0[k];
                deto_h[k] = deto[k];
            }
            deto_h[j] += side == 0 ? h : -h;
            if (!integrate<true, true, true>(deto_h, dt, eel, p, side == 0 ? sig_plus : sig_minus,
                                             Dt_h, J, 1, piv, 1, r, 1))
                return 1;
        }
        for (int i = 0; i < stensor_size; ++i) {
            const double fd = (sig_plus[i] - sig_minus[i]) / (2 * h);
            if (!std::isfinite(fd)) return std::numeric_limits<double>::infinity();
            deviation = std::fmax(deviation, std::fabs(fd - Dt[i * stensor_size + j]) / dmax);
        }
    }
    return deviation;
}

/// \brief Outcome of the host-side checks of a batch, see check_last_step.
struct BatchCheck {
    int failures     = 0;    ///< points whose integration failed
    double sig_error = 0;    ///< worst relative stress deviation from the radial return
    double tangent   = 0;    ///< worst relative tangent deviation on the sampled points
    bool p_monotonic = true; ///< no point saw its viscoplastic multiplier decrease
};

/// \brief Serial checks of the last step of a batch, from the state the
/// batch recorded before it: every point against the radial return,
/// one point out of `sample` against central differences of the
/// tangent operator, and the monotony of p everywhere.
/// \param[in] points     number of integration points
/// \param[in] dt         time step
/// \param[in] soa        true when component k of point i sits at
///                       k * points + i, false when it sits at
///                       i * (number of components) + k
/// \param[in] eel_before elastic strain before the last step
/// \param[in] p_before   viscoplastic multiplier before the last step
/// \param[in] sig        stress after the last step
/// \param[in] p          viscoplastic multiplier after the last step
/// \param[in] Dt         tangent operator after the last step
/// \param[in] ok         per-point success flags
/// \param[in] sample     sampling period of the tangent check
/// \return the outcome
inline BatchCheck check_last_step(const int points, const double dt, const bool soa,
                                  const double* eel_before, const double* p_before,
                                  const double* sig, const double* p, const double* Dt,
                                  const int* ok, const int sample) {
    BatchCheck out;
    const auto at = [&](const double* a, const int i, const int k, const int nc) {
        return soa ? a[static_cast<std::size_t>(k) * points + i]
                   : a[static_cast<std::size_t>(i) * nc + k];
    };
    for (int i = 0; i < points; ++i) {
        if (!ok[i]) {
            ++out.failures;
            continue;
        }
        double eel0[stensor_size], deto[stensor_size], sig_ref[stensor_size];
        for (int k = 0; k < stensor_size; ++k)
            eel0[k] = at(eel_before, i, k, stensor_size);
        strain_increment(static_cast<unsigned long long>(i), deto);
        radial_return(eel0, deto, dt, sig_ref);
        double smax = 0, err = 0;
        for (int k = 0; k < stensor_size; ++k) {
            const double s = at(sig, i, k, stensor_size);
            if (!std::isfinite(s)) err = std::numeric_limits<double>::infinity();
            smax = std::fmax(smax, std::fabs(sig_ref[k]));
            err  = std::fmax(err, std::fabs(s - sig_ref[k]));
        }
        out.sig_error = std::fmax(out.sig_error, err / smax);
        if (p[i] < p_before[i]) out.p_monotonic = false;
        if (i % sample == 0) {
            double Dt_i[stensor_size * stensor_size];
            for (int k = 0; k < stensor_size * stensor_size; ++k)
                Dt_i[k] = at(Dt, i, k, stensor_size * stensor_size);
            out.tangent =
                std::fmax(out.tangent, tangent_deviation(eel0, p_before[i], deto, dt, Dt_i));
        }
    }
    return out;
}

} // namespace norton

#endif // TDLS_EXAMPLES_NORTON_HPP
