/// \file
/// \brief Stiff chemical kinetics integrated with an implicit
/// Runge-Kutta method: the physics of the implicit_ode examples.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The Robertson kinetics, the classical stiff benchmark (a slow
/// reaction feeding two fast ones), is integrated with the 3-stage
/// Radau IIA method of order 5. The Newton systems of a step have size
/// N = stages x species = 9, fixed by the method and by the chemical
/// mechanism: a property of the program, so TiledLUppSolverStatic
/// applies.
///
/// Following the classical RADAU5 practice, the Newton matrix is built
/// from the Jacobian frozen at the beginning of the step, factorized
/// once, and the factorization is reused by every Newton iteration
/// through substitute(): the canonical reason for the split
/// factorize/substitute interface. When Newton stalls, the Jacobian is
/// refreshed at the last stage and the step retried, as stiff
/// integrators do.
///
/// A temperature factor scales the reaction rates, so that the cells
/// of a batch carry different matrices, generated on the fly from
/// their inputs.

#ifndef TDLS_EXAMPLES_ROBERTSON_HPP
#define TDLS_EXAMPLES_ROBERTSON_HPP

#include <cmath>

#include <tdls/tdls.hpp>

namespace robertson {

constexpr int species = 3;                ///< fixed by the Robertson mechanism
constexpr int stages  = 3;                ///< fixed by the Radau IIA method
constexpr int N       = stages * species; ///< Newton system dimension

/// \brief LU solver of the Newton systems.
using Solver =
    tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double>{.tile_size = 3}>;

/// \brief Butcher matrix of Radau IIA with 3 stages (order 5). The
/// method is stiffly accurate: the solution of the step is the last
/// stage.
/// \param[out] butcher Butcher matrix of the method
TDLS_HOST_DEVICE inline void radau_butcher(double (&butcher)[stages][stages]) {
    const double s6 = std::sqrt(6.0);
    butcher[0][0]   = (88 - 7 * s6) / 360;
    butcher[0][1]   = (296 - 169 * s6) / 1800;
    butcher[0][2]   = (-2 + 3 * s6) / 225;
    butcher[1][0]   = (296 + 169 * s6) / 1800;
    butcher[1][1]   = (88 + 7 * s6) / 360;
    butcher[1][2]   = (-2 - 3 * s6) / 225;
    butcher[2][0]   = (16 - s6) / 36;
    butcher[2][1]   = (16 + s6) / 36;
    butcher[2][2]   = 1.0 / 9;
}

/// \brief Right-hand side of the Robertson kinetics with a temperature
/// factor scaling all reaction rates.
/// \param[in]  theta temperature factor of the cell
/// \param[in]  y     species concentrations
/// \param[out] f     time derivatives
TDLS_HOST_DEVICE inline void rhs(const double theta, const double* y, double* f) {
    f[0] = theta * (-0.04 * y[0] + 1e4 * y[1] * y[2]);
    f[1] = theta * (0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]);
    f[2] = theta * (3e7 * y[1] * y[1]);
}

/// \brief Analytic Jacobian of the scaled Robertson kinetics, row-major
/// 3 x 3.
/// \param[in]  theta temperature factor of the cell
/// \param[in]  y     species concentrations
/// \param[out] J     derivative of rhs with respect to y
TDLS_HOST_DEVICE inline void jacobian(const double theta, const double* y, double* J) {
    J[0] = theta * -0.04;
    J[1] = theta * 1e4 * y[2];
    J[2] = theta * 1e4 * y[1];
    J[3] = theta * 0.04;
    J[4] = theta * (-1e4 * y[2] - 6e7 * y[1]);
    J[5] = theta * -1e4 * y[1];
    J[6] = 0.0;
    J[7] = theta * 6e7 * y[1];
    J[8] = 0.0;
}

/// \brief Entry (row, col) of the Newton matrix of a Radau IIA step:
/// M[(i,a),(j,b)] = delta_ij delta_ab - h butcher[i][j] J[a][b], with
/// row = i * species + a and col = j * species + b.
/// \param[in] row     row of the entry
/// \param[in] col     column of the entry
/// \param[in] butcher Butcher matrix of the method
/// \param[in] J       Jacobian of the kinetics, row-major 3 x 3
/// \param[in] h       time step
/// \return the entry
TDLS_HOST_DEVICE inline double newton_entry(const int row, const int col,
                                            const double (&butcher)[stages][stages],
                                            const double* J, const double h) {
    const int i = row / species, a = row % species;
    const int j = col / species, b = col % species;
    return (i == j && a == b ? 1.0 : 0.0) - h * butcher[i][j] * J[a * species + b];
}

/// \brief Newton matrix of a Radau IIA step with the Jacobian frozen at
/// y: M[(i,a),(j,b)] = delta_ij delta_ab - h butcher[i][j] J[a][b],
/// constant during the whole step by construction.
/// \tparam internal_matrix residency of M
/// \param[in]  butcher Butcher matrix of the method
/// \param[in]  theta   temperature factor of the cell
/// \param[in]  y       state the Jacobian is frozen at
/// \param[in]  h       time step
/// \param[out] M       matrix storage, N x N elements
/// \param[in]  stride  element stride of M (external mode)
template<bool internal_matrix>
TDLS_HOST_DEVICE inline void newton_matrix(const double (&butcher)[stages][stages],
                                           const double theta, const double* y, const double h,
                                           double* M, const int stride) {
    double J[species * species];
    jacobian(theta, y, J);
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            M[internal_matrix ? row * N + col : (row * N + col) * stride] =
                newton_entry(row, col, butcher, J, h);
}

/// \brief One Radau IIA step with a frozen Jacobian: the Newton matrix
/// is factorized once and the factorization is reused by every Newton
/// iteration. The Jacobian is frozen at the step start and, when Newton
/// stalls, refreshed at the last stage (strong transients at high
/// temperature factors).
///
/// The solver operands (matrix, pivot, residual, correction) are the
/// caller's: local arrays under the internal residencies, slices of a
/// batch in remote memory under the external ones. The stage values
/// stay local.
/// \tparam internal_rhs    residency of r and dz
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of M
/// \param[in]     butcher    Butcher matrix of the method
/// \param[in]     theta      temperature factor of the cell
/// \param[in]     h          time step
/// \param[in,out] y          cell state, advanced by h on success
/// \param[in,out] M          Newton matrix storage, N x N elements
/// \param[in]     M_stride   element stride of M (external mode)
/// \param[in,out] piv        pivot storage, N elements
/// \param[in]     piv_stride element stride of piv (external mode)
/// \param[in,out] r          residual storage, N elements
/// \param[in,out] dz         correction storage, N elements
/// \param[in]     rhs_stride element stride of r and dz (external mode)
/// \return true when the factorizations succeeded and Newton converged
template<bool internal_rhs, bool internal_piv, bool internal_matrix>
[[nodiscard]] TDLS_HOST_DEVICE inline bool
radau_step(const double (&butcher)[stages][stages], const double theta, const double h,
           double (&y)[species], double* M, const int M_stride, int* piv, const int piv_stride,
           double* r, double* dz, const int rhs_stride) {
    // Newton iterates on the stacked stage values Z = (Y1, Y2, Y3),
    // started from the current state and kept across refreshes.
    double Z[N];
    for (int i = 0; i < stages; ++i)
        for (int a = 0; a < species; ++a)
            Z[i * species + a] = y[a];

    for (int attempt = 0; attempt < 3; ++attempt) {
        newton_matrix<internal_matrix>(
            butcher, theta, attempt == 0 ? y : &Z[(stages - 1) * species], h, M, M_stride);
        if (!Solver::factorize<internal_piv, internal_matrix>(M, M_stride, piv, piv_stride))
            return false;

        double correction = 1.0;
        for (int iter = 0; iter < 30 && correction > 1e-12; ++iter) {
            // Residual R_i = Y_i - y - h sum_j butcher[i][j] f(Y_j).
            double f[stages][species];
            for (int j = 0; j < stages; ++j)
                rhs(theta, &Z[j * species], f[j]);
            for (int i = 0; i < stages; ++i)
                for (int a = 0; a < species; ++a) {
                    double acc = Z[i * species + a] - y[a];
                    for (int j = 0; j < stages; ++j)
                        acc -= h * butcher[i][j] * f[j][a];
                    const int e                          = i * species + a;
                    r[internal_rhs ? e : e * rhs_stride] = -acc;
                }
            Solver::substitute<internal_rhs, internal_piv, internal_matrix>(
                M, M_stride, piv, piv_stride, r, dz, rhs_stride);
            correction = 0.0;
            for (int e = 0; e < N; ++e) {
                const double d = dz[internal_rhs ? e : e * rhs_stride];
                if (!std::isfinite(d)) return false;
                Z[e] += d;
                correction = std::fmax(correction, std::fabs(d));
            }
        }
        if (correction <= 1e-12) {
            // Stiffly accurate method: the new state is the last stage.
            for (int a = 0; a < species; ++a)
                y[a] = Z[(stages - 1) * species + a];
            return true;
        }
    }
    return false;
}

/// \brief LU solver of the Newton systems for a group of threads, the
/// solver of the CooperativeLUpp examples.
/// \tparam rows_per_thread rows held by each thread of the group
/// \tparam unroll          unroll policy: true on GPU, false on CPU
template<int rows_per_thread, bool unroll>
using GroupSolver = tdls::CooperativeLUppSolverStatic<double, N,
                                                      tdls::CooperativeLUppConfig<double>{
                                                          .rows_per_thread = rows_per_thread,
                                                          .unroll_loops    = unroll}>;

/// \brief One Radau IIA step with a frozen Jacobian and a CooperativeLUpp
/// solver, called by every thread of the group that solves the cell.
///
/// The scheme is the one of radau_step. Every thread keeps its own copy
/// of the stage values and computes the whole residual, which costs
/// little at N = 9. It builds only the rows of the Newton matrix and the
/// entries of the residual that it holds in the solver, so the solver
/// reads them without a barrier. On return, the correction is visible
/// to the whole group: in dz when it is external, in the workspace
/// otherwise, where each thread copies its entries. A barrier then keeps
/// the next iteration from overwriting it before every thread has read
/// it.
///
/// Every group sharing a barrier makes the same calls: a group that has
/// converged, or failed, keeps pace without updating its state while
/// `any` reports a group still iterating. `any` reduces a flag over the
/// scope of the barrier: the flag itself when the barrier covers exactly
/// the group, the logical or of the work-group under a SYCL work-group
/// barrier.
/// \tparam internal_rhs    residency of r and dz: the slice of the thread
///         or the whole vectors
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of M
/// \tparam Solver          a GroupSolver
/// \tparam Sync            callable type of the barrier
/// \tparam Any             callable type of the reduction
/// \param[in]     tx         rank of the thread in its group
/// \param[in]     sync       barrier of the group
/// \param[in]     any        reduction of a flag over the scope of the
///                barrier
/// \param[in]     butcher    Butcher matrix of the method
/// \param[in]     theta      temperature factor of the cell
/// \param[in]     h          time step
/// \param[in,out] y          cell state, advanced by h on success
/// \param[in,out] M          Newton matrix storage: the rows of the thread,
///                or the whole matrix
/// \param[in]     M_stride   element stride of M (external mode)
/// \param[in,out] piv        pivot storage
/// \param[in]     piv_stride element stride of piv (external mode)
/// \param[in,out] r          residual storage
/// \param[in,out] dz         correction storage
/// \param[in]     rhs_stride element stride of r and dz (external mode)
/// \param[in,out] work       workspace of the group,
///                Solver::workspace_size elements
/// \return true when the factorizations succeeded and Newton converged,
///         the same verdict in every thread of the group
template<bool internal_rhs, bool internal_piv, bool internal_matrix, typename Solver, typename Sync,
         typename Any>
[[nodiscard]] TDLS_HOST_DEVICE inline bool
radau_step_group(const int tx, Sync& sync, Any& any, const double (&butcher)[stages][stages],
                 const double theta, const double h, double (&y)[species], double* M,
                 const int M_stride, int* piv, const int piv_stride, double* r, double* dz,
                 const int rhs_stride, double* work) {
    constexpr int rows    = Solver::rows_per_thread;
    constexpr int threads = Solver::threads_per_system;
    // Addressing of the rows of the thread: slot K holds row tx + K * threads.
    auto M_at = [=](const int K, const int col) -> double& {
        return M[internal_matrix ? K * N + col : ((tx + K * threads) * N + col) * M_stride];
    };
    auto r_at = [=](const int K) -> double& {
        return r[internal_rhs ? K : (tx + K * threads) * rhs_stride];
    };
    auto dz_at = [=](const int K) -> double& {
        return dz[internal_rhs ? K : (tx + K * threads) * rhs_stride];
    };
    // The whole correction, visible to every thread: from dz when it is
    // external, through the workspace otherwise.
    auto share = [&](double* d) {
        if constexpr (internal_rhs) {
            for (int K = 0; K < rows && tx + K * threads < N; ++K)
                work[tx + K * threads] = dz_at(K);
            sync();
            for (int e = 0; e < N; ++e)
                d[e] = work[e];
        } else {
            for (int e = 0; e < N; ++e)
                d[e] = dz[e * rhs_stride];
        }
        sync();
    };

    double Z[N];
    for (int i = 0; i < stages; ++i)
        for (int a = 0; a < species; ++a)
            Z[i * species + a] = y[a];

    bool done = false, success = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (!any(!done)) break;
        // The rows of the thread of the Newton matrix, the Jacobian frozen
        // at the step start, or refreshed at the last stage.
        double J[species * species];
        jacobian(theta, attempt == 0 ? y : &Z[(stages - 1) * species], J);
        for (int K = 0; K < rows && tx + K * threads < N; ++K)
            for (int col = 0; col < N; ++col)
                M_at(K, col) = newton_entry(tx + K * threads, col, butcher, J, h);
        const bool factored = Solver::template factorize<internal_piv, internal_matrix>(
            tx, M, M_stride, piv, piv_stride, work, sync);
        if (!factored) done = true;

        double correction = 1.0;
        for (int iter = 0; iter < 30; ++iter) {
            const bool iterating = !done && correction > 1e-12;
            if (!any(iterating)) break;
            // Residual R_i = Y_i - y - h sum_j butcher[i][j] f(Y_j), the
            // entries of the rows of the thread.
            double f[stages][species];
            for (int j = 0; j < stages; ++j)
                rhs(theta, &Z[j * species], f[j]);
            for (int K = 0; K < rows && tx + K * threads < N; ++K) {
                const int e = tx + K * threads;
                const int i = e / species, a = e % species;
                double acc = Z[e] - y[a];
                for (int j = 0; j < stages; ++j)
                    acc -= h * butcher[i][j] * f[j][a];
                r_at(K) = -acc;
            }
            Solver::template substitute<internal_rhs, internal_piv, internal_matrix>(
                tx, M, M_stride, piv, piv_stride, r, dz, rhs_stride, work, sync);
            double d[N];
            share(d);
            if (!iterating) continue;
            correction = 0.0;
            for (int e = 0; e < N; ++e) {
                if (!std::isfinite(d[e])) {
                    done = true;
                    break;
                }
                Z[e] += d[e];
                correction = std::fmax(correction, std::fabs(d[e]));
            }
        }
        if (!done && correction <= 1e-12) {
            // Stiffly accurate method: the new state is the last stage.
            for (int a = 0; a < species; ++a)
                y[a] = Z[(stages - 1) * species + a];
            done    = true;
            success = true;
        }
    }
    return success;
}

} // namespace robertson

#endif // TDLS_EXAMPLES_ROBERTSON_HPP
