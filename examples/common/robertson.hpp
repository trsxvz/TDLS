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
    for (int i = 0; i < stages; ++i)
        for (int j = 0; j < stages; ++j)
            for (int a = 0; a < species; ++a)
                for (int b = 0; b < species; ++b) {
                    const int e = (i * species + a) * N + (j * species + b);
                    M[internal_matrix ? e : e * stride] =
                        (i == j && a == b ? 1.0 : 0.0) - h * butcher[i][j] * J[a * species + b];
                }
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

} // namespace robertson

#endif // TDLS_EXAMPLES_ROBERTSON_HPP
