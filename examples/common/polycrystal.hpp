/// \file
/// \brief Implicit integration of a Berveiller-Zaoui polycrystal: the
/// physics of the polycrystal and aggregate examples.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A material point is an aggregate of G grains of a face-centred cubic
/// crystal, each with its own orientation and the volume fraction
/// phi = 1 / G. Elasticity is isotropic and uniform. Every grain flows
/// on its 12 slip systems {111}<110> by a Norton law,
///
///   dgamma_s/dt = gdot0 sign(tau_s) |tau_s / tau0|^m,   tau_s = sig_g : mu_s,
///
/// with mu_s the symmetric Schmid tensor of system s in the grain. The
/// macroscopic strain is imposed. The macroscopic stress follows from
/// the macroscopic elastic strain, Sig = D : Eel, and the stress of a
/// grain from the Berveiller-Zaoui localization, with a constant
/// accommodation factor alpha,
///
///   sig_g = Sig + c (Evp - evp_g),   c = 2 mu (1 - beta) alpha,
///   Evp = sum_g phi evp_g.
///
/// A time step is integrated by the implicit Euler scheme, as the
/// polycrystal behaviours of MFront do. The unknowns are the increments
/// of the macroscopic elastic and viscoplastic strains (12, the border),
/// then the increment of the viscoplastic strain of each grain (6 per
/// grain, the blocks). The residual is
///
///   f_Eel = dEel + dEvp - dEto
///   f_Evp = dEvp - sum_g phi devp_g
///   f_g   = devp_g - dt sum_s dgamma_s/dt mu_s
///
/// and its jacobian is an arrowhead matrix. With G_g = dt sum_s
/// d(dgamma_s/dt)/dtau mu_s (x) mu_s, the block of grain g is I + c G_g.
/// Its rows meet the border through -2 mu G_g (columns of dEel, the
/// Schmid tensors being deviatoric) and -c G_g (columns of dEvp). Grain g
/// meets the border through the rows of f_Evp only, its unknown i in
/// row 6 + i, with the coefficient -phi: the diagonal coupling from
/// coupling_row 6. Two grains never couple. ArrowheadLUpp solves it in a
/// time linear in G, where a dense LU of the whole jacobian costs
/// (12 + 6 G)^3 / 3 multiply-adds.
///
/// The jacobian is assembled straight into the Packed storage through
/// the storage map: the entries of the pattern only, without index
/// arrays. The per-point vectors (state, increments, residual) follow
/// the order of the unknowns: Eel, Evp, then evp of each grain.
/// Symmetric tensors are 6-vectors in the TFEL convention (xx, yy, zz,
/// sqrt(2) xy, sqrt(2) xz, sqrt(2) yz), as in norton.hpp.

#ifndef TDLS_EXAMPLES_POLYCRYSTAL_HPP
#define TDLS_EXAMPLES_POLYCRYSTAL_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include <tdls/tdls.hpp>

#include "hash01.hpp"

namespace polycrystal {

constexpr int stensor_size = 6;  ///< components of a symmetric tensor in 3D
constexpr int border       = 12; ///< macroscopic unknowns: dEel then dEvp
constexpr int slip_systems = 12; ///< {111}<110> slip systems of the FCC crystal
/// Grains of the polycrystal examples, fixed at compile time.
constexpr int fixed_grains = 12;

// Material, in SI units.
constexpr double young  = 200e9;                                  ///< Young modulus (Pa)
constexpr double nu     = 0.3;                                    ///< Poisson ratio
constexpr double lambda = young * nu / ((1 + nu) * (1 - 2 * nu)); ///< first Lame coefficient
constexpr double mu     = young / (2 * (1 + nu));                 ///< shear modulus
constexpr double gdot0  = 1e-3;                                   ///< reference slip rate (1/s)
constexpr double tau0   = 100e6; ///< reference resolved stress (Pa)
constexpr double m      = 10;    ///< Norton exponent
/// Eshelby coefficient of a sphere, beta = 2 (4 - 5 nu) / (15 (1 - nu)).
constexpr double beta = 2 * (4 - 5 * nu) / (15 * (1 - nu));
/// Plastic accommodation of the Berveiller-Zaoui model, kept constant.
constexpr double alpha = 0.05;
/// Localization coefficient c = 2 mu (1 - beta) alpha.
constexpr double c = 2 * mu * (1 - beta) * alpha;

constexpr double newton_epsilon = 1e-15; ///< convergence criterion on max |f| (strain units)
constexpr int newton_max_iter   = 50;    ///< Newton iterations before giving up

/// \brief Configuration of the arrowhead solvers: the diagonal coupling
/// of the rows of f_Evp, the Packed storage.
constexpr tdls::ArrowheadLUppConfig<double> config{.coupling = tdls::ArrowheadCoupling::Diagonal,
                                                   .storage  = tdls::ArrowheadStorage::Packed};
/// Index of every entry of the jacobian in the Packed storage.
using Map = tdls::ArrowheadStorageMap<config.storage, config.coupling, config.layout>;
/// Index of every entry of the jacobian in a full row-major matrix: the
/// dense reference.
using FullMap = tdls::ArrowheadStorageMap<tdls::ArrowheadStorage::Full, config.coupling,
                                          tdls::MatrixLayout::RowMajor>;

/// \brief Shape of the jacobian of an aggregate of G grains: the 12
/// macroscopic unknowns, then 6 per grain, coupled through rows 6 to 11.
/// \param[in] grains number of grains
/// \return the shape
TDLS_HOST_DEVICE constexpr tdls::ArrowheadShape shape_of(const int grains) {
    return {border, stensor_size, grains, stensor_size};
}

/// \brief Number of unknowns, 12 + 6 G.
/// \param[in] grains number of grains
/// \return the dimension of the Newton system
TDLS_HOST_DEVICE constexpr int unknowns(const int grains) {
    return border + stensor_size * grains;
}

/// \brief Solver of the polycrystal examples, the shape fixed at compile
/// time.
/// \tparam Grains number of grains
template<int Grains>
using StaticSolver = tdls::ArrowheadLUppSolverStatic<double, shape_of(Grains), config>;
/// \brief Solver of the aggregate examples, the shape given at run time.
using DynamicSolver = tdls::ArrowheadLUppSolverDynamic<double, config>;

/// \brief Strain increment imposed on an integration point at every
/// time step: a tension along x with the elastic Poisson contraction,
/// plus a shear, both scaled per point.
/// \param[in]  point point index
/// \param[out] deto  strain increment of the step
TDLS_HOST_DEVICE inline void strain_increment(const unsigned long long point, double* deto) {
    const double a = 2e-4 * (0.8 + 0.4 * hash01(2 * point));
    const double g = 0.5 * hash01(2 * point + 1);
    deto[0]        = a;
    deto[1]        = -nu * a;
    deto[2]        = -nu * a;
    deto[3]        = a * g;
    deto[4]        = 0;
    deto[5]        = 0;
}

/// \brief Orientation of a grain: a uniform random rotation (Shoemake's
/// quaternion), drawn from the point and grain indices.
/// \param[in]  point point index
/// \param[in]  grain grain index
/// \param[out] R     rotation matrix, row-major
TDLS_HOST_DEVICE inline void orientation(const unsigned long long point, const int grain,
                                         double* R) {
    const unsigned long long key =
        (point * 1000003ull + static_cast<unsigned long long>(grain)) * 3 + 7;
    const double u1 = hash01(key), u2 = hash01(key + 1), u3 = hash01(key + 2);
    const double two_pi = 6.283185307179586;
    const double w      = std::sqrt(1 - u1) * std::sin(two_pi * u2);
    const double x      = std::sqrt(1 - u1) * std::cos(two_pi * u2);
    const double y      = std::sqrt(u1) * std::sin(two_pi * u3);
    const double z      = std::sqrt(u1) * std::cos(two_pi * u3);
    R[0]                = 1 - 2 * (y * y + z * z);
    R[1]                = 2 * (x * y - z * w);
    R[2]                = 2 * (x * z + y * w);
    R[3]                = 2 * (x * y + z * w);
    R[4]                = 1 - 2 * (x * x + z * z);
    R[5]                = 2 * (y * z - x * w);
    R[6]                = 2 * (x * z - y * w);
    R[7]                = 2 * (y * z + x * w);
    R[8]                = 1 - 2 * (x * x + y * y);
}

/// \brief Schmid tensor of a slip system, sym(d (x) n) with the normal
/// and the direction rotated into the grain.
/// \param[in]  R  orientation of the grain
/// \param[in]  s  slip system, 0 to 11: plane s / 3, direction s % 3
/// \param[out] ms Schmid tensor
TDLS_HOST_DEVICE inline void schmid(const double* R, const int s, double* ms) {
    // The 4 planes {111}, and the 3 directions <110> of each plane.
    constexpr double planes[4][3]  = {{1, 1, 1}, {-1, 1, 1}, {1, -1, 1}, {1, 1, -1}};
    constexpr double dirs[4][3][3] = {{{0, 1, -1}, {1, 0, -1}, {1, -1, 0}},
                                      {{0, 1, -1}, {1, 0, 1}, {1, 1, 0}},
                                      {{0, 1, 1}, {1, 0, -1}, {1, 1, 0}},
                                      {{0, 1, 1}, {1, 0, 1}, {1, -1, 0}}};
    constexpr double inv_sqrt3 = 0.5773502691896258, inv_sqrt2 = 0.7071067811865476;
    double nn[3], dd[3];
    for (int a = 0; a < 3; ++a) {
        nn[a] = 0;
        dd[a] = 0;
        for (int b = 0; b < 3; ++b) {
            nn[a] += R[a * 3 + b] * planes[s / 3][b] * inv_sqrt3;
            dd[a] += R[a * 3 + b] * dirs[s / 3][s % 3][b] * inv_sqrt2;
        }
    }
    ms[0] = dd[0] * nn[0];
    ms[1] = dd[1] * nn[1];
    ms[2] = dd[2] * nn[2];
    ms[3] = inv_sqrt2 * (dd[0] * nn[1] + dd[1] * nn[0]);
    ms[4] = inv_sqrt2 * (dd[0] * nn[2] + dd[2] * nn[0]);
    ms[5] = inv_sqrt2 * (dd[1] * nn[2] + dd[2] * nn[1]);
}

/// \brief Isotropic Hooke law, sig = lambda tr(e) Id + 2 mu e.
/// \param[in]  e   strain
/// \param[out] sig stress
TDLS_HOST_DEVICE inline void hooke(const double* e, double* sig) {
    const double t = lambda * (e[0] + e[1] + e[2]);
    for (int i = 0; i < stensor_size; ++i)
        sig[i] = 2 * mu * e[i] + (i < 3 ? t : 0);
}

/// \brief Norton slip rate and its derivative.
/// \param[in]  tau   resolved shear stress
/// \param[out] rate  dgamma/dt
/// \param[out] drate d(dgamma/dt)/dtau
TDLS_HOST_DEVICE inline void slip_rate(const double tau, double& rate, double& drate) {
    const double r = std::fabs(tau) / tau0;
    const double p = std::pow(r, m - 1);
    rate           = (tau < 0 ? -gdot0 : gdot0) * p * r;
    drate          = m * gdot0 * p / tau0;
}

/// \brief Entry i of a per-point vector or of the jacobian storage. The
/// ternary on a template bool folds, exactly as in the solvers.
/// \tparam internal whether the array is a local one (stride ignored)
/// \param[in] v      the array
/// \param[in] i      entry
/// \param[in] stride element stride (external mode)
/// \return the entry
template<bool internal>
TDLS_HOST_DEVICE inline double& at(double* v, const int i, const int stride) {
    return v[internal ? static_cast<std::size_t>(i) : TDLS_STRIDED(i, stride)];
}

/// \brief Stress of a grain at the current increments.
/// \tparam internal_state residency of state and du
/// \param[in]  g            grain
/// \param[in]  Sig          macroscopic stress
/// \param[in]  Evp          macroscopic viscoplastic strain
/// \param[in]  state        state at the beginning of the step
/// \param[in]  du           increments
/// \param[in]  state_stride element stride of state and du
/// \param[out] sig          stress of the grain
template<bool internal_state>
TDLS_HOST_DEVICE inline void grain_stress(const int g, const double* Sig, const double* Evp,
                                          double* state, double* du, const int state_stride,
                                          double* sig) {
    for (int i = 0; i < stensor_size; ++i) {
        const int k = border + stensor_size * g + i;
        sig[i]      = Sig[i] + c * (Evp[i] - at<internal_state>(state, k, state_stride) -
                                    at<internal_state>(du, k, state_stride));
    }
}

/// \brief Residual of the Newton system at the current increments, and
/// its jacobian.
/// \tparam M               storage map of the jacobian
/// \tparam internal_state  residency of state and du
/// \tparam internal_rhs    residency of r
/// \tparam internal_matrix residency of A
/// \param[in]  point        point index
/// \param[in]  grains       number of grains
/// \param[in]  deto         strain increment of the step
/// \param[in]  dt           time step
/// \param[in]  state        state at the beginning of the step
/// \param[in]  du           increments
/// \param[in]  state_stride element stride of state and du
/// \param[out] r            residual, 12 + 6 G entries
/// \param[in]  rhs_stride   element stride of r
/// \param[out] A            jacobian: the entries of the pattern of M
/// \param[in]  A_stride     element stride of A
/// \return max |f|
template<typename M, bool internal_state, bool internal_rhs, bool internal_matrix>
TDLS_HOST_DEVICE inline double residual(const unsigned long long point, const int grains,
                                        const double* deto, const double dt, double* state,
                                        double* du, const int state_stride, double* r,
                                        const int rhs_stride, double* A, const int A_stride) {
    const tdls::ArrowheadShape shape = shape_of(grains);
    const double phi                 = 1.0 / grains;
    // Macroscopic state at the end of the step.
    double Eel[stensor_size], Evp[stensor_size], Sig[stensor_size], mean[stensor_size] = {};
    for (int i = 0; i < stensor_size; ++i) {
        Eel[i] =
            at<internal_state>(state, i, state_stride) + at<internal_state>(du, i, state_stride);
        Evp[i] = at<internal_state>(state, 6 + i, state_stride) +
                 at<internal_state>(du, 6 + i, state_stride);
    }
    hooke(Eel, Sig);
    for (int g = 0; g < grains; ++g)
        for (int i = 0; i < stensor_size; ++i)
            mean[i] += phi * at<internal_state>(du, border + stensor_size * g + i, state_stride);
    // The border rows: Abb = [[I, I], [0, I]].
    double norm = 0;
    for (int i = 0; i < stensor_size; ++i) {
        const double dEel = at<internal_state>(du, i, state_stride);
        const double dEvp = at<internal_state>(du, 6 + i, state_stride);
        const double fe = dEel + dEvp - deto[i], fv = dEvp - mean[i];
        at<internal_rhs>(r, i, rhs_stride)     = fe;
        at<internal_rhs>(r, 6 + i, rhs_stride) = fv;
        norm = std::fmax(norm, std::fmax(std::fabs(fe), std::fabs(fv)));
    }
    for (int i = 0; i < border; ++i)
        for (int j = 0; j < border; ++j)
            at<internal_matrix>(A, M::abb(shape, i, j), A_stride) =
                (i == j || j == i + 6) ? 1.0 : 0.0;
    // The rows of each grain, and its coupling to the rows of f_Evp.
    for (int g = 0; g < grains; ++g) {
        double R[9], sig[stensor_size], f[stensor_size], G[stensor_size * stensor_size] = {};
        orientation(point, g, R);
        grain_stress<internal_state>(g, Sig, Evp, state, du, state_stride, sig);
        for (int i = 0; i < stensor_size; ++i)
            f[i] = at<internal_state>(du, border + stensor_size * g + i, state_stride);
        for (int s = 0; s < slip_systems; ++s) {
            double ms[stensor_size];
            schmid(R, s, ms);
            double tau = 0;
            for (int i = 0; i < stensor_size; ++i)
                tau += sig[i] * ms[i];
            double rate, drate;
            slip_rate(tau, rate, drate);
            for (int i = 0; i < stensor_size; ++i) {
                f[i] -= dt * rate * ms[i];
                for (int j = 0; j < stensor_size; ++j)
                    G[i * stensor_size + j] += dt * drate * ms[i] * ms[j];
            }
        }
        for (int i = 0; i < stensor_size; ++i) {
            at<internal_rhs>(r, border + stensor_size * g + i, rhs_stride) = f[i];
            norm = std::fmax(norm, std::fabs(f[i]));
            at<internal_matrix>(A, M::abk_diag(shape, g, i), A_stride) = -phi;
            for (int j = 0; j < stensor_size; ++j) {
                const double Gij = G[i * stensor_size + j];
                at<internal_matrix>(A, M::d(shape, g, i, j), A_stride) =
                    (i == j ? 1.0 : 0.0) + c * Gij;
                at<internal_matrix>(A, M::akb(shape, g, i, j), A_stride)     = -2 * mu * Gij;
                at<internal_matrix>(A, M::akb(shape, g, i, 6 + j), A_stride) = -c * Gij;
            }
        }
    }
    return norm;
}

/// \brief Integrates one time step at one integration point.
///
/// Every Newton iteration assembles the residual and the jacobian, then
/// solves the arrowhead system in place: by the compile-time solver
/// when Grains > 0, by the runtime solver otherwise. The operands are
/// the caller's: local arrays under the internal residencies, slices of
/// a batch under the external ones. The runtime solver has no residency
/// booleans: it takes the internal operands with a stride of 1, and the
/// pivots of its TiledLUpp solves in piv.
/// \tparam Grains          number of grains at compile time, or 0
/// \tparam internal_state  residency of state and du
/// \tparam internal_rhs    residency of r
/// \tparam internal_work   residency of work and piv
/// \tparam internal_matrix residency of A
/// \param[in]     point        point index
/// \param[in]     grains       number of grains (Grains when it is > 0)
/// \param[in]     deto         strain increment of the step
/// \param[in]     dt           time step
/// \param[in,out] state        state, 12 + 6 G entries, advanced on
///                             success
/// \param[out]    du           increments, 12 + 6 G entries
/// \param[in]     state_stride element stride of state and du
/// \param[out]    A            jacobian storage, Map::size entries
/// \param[in]     A_stride     element stride of A
/// \param[out]    piv          pivots, DynamicSolver::pivot_size entries
///                             (runtime solver only)
/// \param[in]     piv_stride   element stride of piv
/// \param[out]    work         workspace of the solver
/// \param[in]     work_stride  element stride of work
/// \param[out]    r            residual storage, 12 + 6 G entries
/// \param[in]     rhs_stride   element stride of r
/// \param[in,out] iterations   incremented by the Newton iterations
/// \param[in,out] refused      incremented by the blocks the pivot check
///                             refused
/// \return false when Newton did not converge or a jacobian was singular
template<int Grains, bool internal_state, bool internal_rhs, bool internal_work,
         bool internal_matrix>
TDLS_HOST_DEVICE inline bool
integrate(const unsigned long long point, const int grains, const double* deto, const double dt,
          double* state, double* du, const int state_stride, double* A, const int A_stride,
          int* piv, const int piv_stride, double* work, const int work_stride, double* r,
          const int rhs_stride, int& iterations, int& refused) {
    const int n = unknowns(grains);
    for (int i = 0; i < n; ++i)
        at<internal_state>(du, i, state_stride) = 0;
    bool converged = false;
    for (int iter = 0; iter < newton_max_iter; ++iter) {
        const double norm = residual<Map, internal_state, internal_rhs, internal_matrix>(
            point, grains, deto, dt, state, du, state_stride, r, rhs_stride, A, A_stride);
        if (norm < newton_epsilon) {
            converged = true;
            break;
        }
        int refused_now = 0;
        bool ok         = false;
        if constexpr (Grains > 0) {
            (void)piv;
            (void)piv_stride;
            ok = StaticSolver<Grains>::template solve_inplace<internal_rhs, internal_work,
                                                              internal_matrix>(
                A, A_stride, work, work_stride, r, rhs_stride, refused_now);
        } else {
            ok = DynamicSolver::solve_inplace(shape_of(grains), A, internal_matrix ? 1 : A_stride,
                                              piv, internal_work ? 1 : piv_stride, work,
                                              internal_work ? 1 : work_stride, r,
                                              internal_rhs ? 1 : rhs_stride, refused_now);
        }
        if (!ok) return false;
        refused += refused_now;
        ++iterations;
        for (int i = 0; i < n; ++i)
            at<internal_state>(du, i, state_stride) -= at<internal_rhs>(r, i, rhs_stride);
    }
    if (!converged) return false;
    for (int i = 0; i < n; ++i)
        at<internal_state>(state, i, state_stride) += at<internal_state>(du, i, state_stride);
    return true;
}

/// \brief Integrates the loading history of a point from a stress-free
/// state: steps time steps of its strain increment. The state before
/// the last step is copied out, for the checks.
/// \tparam Grains          number of grains at compile time, or 0
/// \tparam internal_state  residency of state and du
/// \tparam internal_rhs    residency of r
/// \tparam internal_work   residency of work and piv
/// \tparam internal_matrix residency of A
/// \param[in]     point         point index
/// \param[in]     grains        number of grains (Grains when it is > 0)
/// \param[in]     steps         number of time steps
/// \param[in]     dt            time step
/// \param[out]    state         state at the end, 12 + 6 G entries
/// \param[out]    du            increments, 12 + 6 G entries
/// \param[in]     state_stride  element stride of state and du
/// \param[out]    before        state before the last step
/// \param[in]     before_stride element stride of before
/// \param[out]    A             jacobian storage
/// \param[in]     A_stride      element stride of A
/// \param[out]    piv           pivots (runtime solver only)
/// \param[in]     piv_stride    element stride of piv
/// \param[out]    work          workspace of the solver
/// \param[in]     work_stride   element stride of work
/// \param[out]    r             residual storage
/// \param[in]     rhs_stride    element stride of r
/// \param[in,out] iterations    incremented by the Newton iterations
/// \param[in,out] refused       incremented by the refused blocks
/// \return false when a step failed
template<int Grains, bool internal_state, bool internal_rhs, bool internal_work,
         bool internal_matrix>
TDLS_HOST_DEVICE inline bool
history(const unsigned long long point, const int grains, const int steps, const double dt,
        double* state, double* du, const int state_stride, double* before, const int before_stride,
        double* A, const int A_stride, int* piv, const int piv_stride, double* work,
        const int work_stride, double* r, const int rhs_stride, int& iterations, int& refused) {
    const int n = unknowns(grains);
    double deto[stensor_size];
    strain_increment(point, deto);
    for (int i = 0; i < n; ++i)
        at<internal_state>(state, i, state_stride) = 0;
    for (int step = 0; step < steps; ++step) {
        if (step == steps - 1)
            for (int i = 0; i < n; ++i)
                before[TDLS_STRIDED(i, before_stride)] = at<internal_state>(state, i, state_stride);
        if (!integrate<Grains, internal_state, internal_rhs, internal_work, internal_matrix>(
                point, grains, deto, dt, state, du, state_stride, A, A_stride, piv, piv_stride,
                work, work_stride, r, rhs_stride, iterations, refused))
            return false;
    }
    return true;
}

/// \brief The dense reference: the same Newton iterations, the whole
/// jacobian assembled in a full matrix and solved by the TiledLUpp
/// runtime solver, in (12 + 6 G)^3 / 3 multiply-adds per iteration.
/// \param[in]     point  point index
/// \param[in]     grains number of grains
/// \param[in]     deto   strain increment of the step
/// \param[in]     dt     time step
/// \param[in,out] state  state, 12 + 6 G entries, advanced on success
/// \return false when Newton did not converge or a jacobian was singular
inline bool integrate_dense(const unsigned long long point, const int grains, const double* deto,
                            const double dt, double* state) {
    const int n = unknowns(grains);
    std::vector<double> du(static_cast<std::size_t>(n)), r(du.size()),
        A(static_cast<std::size_t>(n) * n);
    std::vector<int> piv(du.size());
    for (int iter = 0; iter < newton_max_iter; ++iter) {
        for (double& a : A)
            a = 0; // the zeros outside the pattern
        if (residual<FullMap, true, true, true>(point, grains, deto, dt, state, du.data(), 1,
                                                r.data(), 1, A.data(), 1) < newton_epsilon) {
            for (int i = 0; i < n; ++i)
                state[i] += du[static_cast<std::size_t>(i)];
            return true;
        }
        if (!tdls::TiledLUppSolverDynamic<double>::solve_inplace(n, A.data(), 1, piv.data(), 1,
                                                                 r.data(), 1))
            return false;
        for (std::size_t i = 0; i < du.size(); ++i)
            du[i] -= r[i];
    }
    return false;
}

/// \brief Largest deviation between two vectors, relative to the
/// largest entry of the reference.
/// \param[in] a      vector
/// \param[in] stride element stride of a
/// \param[in] ref    reference vector, contiguous
/// \param[in] n      entries
/// \return max |a - ref| / max |ref|, infinity when a is not finite
inline double deviation(const double* a, const int stride, const double* ref, const int n) {
    double d = 0, s = 0;
    for (int i = 0; i < n; ++i) {
        const double ai = a[TDLS_STRIDED(i, stride)];
        if (!std::isfinite(ai)) return HUGE_VAL;
        d = std::fmax(d, std::fabs(ai - ref[i]));
        s = std::fmax(s, std::fabs(ref[i]));
    }
    return d / s;
}

/// \brief Physical checks of a step.
struct StepCheck {
    double trace       = 0; ///< |tr Evp| / |Evp| at the end of the step, ~0
    double dissipation = 0; ///< smallest sig_g : devp_g over the grains (Pa), >= 0
    double seq         = 0; ///< von Mises macroscopic stress at the end of the step (Pa)
};

/// \brief Checks a step from the states before and after it: the
/// viscoplastic strain is deviatoric, since the Schmid tensors are, and
/// every grain dissipates, since every slip rate has the sign of its
/// resolved shear stress.
/// \param[in] grains number of grains
/// \param[in] before state at the beginning of the step, contiguous
/// \param[in] after  state at the end of the step
/// \param[in] stride element stride of after
/// \return the outcome
inline StepCheck check_step(const int grains, const double* before, const double* after,
                            const int stride) {
    const int n = unknowns(grains);
    std::vector<double> s0(before, before + n), du(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        du[static_cast<std::size_t>(i)] = after[TDLS_STRIDED(i, stride)] - before[i];
    double Eel[stensor_size], Evp[stensor_size], Sig[stensor_size];
    for (int i = 0; i < stensor_size; ++i) {
        Eel[i] = after[TDLS_STRIDED(i, stride)];
        Evp[i] = after[TDLS_STRIDED(6 + i, stride)];
    }
    hooke(Eel, Sig);
    StepCheck out;
    double norm = 0;
    for (int i = 0; i < stensor_size; ++i)
        norm += Evp[i] * Evp[i];
    out.trace       = std::fabs(Evp[0] + Evp[1] + Evp[2]) / std::fmax(std::sqrt(norm), 1e-300);
    out.dissipation = HUGE_VAL;
    for (int g = 0; g < grains; ++g) {
        double sig[stensor_size], w = 0;
        grain_stress<true>(g, Sig, Evp, s0.data(), du.data(), 1, sig);
        for (int i = 0; i < stensor_size; ++i)
            w += sig[i] * du[static_cast<std::size_t>(border + stensor_size * g + i)];
        out.dissipation = std::fmin(out.dissipation, w);
    }
    const double t = (Sig[0] + Sig[1] + Sig[2]) / 3;
    double d2      = 0;
    for (int i = 0; i < stensor_size; ++i)
        d2 += (Sig[i] - (i < 3 ? t : 0)) * (Sig[i] - (i < 3 ? t : 0));
    out.seq = std::sqrt(1.5 * d2);
    return out;
}

/// \brief Outcome of the host-side checks of a batch, see check_batch.
struct BatchCheck {
    int failures       = 0;        ///< points whose integration failed
    double trace       = 0;        ///< worst |tr Evp| / |Evp|
    double dissipation = HUGE_VAL; ///< smallest grain dissipation of the last step (Pa)
    double reference   = 0;        ///< worst deviation from the reference, sampled points
    double seq         = 0;        ///< largest von Mises macroscopic stress (Pa)
};

/// \brief Serial checks of the last step of a batch, from the states
/// the batch recorded before and after it: every point by check_step,
/// and one point out of `sample` against its whole history integrated
/// again on the host by a reference.
/// \param[in] points       number of integration points
/// \param[in] grains       number of grains
/// \param[in] steps        number of time steps
/// \param[in] dt           time step
/// \param[in] before       states before the last step: entry i of point
///                         p at p * point_stride + i * stride
/// \param[in] after        states after the last step, same layout
/// \param[in] point_stride distance between two points
/// \param[in] stride       distance between two entries of a point
/// \param[in] ok           per-point success flags
/// \param[in] sample       sampling period of the reference
/// \param[in] dense        the reference: the dense LU when true, the
///                         runtime arrowhead solver otherwise
/// \return the outcome
inline BatchCheck check_batch(const int points, const int grains, const int steps, const double dt,
                              const double* before, const double* after,
                              const std::size_t point_stride, const int stride, const int* ok,
                              const int sample, const bool dense) {
    const int n                      = unknowns(grains);
    const tdls::ArrowheadShape shape = shape_of(grains);
    BatchCheck out;
    std::vector<double> b(static_cast<std::size_t>(n)), ref(b.size()), du(b.size()), r(b.size()),
        A(static_cast<std::size_t>(Map::size(shape))),
        work(static_cast<std::size_t>(DynamicSolver::workspace_size(shape)));
    std::vector<int> piv(static_cast<std::size_t>(DynamicSolver::pivot_size(shape)));
    for (int p = 0; p < points; ++p) {
        if (!ok[p]) {
            ++out.failures;
            continue;
        }
        const double* bp = before + p * point_stride;
        const double* ap = after + p * point_stride;
        for (int i = 0; i < n; ++i)
            b[static_cast<std::size_t>(i)] = bp[TDLS_STRIDED(i, stride)];
        const StepCheck step = check_step(grains, b.data(), ap, stride);
        out.trace            = std::fmax(out.trace, step.trace);
        out.dissipation      = std::fmin(out.dissipation, step.dissipation);
        out.seq              = std::fmax(out.seq, step.seq);
        if (p % sample != 0) continue;
        const auto point = static_cast<unsigned long long>(p);
        int iterations = 0, refused = 0;
        bool done = true;
        if (dense) {
            double deto[stensor_size];
            strain_increment(point, deto);
            for (double& v : ref)
                v = 0;
            for (int s = 0; s < steps && done; ++s)
                done = integrate_dense(point, grains, deto, dt, ref.data());
        } else {
            done = history<0, false, false, false, false>(
                point, grains, steps, dt, ref.data(), du.data(), 1, b.data(), 1, A.data(), 1,
                piv.data(), 1, work.data(), 1, r.data(), 1, iterations, refused);
        }
        out.reference =
            done ? std::fmax(out.reference, deviation(ap, stride, ref.data(), n)) : HUGE_VAL;
    }
    return out;
}

} // namespace polycrystal

#endif // TDLS_EXAMPLES_POLYCRYSTAL_HPP
