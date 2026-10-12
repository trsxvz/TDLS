/// \file
/// \brief Iwan multisurface plasticity integrated the MFront way: the
/// physics of the iwan examples.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The Iwan model describes the cyclic behaviour of soils by Ns nested
/// von Mises surfaces with linear kinematic hardening, in series with
/// isotropic elasticity. Surface i has the radius Y_i, the back stress
/// X_i, the hardening modulus C_i and the plastic multiplier p_i:
///
///   eto = eel + sum_i evp_i,   sig = D : eel,
///   devp_i = dp_i n_i,   n_i = 3/2 dev(sig - X_i) / seq_i,
///   dX_i = C_i dp_i n_i,   seq_i = sigmaeq(sig - X_i) = Y_i while active.
///
/// The integration follows the Iwan behaviour of the MFront gallery
/// (Iwan.mfront, after the first algorithm of Simo and Hughes): the
/// implicit Euler scheme on the unknowns deel (6) and dp_i (Ns = 24),
/// with the residual
///
///   f_eel = deel + sum_{i active} dp_i n_i - deto
///   f_p_i = (seq_i - 3/2 C_i dp_i - Y_i) / E    (surface i active)
///   f_p_i = dp_i                                 (surface i inactive)
///
/// where seq_i and n_i are evaluated with the back stress at the
/// beginning of the step. The status of each surface comes from the
/// elastic prediction, then is corrected at convergence: a surface whose
/// dp_i is negative is deactivated, an inactive surface beyond its
/// radius is activated, and the iterations go on.
///
/// The jacobian is an arrowhead matrix: the border is deel, each surface
/// owns a block of one unknown, dp_i, coupled to every component of
/// f_eel through n_i (the dense coupling). Two surfaces never couple.
/// The block of an active surface is -3/2 C_i / E: on the outer
/// surfaces of a soil, C_i is a small fraction of E while the coupling
/// n_i is of order one, and the partial pivoting of the whole matrix
/// would take its pivots in the border rows. The pivot check of
/// ArrowheadLUpp refuses these blocks, and its second pass eliminates
/// them with the border rows (the refused blocks are counted).
///
/// The surfaces are calibrated on a hyperbolic skeleton curve in simple
/// shear, tau = G0 gamma / (1 + gamma / gamma_ref), sampled at Ns + 1
/// log-spaced shear strains. Surface i yields at point i: in simple
/// shear, sigmaeq = sqrt(3) |tau - X|, so Y_i = sqrt(3) tau_i, and the
/// plastic shear strain of a flowing surface grows by 2 dtau / C_i,
/// which sets C_i from the slopes of the consecutive segments. The
/// elastic shear modulus is the secant to the first point. The model
/// then follows the piecewise linear skeleton through the points
/// exactly under monotonic loading, and Masing's rule exactly under
/// reversals, for any deviatoric direction of a proportional loading.
/// The examples check both.
///
/// Symmetric tensors are 6-vectors in the TFEL convention, as in
/// norton.hpp. A deviatoric unit direction N carries the shear strain
/// gamma as e = gamma / sqrt(2) N and the shear stress tau as
/// dev(sig) = sqrt(2) tau N.

#ifndef TDLS_EXAMPLES_IWAN_HPP
#define TDLS_EXAMPLES_IWAN_HPP

#include <cmath>
#include <cstddef>

#include <tdls/tdls.hpp>

#include "hash01.hpp"

namespace iwan {

constexpr int stensor_size = 6;                       ///< components of a symmetric tensor in 3D
constexpr int surfaces     = 24;                      ///< Ns, the yield surfaces
constexpr int N            = stensor_size + surfaces; ///< unknowns: deel, then dp_i

// Hyperbolic skeleton of a soft soil, in SI units.
constexpr double G0          = 72e6; ///< small-strain shear modulus of the skeleton (Pa)
constexpr double gamma_ref   = 2e-4; ///< reference shear strain: tau tends to G0 gamma_ref
constexpr double nu          = 0.3;  ///< Poisson ratio
constexpr double gamma_first = 1e-6; ///< first calibration point
constexpr double gamma_last  = 1e-1; ///< last calibration point
constexpr double sqrt2       = 1.4142135623730951;
constexpr double sqrt3       = 1.7320508075688772;

constexpr double newton_epsilon = 1e-15; ///< convergence criterion on max |f| (strain units)
constexpr int newton_max_iter   = 100;   ///< iterations, status corrections included

/// The shape: deel the border, one block of one unknown per surface.
constexpr tdls::ArrowheadShape shape{.border = stensor_size, .block = 1, .blocks = surfaces};
/// \brief Configuration of the solver: the dense coupling (the
/// default), the Packed storage.
constexpr tdls::ArrowheadLUppConfig<double> config{.storage = tdls::ArrowheadStorage::Packed};
/// \brief Solver of the Newton systems: the shape is fixed by the
/// modelling hypothesis (3D) and the number of surfaces.
using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
/// Index of every entry of the jacobian in the Packed storage.
using Map = Solver::storage_map;

/// \brief Elasticity and surfaces, from the calibration.
struct Material {
    double mu;                  ///< shear modulus (Pa)
    double lambda;              ///< first Lame coefficient (Pa)
    double young;               ///< Young modulus (Pa)
    double gamma[surfaces + 1]; ///< shear strains of the calibration points
    double tau[surfaces + 1];   ///< shear stresses of the calibration points (Pa)
    double Y[surfaces];         ///< radii of the surfaces (Pa)
    double C[surfaces];         ///< hardening moduli of the surfaces (Pa)
};

/// \brief The hyperbolic skeleton curve.
/// \param[in] gamma shear strain
/// \return the shear stress
TDLS_HOST_DEVICE inline double hyperbola(const double gamma) {
    return G0 * gamma / (1 + gamma / gamma_ref);
}

/// \brief Calibrates the surfaces on the hyperbolic skeleton.
/// \return the material
TDLS_HOST_DEVICE inline Material calibrate() {
    Material mat{};
    const double q = std::pow(gamma_last / gamma_first, 1.0 / surfaces);
    for (int j = 0; j <= surfaces; ++j) {
        mat.gamma[j] = gamma_first * std::pow(q, j);
        mat.tau[j]   = hyperbola(mat.gamma[j]);
    }
    mat.mu          = mat.tau[0] / mat.gamma[0];
    mat.young       = 2 * mat.mu * (1 + nu);
    mat.lambda      = 2 * mat.mu * nu / (1 - 2 * nu);
    double previous = mat.mu; // slope of the previous segment
    for (int i = 0; i < surfaces; ++i) {
        const double slope = (mat.tau[i + 1] - mat.tau[i]) / (mat.gamma[i + 1] - mat.gamma[i]);
        mat.C[i]           = 2 / (1 / slope - 1 / previous);
        mat.Y[i]           = sqrt3 * mat.tau[i];
        previous           = slope;
    }
    return mat;
}

/// \brief The piecewise linear skeleton the model follows: the
/// calibration points joined by segments, the last one extended.
/// \param[in] mat   material
/// \param[in] gamma shear strain, >= 0
/// \return the shear stress
TDLS_HOST_DEVICE inline double skeleton(const Material& mat, const double gamma) {
    if (gamma <= mat.gamma[0]) return mat.mu * gamma;
    int i = 0;
    while (i < surfaces - 1 && gamma > mat.gamma[i + 1])
        ++i;
    const double slope = (mat.tau[i + 1] - mat.tau[i]) / (mat.gamma[i + 1] - mat.gamma[i]);
    return mat.tau[i] + slope * (gamma - mat.gamma[i]);
}

/// \brief Deviatoric unit direction of the loading of a point: simple
/// shear in the xy plane for point 0, a direction drawn from the index
/// for the others.
/// \param[in]  point point index
/// \param[out] dir   the direction, deviatoric, of unit norm
TDLS_HOST_DEVICE inline void direction(const unsigned long long point, double* dir) {
    for (int i = 0; i < stensor_size; ++i)
        dir[i] = point == 0 ? (i == 3 ? 1.0 : 0.0) : 2 * hash01(6 * point + i) - 1;
    const double t = (dir[0] + dir[1] + dir[2]) / 3;
    double norm    = 0;
    for (int i = 0; i < stensor_size; ++i) {
        dir[i] -= i < 3 ? t : 0;
        norm += dir[i] * dir[i];
    }
    for (int i = 0; i < stensor_size; ++i)
        dir[i] /= std::sqrt(norm);
}

/// \brief Isotropic Hooke law, sig = lambda tr(e) Id + 2 mu e.
/// \param[in]  mat material
/// \param[in]  e   strain
/// \param[out] sig stress
TDLS_HOST_DEVICE inline void hooke(const Material& mat, const double* e, double* sig) {
    const double t = mat.lambda * (e[0] + e[1] + e[2]);
    for (int i = 0; i < stensor_size; ++i)
        sig[i] = 2 * mat.mu * e[i] + (i < 3 ? t : 0);
}

/// \brief Relative stress of a surface: its deviatoric part, the
/// equivalent stress and the flow direction.
/// \param[in]  mat  material
/// \param[in]  sig  stress
/// \param[in]  X    back stress of the surface
/// \param[out] n    flow direction 3/2 dev(sig - X) / seq
/// \param[out] iseq 1 / seq, clamped away from zero as in MFront
/// \return seq = sigmaeq(sig - X)
TDLS_HOST_DEVICE inline double relative(const Material& mat, const double* sig, const double* X,
                                        double* n, double& iseq) {
    const double t = (sig[0] + sig[1] + sig[2]) / 3;
    double s2      = 0;
    for (int i = 0; i < stensor_size; ++i) {
        n[i] = sig[i] - (i < 3 ? t : 0) - X[i];
        s2 += n[i] * n[i];
    }
    const double seq = std::sqrt(1.5 * s2);
    iseq             = 1 / std::fmax(seq, 1e-12 * mat.young);
    for (int i = 0; i < stensor_size; ++i)
        n[i] *= 1.5 * iseq;
    return seq;
}

/// \brief Entry i of the residual or of the jacobian storage. The
/// ternary on a template bool folds, exactly as in the solver.
/// \tparam internal whether the array is a local one (stride ignored)
/// \param[in] v      the array
/// \param[in] i      entry
/// \param[in] stride element stride (external mode)
/// \return the entry
template<bool internal>
TDLS_HOST_DEVICE inline double& at(double* v, const int i, const int stride) {
    return v[internal ? static_cast<std::size_t>(i) : TDLS_STRIDED(i, stride)];
}

/// \brief Integrates one time step at one integration point.
///
/// The state (eel, the back stresses) is the caller's local data. The
/// solver operands (jacobian, workspace, residual) are local arrays
/// under the internal residencies, slices of a batch under the external
/// ones.
/// \tparam internal_rhs    residency of r
/// \tparam internal_work   residency of work
/// \tparam internal_matrix residency of A
/// \param[in]     mat         material
/// \param[in]     deto        strain increment of the step
/// \param[in,out] eel         elastic strain, advanced on success
/// \param[in,out] X           back stresses, 6 entries per surface,
///                            advanced on success
/// \param[out]    sig         stress at the end of the step
/// \param[out]    A           jacobian storage, Solver::storage_size
///                            entries
/// \param[in]     A_stride    element stride of A
/// \param[out]    work        workspace, Solver::workspace_size entries
/// \param[in]     work_stride element stride of work
/// \param[out]    r           residual storage, N entries
/// \param[in]     rhs_stride  element stride of r
/// \param[in,out] iterations  incremented by the Newton iterations
/// \param[in,out] refused     incremented by the blocks the pivot check
///                            refused
/// \return false when Newton did not converge or a jacobian was singular
template<bool internal_rhs, bool internal_work, bool internal_matrix>
TDLS_HOST_DEVICE inline bool integrate(const Material& mat, const double* deto, double* eel,
                                       double* X, double* sig, double* A, const int A_stride,
                                       double* work, const int work_stride, double* r,
                                       const int rhs_stride, int& iterations, int& refused) {
    // A local copy of the shape: device code cannot reach an object of
    // namespace scope.
    constexpr tdls::ArrowheadShape sh = shape;
    // Status of the surfaces from the elastic prediction, which is also
    // the first iterate.
    bool active[surfaces];
    double deel[stensor_size], dp[surfaces] = {}, n[stensor_size], e[stensor_size], iseq;
    for (int i = 0; i < stensor_size; ++i) {
        deel[i] = deto[i];
        e[i]    = eel[i] + deel[i];
    }
    hooke(mat, e, sig);
    for (int k = 0; k < surfaces; ++k)
        active[k] = relative(mat, sig, X + stensor_size * k, n, iseq) > mat.Y[k];

    bool converged = false;
    for (int iter = 0; iter < newton_max_iter && !converged; ++iter) {
        for (int i = 0; i < stensor_size; ++i)
            e[i] = eel[i] + deel[i];
        hooke(mat, e, sig);
        // Residual and jacobian. The border rows start from f_eel = deel -
        // deto and Abb = I.
        for (int i = 0; i < stensor_size; ++i) {
            at<internal_rhs>(r, i, rhs_stride) = deel[i] - deto[i];
            for (int j = 0; j < stensor_size; ++j)
                at<internal_matrix>(A, Map::abb(sh, i, j), A_stride) = i == j ? 1.0 : 0.0;
        }
        for (int k = 0; k < surfaces; ++k) {
            const double seq = relative(mat, sig, X + stensor_size * k, n, iseq);
            const double on  = active[k] ? 1.0 : 0.0;
            at<internal_rhs>(r, stensor_size + k, rhs_stride) =
                active[k] ? (seq - 1.5 * mat.C[k] * dp[k] - mat.Y[k]) / mat.young : dp[k];
            at<internal_matrix>(A, Map::d(sh, k, 0, 0), A_stride) =
                active[k] ? -1.5 * mat.C[k] / mat.young : 1.0;
            for (int i = 0; i < stensor_size; ++i) {
                at<internal_rhs>(r, i, rhs_stride) += on * dp[k] * n[i];
                at<internal_matrix>(A, Map::abk(sh, k, i, 0), A_stride) = on * n[i];
                at<internal_matrix>(A, Map::akb(sh, k, 0, i), A_stride) =
                    on * 2 * (mat.mu / mat.young) * n[i];
                // d(dp_k n_k)/d(deel) = 2 mu dp_k / seq_k (3/2 K - n_k (x) n_k)
                for (int j = 0; j < stensor_size; ++j)
                    at<internal_matrix>(A, Map::abb(sh, i, j), A_stride) +=
                        on * 2 * mat.mu * dp[k] * iseq *
                        (1.5 * ((i == j ? 1.0 : 0.0) - (i < 3 && j < 3 ? 1.0 / 3 : 0.0)) -
                         n[i] * n[j]);
            }
        }
        double norm = 0;
        for (int i = 0; i < N; ++i)
            norm = std::fmax(norm, std::fabs(at<internal_rhs>(r, i, rhs_stride)));

        if (norm < newton_epsilon) {
            // Status consistency, as in the MFront gallery: the iterations
            // go on until no surface changes.
            converged = true;
            for (int k = 0; k < surfaces; ++k) {
                if (active[k] && dp[k] < 0) {
                    active[k] = false;
                    converged = false;
                } else if (!active[k] && relative(mat, sig, X + stensor_size * k, n, iseq) >
                                             mat.Y[k] * (1 + 1e-10)) {
                    active[k] = true;
                    converged = false;
                }
            }
            continue;
        }
        int refused_now = 0;
        if (!Solver::solve_inplace<internal_rhs, internal_work, internal_matrix>(
                A, A_stride, work, work_stride, r, rhs_stride, refused_now))
            return false;
        refused += refused_now;
        ++iterations;
        for (int i = 0; i < stensor_size; ++i)
            deel[i] -= at<internal_rhs>(r, i, rhs_stride);
        for (int k = 0; k < surfaces; ++k)
            dp[k] -= at<internal_rhs>(r, stensor_size + k, rhs_stride);
    }
    if (!converged) return false;

    // The back stresses move along the flow directions of the converged
    // stress.
    for (int k = 0; k < surfaces; ++k) {
        if (!active[k]) continue;
        relative(mat, sig, X + stensor_size * k, n, iseq);
        for (int i = 0; i < stensor_size; ++i)
            X[stensor_size * k + i] += mat.C[k] * dp[k] * n[i];
    }
    for (int i = 0; i < stensor_size; ++i)
        eel[i] += deel[i];
    return true;
}

/// Steps from the stress-free state to the peak of the cycle of a batch
/// point; twice as many lead to the opposite strain.
constexpr int load_steps = 16;

/// \brief Shear strain amplitude of the cycle of a batch point,
/// log-uniform between the middle and the last calibration points.
/// \param[in] mat   material
/// \param[in] point point index
/// \return the amplitude
TDLS_HOST_DEVICE inline double amplitude(const Material& mat, const unsigned long long point) {
    const double low = mat.gamma[surfaces / 2], high = mat.gamma[surfaces];
    return low * std::pow(high / low, hash01((1ull << 40) + point));
}

/// \brief Shear stress carried along a direction, tau = sig : N / sqrt(2).
/// \param[in] sig stress
/// \param[in] dir deviatoric unit direction
/// \return the shear stress
TDLS_HOST_DEVICE inline double shear(const double* sig, const double* dir) {
    double s = 0;
    for (int i = 0; i < stensor_size; ++i)
        s += sig[i] * dir[i];
    return s / sqrt2;
}

/// \brief Part of the deviatoric stress off the loading direction,
/// relative to the deviatoric stress: zero under a proportional loading.
/// \param[in] sig stress
/// \param[in] dir deviatoric unit direction
/// \return |dev(sig) - (sig : N) N| / |dev(sig)|
TDLS_HOST_DEVICE inline double off_direction(const double* sig, const double* dir) {
    const double t = (sig[0] + sig[1] + sig[2]) / 3, a = sqrt2 * shear(sig, dir);
    double off = 0, s = 0;
    for (int i = 0; i < stensor_size; ++i) {
        const double d = sig[i] - (i < 3 ? t : 0);
        off += (d - a * dir[i]) * (d - a * dir[i]);
        s += d * d;
    }
    return std::sqrt(off / s);
}

/// \brief The shear cycle of a batch point from the stress-free state:
/// load_steps steps up to its amplitude along its direction, then
/// 2 load_steps steps down to the opposite strain. Under this
/// proportional loading the stress stays along the direction, reaches
/// the skeleton at the peak and, by Masing's rule, its opposite at the
/// end.
/// \tparam internal_rhs    residency of r
/// \tparam internal_work   residency of work
/// \tparam internal_matrix residency of A
/// \param[in]     mat         material
/// \param[in]     point       point index
/// \param[out]    A           jacobian storage
/// \param[in]     A_stride    element stride of A
/// \param[out]    work        workspace
/// \param[in]     work_stride element stride of work
/// \param[out]    r           residual storage
/// \param[in]     rhs_stride  element stride of r
/// \param[out]    tau_peak    shear stress at the peak
/// \param[out]    tau_end     shear stress at the end
/// \param[out]    off         off_direction of the stress at the end
/// \param[in,out] iterations  incremented by the Newton iterations
/// \param[in,out] refused     incremented by the refused blocks
/// \return false when a step failed
template<bool internal_rhs, bool internal_work, bool internal_matrix>
TDLS_HOST_DEVICE inline bool cycle(const Material& mat, const unsigned long long point, double* A,
                                   const int A_stride, double* work, const int work_stride,
                                   double* r, const int rhs_stride, double& tau_peak,
                                   double& tau_end, double& off, int& iterations, int& refused) {
    double dir[stensor_size], eel[stensor_size] = {}, X[stensor_size * surfaces] = {};
    double sig[stensor_size], deto[stensor_size];
    direction(point, dir);
    const double dg = amplitude(mat, point) / load_steps;
    for (int step = 0; step < 3 * load_steps; ++step) {
        for (int i = 0; i < stensor_size; ++i)
            deto[i] = (step < load_steps ? dg : -dg) / sqrt2 * dir[i];
        if (!integrate<internal_rhs, internal_work, internal_matrix>(
                mat, deto, eel, X, sig, A, A_stride, work, work_stride, r, rhs_stride, iterations,
                refused))
            return false;
        if (step == load_steps - 1) tau_peak = shear(sig, dir);
    }
    tau_end = shear(sig, dir);
    off     = off_direction(sig, dir);
    return true;
}

/// \brief Deviation of the cycle of a batch point from the expected
/// curve: the skeleton at the peak, its opposite at the end, and no
/// stress off the direction.
/// \param[in] mat      material
/// \param[in] point    point index
/// \param[in] tau_peak shear stress at the peak
/// \param[in] tau_end  shear stress at the end
/// \param[in] off      off_direction of the stress at the end
/// \return the largest deviation, relative to the peak stress
inline double cycle_deviation(const Material& mat, const unsigned long long point,
                              const double tau_peak, const double tau_end, const double off) {
    const double F = skeleton(mat, amplitude(mat, point));
    if (!std::isfinite(tau_peak) || !std::isfinite(tau_end) || !std::isfinite(off)) return HUGE_VAL;
    return std::fmax(std::fmax(std::fabs(tau_peak - F), std::fabs(tau_end + F)) / F, off);
}

} // namespace iwan

#endif // TDLS_EXAMPLES_IWAN_HPP
