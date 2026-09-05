/// \file
/// \brief Love's integral equation discretized by the Nystroem method:
/// the physics of the integral_equation examples.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Love's equation gives the potential of a circular parallel-plate
/// capacitor with plate separation d:
///
///   u(x) + (1/pi) integral of d / (d^2 + (x-y)^2) u(y) dy = g(x)
///
/// The Nystroem method replaces the integral by a quadrature over n
/// points, which couples every point to every other: the linear system
/// is dense by nature, exactly what a dense direct solver is for. The
/// quadrature resolution n is an accuracy versus cost knob chosen when
/// the computation is launched, so the dimension is a runtime value
/// and TiledLUppSolverDynamic applies.
///
/// Every instance factorizes once and substitutes two right-hand
/// sides: a manufactured one (the discrete operator applied to a known
/// solution, which the solve must return to solver accuracy, whatever
/// the resolution) and the physical unit potential.

#ifndef TDLS_EXAMPLES_LOVE_HPP
#define TDLS_EXAMPLES_LOVE_HPP

#include <cmath>
#include <cstddef>
#include <limits>

#include <tdls/tdls.hpp>

namespace love {

/// \brief LU solver of the Nystroem systems.
using Solver = tdls::TiledLUppSolverDynamic<double>;

constexpr double d_min = 0.5; ///< smallest plate separation of a sweep
constexpr double d_max = 2.0; ///< largest plate separation of a sweep

/// \brief Plate separation of instance c of a sweep of `instances`
/// values, on the ramp from d_min to d_max.
/// \param[in] c         instance index
/// \param[in] instances number of instances of the sweep
/// \return the plate separation
TDLS_HOST_DEVICE inline double separation(const int c, const int instances) {
    return d_min + (d_max - d_min) * c / (instances - 1);
}

/// \brief Love kernel of the parallel-plate capacitor.
/// \param[in] x first quadrature point
/// \param[in] y second quadrature point
/// \param[in] d plate separation
/// \return the kernel value
TDLS_HOST_DEVICE inline double kernel(const double x, const double y, const double d) {
    return d / ((d * d + (x - y) * (x - y)) * 3.14159265358979324);
}

/// \brief The manufactured solution used to check every solve.
/// \param[in] x quadrature point
/// \return the manufactured value
TDLS_HOST_DEVICE inline double manufactured(const double x) {
    return std::exp(x);
}

/// \brief Quadrature node i of the n-point trapezoid rule on [-1, 1].
/// \param[in] i node index
/// \param[in] n quadrature resolution
/// \return the abscissa
TDLS_HOST_DEVICE inline double node(const int i, const int n) {
    return -1.0 + i * (2.0 / (n - 1));
}

/// \brief Nystroem system on [-1, 1] with the trapezoid rule: the
/// identity plus the discretized integral operator. The manufactured
/// right-hand side is the discrete operator applied to the known
/// solution, accumulated during the assembly.
/// \param[in]  n        quadrature resolution
/// \param[in]  d        plate separation
/// \param[out] A        matrix, element (i, j) at A[(i * n + j) * A_stride]
/// \param[in]  A_stride element stride of A
/// \param[out] g        manufactured right-hand side, entry i at g[i * g_stride]
/// \param[in]  g_stride element stride of g
TDLS_HOST_DEVICE inline void assemble(const int n, const double d, double* A, const int A_stride,
                                      double* g, const int g_stride) {
    const double h = 2.0 / (n - 1);
    for (int i = 0; i < n; ++i) {
        const double xi = node(i, n);
        double acc      = 0.0;
        for (int j = 0; j < n; ++j) {
            const double yj = node(j, n);
            const double wj = (j == 0 || j == n - 1) ? h / 2 : h;
            const double a  = (i == j ? 1.0 : 0.0) + wj * kernel(xi, yj, d);
            A[static_cast<std::size_t>(i * n + j) * A_stride] = a;
            acc += a * manufactured(yj);
        }
        g[static_cast<std::size_t>(i) * g_stride] = acc;
    }
}

/// \brief Largest deviation of a solution from the manufactured one.
/// \param[in] n      quadrature resolution
/// \param[in] u      solution, entry i at u[i * stride]
/// \param[in] stride element stride of u
/// \return the deviation
TDLS_HOST_DEVICE inline double manufactured_error(const int n, const double* u, const int stride) {
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = u[static_cast<std::size_t>(i) * stride];
        if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
        e = std::fmax(e, std::fabs(v - manufactured(node(i, n))));
    }
    return e;
}

} // namespace love

#endif // TDLS_EXAMPLES_LOVE_HPP
