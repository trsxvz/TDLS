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

/// \brief Row i of the Nystroem system and of its manufactured
/// right-hand side, the unit of assemble: the CooperativeLUpp examples
/// assemble in each thread the rows that it holds in the solver.
/// \param[in]  i        row
/// \param[in]  n        quadrature resolution
/// \param[in]  d        plate separation
/// \param[out] A        matrix, element (i, j) at A[(i * n + j) * A_stride]
/// \param[in]  A_stride element stride of A
/// \param[out] g        manufactured right-hand side, entry i at g[i * g_stride]
/// \param[in]  g_stride element stride of g
TDLS_HOST_DEVICE inline void assemble_row(const int i, const int n, const double d, double* A,
                                          const int A_stride, double* g, const int g_stride) {
    const double h  = 2.0 / (n - 1);
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
    for (int i = 0; i < n; ++i)
        assemble_row(i, n, d, A, A_stride, g, g_stride);
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

/// \brief LU solver of the Nystroem systems for a group of threads, the
/// solver of the CooperativeLUpp examples: the dimension is a runtime
/// value, the rows per thread a compile-time one.
/// \tparam rows_per_thread rows held by each thread of the group
template<int rows_per_thread>
using GroupSolver =
    tdls::CooperativeLUppSolverDynamic<double, tdls::CooperativeLUppConfig<double>{
                                                   .rows_per_thread = rows_per_thread}>;

/// \brief One capacitor instance with a CooperativeLUpp solver, called by
/// every thread of the group that solves it.
///
/// Each thread assembles the rows it holds in the solver, which reads
/// them without a barrier. The group factorizes once and substitutes two
/// right-hand sides: the manufactured one, then the unit potential. The
/// results are visible to every thread on return; a barrier keeps the
/// second substitution from overwriting u before every thread has read
/// it.
/// \tparam Solver a GroupSolver
/// \tparam Sync   callable type of the barrier
/// \param[in]  n          quadrature resolution
/// \param[in]  tx         rank of the thread in its group
/// \param[in]  sync       barrier of the group
/// \param[in]  d          plate separation
/// \param[out] A          matrix, element (i, j) at A[(i * n + j) * A_stride]
/// \param[in]  A_stride   element stride of A
/// \param[out] piv        pivot, entry i at piv[i * piv_stride]
/// \param[in]  piv_stride element stride of piv
/// \param[out] g          right-hand side, entry i at g[i * rhs_stride]
/// \param[out] u          solution, entry i at u[i * rhs_stride]
/// \param[in]  rhs_stride element stride of g and u
/// \param[in,out] work    workspace of the group, Solver::workspace_size(n)
///             elements
/// \param[out] err        deviation of the manufactured solve
/// \param[out] u_mid      unit potential at the center of the plate
/// \return false on a singular matrix, the same verdict in every thread
template<typename Solver, typename Sync>
TDLS_HOST_DEVICE inline bool
capacitor_group(const int n, const int tx, Sync& sync, const double d, double* A,
                const int A_stride, int* piv, const int piv_stride, double* g, double* u,
                const int rhs_stride, double* work, double& err, double& u_mid) {
    const int threads = Solver::threads_per_system(n);
    for (int i = tx; i < n; i += threads)
        assemble_row(i, n, d, A, A_stride, g, rhs_stride);
    const bool ok = Solver::factorize(n, tx, A, A_stride, piv, piv_stride, work, sync);

    Solver::substitute(n, tx, A, A_stride, piv, piv_stride, g, u, rhs_stride, work, sync);
    err = manufactured_error(n, u, rhs_stride);
    sync();

    for (int i = tx; i < n; i += threads)
        g[static_cast<std::size_t>(i) * rhs_stride] = 1.0;
    Solver::substitute(n, tx, A, A_stride, piv, piv_stride, g, u, rhs_stride, work, sync);
    u_mid = u[static_cast<std::size_t>((n - 1) / 2) * rhs_stride];
    return ok;
}

} // namespace love

#endif // TDLS_EXAMPLES_LOVE_HPP
