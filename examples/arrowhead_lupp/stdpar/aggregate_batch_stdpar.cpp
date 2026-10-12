/// \file
/// \brief Example: Berveiller-Zaoui aggregates whose number of grains is
/// given at run time, on a batch of integration points on the parallel
/// STL, on the runtime ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at run time: exactly as in the other aggregate
/// examples, the number of grains is a datum of the computation (240 by
/// default, 1452 unknowns), so ArrowheadLUppSolverDynamic takes the
/// shape as its first argument.
///
/// Why the parallel STL: std::for_each with the par_unseq policy over
/// the point indices expresses the batch without naming a backend. The
/// same source runs on the CPU cores and on a GPU (nvc++ -stdpar=gpu),
/// where the vectors live in managed memory.
///
/// A lambda has no local array of runtime size: every operand of a
/// point (state, increments, jacobian, workspace, pivots, residual) is
/// its slice of a batch vector, structure-of-arrays, entry k of point i
/// at k * points + i. The solver walks each slice with the stride
/// points, so that consecutive points touch consecutive addresses. The
/// self-checks run serially afterwards: every point for the deviatoric
/// viscoplastic strain and the dissipation of every grain, and a sample
/// of points against their whole history integrated again on the host.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

#include "polycrystal.hpp"

int main(int argc, char** argv) {
    using namespace polycrystal;
    const int points = argc > 1 ? std::atoi(argv[1]) : 64;
    const int G      = argc > 2 ? std::atoi(argv[2]) : 240;
    if (points < 1 || G < 1) {
        std::printf("usage: %s [number of integration points >= 1] [number of grains >= 1]\n",
                    argv[0]);
        return 1;
    }
    const tdls::ArrowheadShape shape = shape_of(G);
    const int n                      = unknowns(G);
    const double dt                  = 1.0; // time step (s)
    const int steps                  = 20;  // integrate each point to t = 20 s

    // The batch, structure-of-arrays: one vector per operand, sized from
    // the shape.
    const auto P = static_cast<std::size_t>(points);
    std::vector<double> state(static_cast<std::size_t>(n) * P), before(state.size()),
        du(state.size()), r(state.size()), A(static_cast<std::size_t>(Map::size(shape)) * P),
        work(static_cast<std::size_t>(DynamicSolver::workspace_size(shape)) * P);
    std::vector<int> piv(static_cast<std::size_t>(DynamicSolver::pivot_size(shape)) * P);
    std::vector<int> ok(points), iterations(points), refused(points);
    double* state_p   = state.data();
    double* before_p  = before.data();
    double* du_p      = du.data();
    double* r_p       = r.data();
    double* A_p       = A.data();
    double* work_p    = work.data();
    int* piv_p        = piv.data();
    int* ok_p         = ok.data();
    int* iterations_p = iterations.data();
    int* refused_p    = refused.data();

    std::vector<int> ids(points);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int i) {
        int it = 0, rf = 0;
        // Grains = 0: the runtime solver. Every operand is external, the
        // slice of point i walked with the stride points.
        const bool success = history<0, false, false, false, false>(
            static_cast<unsigned long long>(i), G, steps, dt, state_p + i, du_p + i, points,
            before_p + i, points, A_p + i, points, piv_p + i, points, work_p + i, points, r_p + i,
            points, it, rf);
        ok_p[i]         = success ? 1 : 0;
        iterations_p[i] = it;
        refused_p[i]    = rf;
    });

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), state.data(), 1,
                                             points, ok.data(), 16, false);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, grains = %d, Newton iterations = %lld, refused blocks = %lld\n",
                points, G, total_iterations, total_refused);
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the host = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
