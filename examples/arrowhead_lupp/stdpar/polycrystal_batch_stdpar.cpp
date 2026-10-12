/// \file
/// \brief Example: a Berveiller-Zaoui polycrystal of 12 grains on a
/// batch of integration points on the parallel STL, on the compile-time
/// ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at compile time: exactly as in the other
/// polycrystal examples, the number of grains is a property of the
/// behaviour, so the 84 x 84 arrowhead jacobian has a compile-time
/// shape.
///
/// Why the parallel STL: std::for_each with the par_unseq policy over
/// the point indices expresses the batch without naming a backend. The
/// same source runs on the CPU cores (nvc++ -stdpar=multicore, or
/// libstdc++ on oneTBB) and on a GPU (nvc++ -stdpar=gpu), where the
/// lambda becomes a kernel and the vectors it reaches through captured
/// pointers live in managed memory.
///
/// The state, the Newton system and the workspace of the solver are
/// local arrays of the lambda, about 18 KB per point. The states before
/// and after the last step go to the vectors, structure-of-arrays. The
/// self-checks run serially afterwards: every point for the deviatoric
/// viscoplastic strain and the dissipation of every grain, and a sample
/// of points against their whole history integrated again with a dense
/// LU.

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
    constexpr int G  = fixed_grains;
    constexpr int n  = unknowns(G);
    using Solver     = StaticSolver<G>;
    const int points = argc > 1 ? std::atoi(argv[1]) : 2048;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const double dt = 1.0; // time step (s)
    const int steps = 20;  // integrate each point to t = 20 s

    // Per-point outputs, structure-of-arrays: entry k of point i at
    // k * points + i.
    std::vector<double> before(static_cast<std::size_t>(n) * points), after(before.size());
    std::vector<int> ok(points), iterations(points), refused(points);
    double* before_p  = before.data();
    double* after_p   = after.data();
    int* ok_p         = ok.data();
    int* iterations_p = iterations.data();
    int* refused_p    = refused.data();

    std::vector<int> ids(points);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int i) {
        double state[n], du[n], r[n], A[Solver::storage_size], work[Solver::workspace_size];
        int it = 0, rf = 0;
        const bool success = history<G, true, true, true, true>(
            static_cast<unsigned long long>(i), G, steps, dt, state, du, 1, before_p + i, points, A,
            1, nullptr, 1, work, 1, r, 1, it, rf);
        for (int k = 0; k < n; ++k)
            after_p[static_cast<std::size_t>(k) * points + i] = state[k];
        ok_p[i]         = success ? 1 : 0;
        iterations_p[i] = it;
        refused_p[i]    = rf;
    });

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), after.data(), 1,
                                             points, ok.data(), 64, true);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, Newton iterations = %lld, refused blocks = %lld\n", points,
                total_iterations, total_refused);
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the dense LU = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
