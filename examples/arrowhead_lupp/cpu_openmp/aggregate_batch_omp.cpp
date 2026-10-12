/// \file
/// \brief Example: Berveiller-Zaoui aggregates whose number of grains is
/// given at run time, on a batch of integration points, parallelized
/// with OpenMP on the runtime ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at run time: exactly as in the sequential
/// aggregate example, the number of grains is a datum of the
/// computation (240 by default, 1452 unknowns), so
/// ArrowheadLUppSolverDynamic takes the shape as its first argument.
///
/// Why a batch appears: every integration point of a mesh integrates
/// the same behaviour independently. The jacobian of a point holds
/// 27504 entries in the Packed storage, too many for the stack: each
/// thread allocates its operands once, sized from the shape, and reuses
/// them for every point it integrates. The states before and after the
/// last step go to shared arrays, point-major. The self-checks run
/// serially afterwards: every point for the deviatoric viscoplastic
/// strain and the dissipation of every grain, and a sample of points
/// against their whole history integrated again on the host.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

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

    // Shared per-point outputs, point-major.
    std::vector<double> before(static_cast<std::size_t>(n) * points), after(before.size());
    std::vector<int> ok(points), iterations(points), refused(points);

#pragma omp parallel
    {
        // The operands of the thread, sized from the shape.
        std::vector<double> du(static_cast<std::size_t>(n)), r(du.size()),
            A(static_cast<std::size_t>(Map::size(shape))),
            work(static_cast<std::size_t>(DynamicSolver::workspace_size(shape)));
        std::vector<int> piv(static_cast<std::size_t>(DynamicSolver::pivot_size(shape)));
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < points; ++i) {
            int it = 0, rf = 0;
            // Grains = 0: the runtime solver, every operand with a stride
            // of 1.
            ok[i] = history<0, false, false, false, false>(
                        static_cast<unsigned long long>(i), G, steps, dt,
                        &after[static_cast<std::size_t>(i) * n], du.data(), 1,
                        &before[static_cast<std::size_t>(i) * n], 1, A.data(), 1, piv.data(), 1,
                        work.data(), 1, r.data(), 1, it, rf)
                        ? 1
                        : 0;
            iterations[i] = it;
            refused[i]    = rf;
        }
    }

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), after.data(),
                                             static_cast<std::size_t>(n), 1, ok.data(), 16, false);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, grains = %d, threads = %d, Newton iterations = %lld, "
                "refused blocks = %lld\n",
                points, G, omp_get_max_threads(), total_iterations, total_refused);
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the host = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
