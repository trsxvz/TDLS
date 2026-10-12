/// \file
/// \brief Example: a Berveiller-Zaoui polycrystal of 12 grains on a
/// batch of integration points, parallelized with OpenMP on the
/// compile-time ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at compile time: exactly as in the sequential
/// polycrystal example, the number of grains is a property of the
/// behaviour, so the 84 x 84 arrowhead jacobian has a compile-time
/// shape.
///
/// Why a batch appears: every integration point of a mesh integrates
/// the same behaviour independently. One omp parallel for distributes
/// the points; each iteration keeps its state, its Newton system and the
/// workspace of the solver on its own stack (local residency), and
/// writes the states before and after the last step to shared arrays,
/// point-major. The self-checks run serially afterwards: every point
/// for the deviatoric viscoplastic strain and the dissipation of every
/// grain, and a sample of points against their whole history integrated
/// again with a dense LU.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

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

    // Shared per-point outputs, point-major.
    std::vector<double> before(static_cast<std::size_t>(n) * points), after(before.size());
    std::vector<int> ok(points), iterations(points), refused(points);

#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < points; ++i) {
        double du[n], r[n], A[Solver::storage_size], work[Solver::workspace_size];
        double* state = &after[static_cast<std::size_t>(i) * n];
        int it = 0, rf = 0;
        // The state is integrated in place in its slice of the shared
        // array: contiguous, so it is passed as a local array.
        ok[i] = history<G, true, true, true, true>(static_cast<unsigned long long>(i), G, steps, dt,
                                                   state, du, 1,
                                                   &before[static_cast<std::size_t>(i) * n], 1, A,
                                                   1, nullptr, 1, work, 1, r, 1, it, rf)
                    ? 1
                    : 0;
        iterations[i] = it;
        refused[i]    = rf;
    }

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check =
        check_batch(points, G, steps, dt, before.data(), after.data(), n, 1, ok.data(), 64, true);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, threads = %d, Newton iterations = %lld, refused blocks = %lld\n",
                points, omp_get_max_threads(), total_iterations, total_refused);
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the dense LU = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
