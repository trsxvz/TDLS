/// \file
/// \brief Example: Iwan multisurface plasticity on a batch of
/// integration points, parallelized with OpenMP on the compile-time
/// ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at compile time: exactly as in the sequential
/// iwan example, the 6 components of the elastic strain and the 24
/// surfaces are constants of the behaviour.
///
/// Why a batch appears: within a time step of a nonlinear finite
/// element solver, every integration point of the mesh integrates the
/// same law independently of every other point. One omp parallel for
/// distributes the points; each iteration keeps its Newton system on its
/// own stack (local residency) and writes its outcome to shared arrays.
/// The self-checks run serially afterwards.
///
/// Each point runs its own shear cycle (iwan.hpp): a deviatoric
/// direction and an amplitude drawn from its index, up to the peak, then
/// down to the opposite strain, in uniform steps that cross several
/// corners of the skeleton at once. The status corrections of the
/// Newton loop sort out the surfaces, and the stress must reach the
/// skeleton at the peak and its opposite at the end.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

#include "iwan.hpp"

int main(int argc, char** argv) {
    using namespace iwan;
    const int points = argc > 1 ? std::atoi(argv[1]) : 8192;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const Material mat = calibrate();

    // Shared per-point outcomes, written once per point by its iteration.
    std::vector<double> tau_peak(points), tau_end(points), off(points);
    std::vector<int> ok(points), iterations(points), refused(points);

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < points; ++i) {
        double A[Solver::storage_size], work[Solver::workspace_size], r[N];
        int it = 0, rf = 0;
        ok[i] = cycle<true, true, true>(mat, static_cast<unsigned long long>(i), A, 1, work, 1, r,
                                        1, tau_peak[i], tau_end[i], off[i], it, rf)
                    ? 1
                    : 0;
        iterations[i] = it;
        refused[i]    = rf;
    }

    // Serial checks over the gathered outcomes.
    int failures               = 0;
    long long total_iterations = 0, total_refused = 0;
    double worst = 0;
    for (int i = 0; i < points; ++i) {
        if (!ok[i]) {
            ++failures;
            continue;
        }
        worst = std::fmax(worst, cycle_deviation(mat, static_cast<unsigned long long>(i),
                                                 tau_peak[i], tau_end[i], off[i]));
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, threads = %d, Newton iterations = %lld, refused blocks = %lld\n",
                points, omp_get_max_threads(), total_iterations, total_refused);
    std::printf("failures = %d, deviation from the skeleton and Masing's rule = %.1e\n", failures,
                worst);
    return failures == 0 && worst < 1e-9 ? 0 : 1;
}
