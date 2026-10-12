/// \file
/// \brief Example: Iwan multisurface plasticity on a batch of
/// integration points on the parallel STL, on the compile-time
/// ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at compile time: exactly as in the other iwan
/// examples, the 6 components of the elastic strain and the 24 surfaces
/// are constants of the behaviour.
///
/// Why the parallel STL: std::for_each with the par_unseq policy over
/// the point indices expresses the batch without naming a backend. The
/// same source runs on the CPU cores (nvc++ -stdpar=multicore, or
/// libstdc++ on oneTBB) and on a GPU (nvc++ -stdpar=gpu), where the
/// lambda becomes a kernel and the vectors it reaches through captured
/// pointers live in managed memory. The material is captured by value.
///
/// The Newton systems and the workspace of the solver are local arrays
/// of the lambda. Each point runs its own shear cycle (iwan.hpp), and
/// the stress must reach the skeleton at the peak and its opposite at
/// the end.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

#include "iwan.hpp"

int main(int argc, char** argv) {
    using namespace iwan;
    const int points = argc > 1 ? std::atoi(argv[1]) : 8192;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const Material mat = calibrate();

    // Per-point outcomes, reached from the lambda through pointers
    // captured by value.
    std::vector<double> tau_peak(points), tau_end(points), off(points);
    std::vector<int> ok(points), iterations(points), refused(points);
    double* tau_peak_p = tau_peak.data();
    double* tau_end_p  = tau_end.data();
    double* off_p      = off.data();
    int* ok_p          = ok.data();
    int* iterations_p  = iterations.data();
    int* refused_p     = refused.data();

    // The point indices, materialized: a random-access range that every
    // parallel STL implementation distributes.
    std::vector<int> ids(points);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int i) {
        double A[Solver::storage_size], work[Solver::workspace_size], r[N];
        double peak = 0, end = 0, o = 0;
        int it = 0, rf = 0;
        ok_p[i] = cycle<true, true, true>(mat, static_cast<unsigned long long>(i), A, 1, work, 1, r,
                                          1, peak, end, o, it, rf)
                      ? 1
                      : 0;
        tau_peak_p[i]   = peak;
        tau_end_p[i]    = end;
        off_p[i]        = o;
        iterations_p[i] = it;
        refused_p[i]    = rf;
    });

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

    std::printf("points = %d, Newton iterations = %lld, refused blocks = %lld\n", points,
                total_iterations, total_refused);
    std::printf("failures = %d, deviation from the skeleton and Masing's rule = %.1e\n", failures,
                worst);
    return failures == 0 && worst < 1e-9 ? 0 : 1;
}
