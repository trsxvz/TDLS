/// \file
/// \brief Example: Norton viscoplasticity on a batch of integration
/// points on the parallel STL, on the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: exactly as in the other
/// norton_law examples, N = 7 is fixed by the law and by the modelling
/// hypothesis, the shape of the systems MFront-generated behaviours
/// hand to TDLS.
///
/// Why the parallel STL: std::for_each with the par_unseq policy over
/// the point indices expresses the batch without naming a backend. The
/// same source runs on the CPU cores (nvc++ -stdpar=multicore, or
/// libstdc++ on oneTBB) and on a GPU (nvc++ -stdpar=gpu), where the
/// lambda becomes a kernel and the vectors it reaches through captured
/// pointers live in managed memory.
///
/// The Newton systems are local arrays of the lambda, so each point
/// solves in registers on a device; the per-point outputs live in the
/// vectors, structure-of-arrays. The state before the last step is
/// recorded, so that the last step can be checked on every point
/// against the radial return, and on a sample of points against central
/// differences of the tangent operator.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

#include "norton.hpp"

int main(int argc, char** argv) {
    using namespace norton;
    const int points = argc > 1 ? std::atoi(argv[1]) : 65536;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const double dt = 1.0; // time step (s)
    const int steps = 10;  // integrate each point to t = 10 s

    // Per-point outputs, structure-of-arrays, reached from the lambda
    // through pointers captured by value.
    std::vector<double> eel_before(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> sig(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> Dt(static_cast<std::size_t>(stensor_size) * stensor_size * points);
    std::vector<double> p_before(points), p(points);
    std::vector<int> ok(points);
    double* eel_before_p = eel_before.data();
    double* sig_p        = sig.data();
    double* Dt_p         = Dt.data();
    double* p_before_p   = p_before.data();
    double* p_p          = p.data();
    int* ok_p            = ok.data();

    // The point indices, materialized: a random-access range that every
    // parallel STL implementation distributes.
    std::vector<int> ids(points);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int i) {
        double eel[stensor_size] = {};
        double p_i               = 0;
        double sig_i[stensor_size], Dt_i[stensor_size * stensor_size];
        double J[N * N], r[N];
        int piv[N];
        double deto[stensor_size];
        strain_increment(static_cast<unsigned long long>(i), deto);

        bool success = true;
        for (int s = 0; s < steps && success; ++s) {
            if (s == steps - 1) {
                for (int k = 0; k < stensor_size; ++k)
                    eel_before_p[static_cast<std::size_t>(k) * points + i] = eel[k];
                p_before_p[i] = p_i;
            }
            success =
                integrate<true, true, true>(deto, dt, eel, p_i, sig_i, Dt_i, J, 1, piv, 1, r, 1);
        }

        for (int k = 0; k < stensor_size; ++k)
            sig_p[static_cast<std::size_t>(k) * points + i] = sig_i[k];
        for (int k = 0; k < stensor_size * stensor_size; ++k)
            Dt_p[static_cast<std::size_t>(k) * points + i] = Dt_i[k];
        p_p[i]  = p_i;
        ok_p[i] = success ? 1 : 0;
    });

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check = check_last_step(points, dt, true, eel_before.data(), p_before.data(),
                                             sig.data(), p.data(), Dt.data(), ok.data(), 1024);

    std::printf("points = %d, stress deviation from the radial return = %.1e, "
                "tangent deviation = %.1e\n",
                points, check.sig_error, check.tangent);
    return check.failures == 0 && check.sig_error < 1e-9 && check.tangent < 1e-5 &&
                   check.p_monotonic
               ? 0
               : 1;
}
