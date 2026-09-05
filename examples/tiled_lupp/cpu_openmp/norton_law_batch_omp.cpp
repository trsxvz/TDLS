/// \file
/// \brief Example: Norton viscoplasticity on a batch of integration
/// points, parallelized with OpenMP on the compile-time TiledLUpp
/// solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: exactly as in the
/// sequential norton_law example, the unknowns are the elastic strain
/// increment and the viscoplastic multiplier, N = 7 in 3D, a property
/// of the law and of the modelling hypothesis, not of the data.
///
/// Why a batch appears: within a time step of a nonlinear finite
/// element solver, every integration point of the mesh integrates the
/// same law independently of every other point. The constitutive-law
/// evaluation is embarrassingly parallel, and it is the workload TDLS
/// is written for. One omp parallel for distributes the points; each
/// iteration keeps its Newton system on its own stack (local residency)
/// and writes the per-point outputs to shared arrays. The loop only
/// computes and the self-checks run serially afterwards, exactly as in
/// the GPU examples; the pragma thus stays plain OpenMP 2.0.
///
/// Each point carries its own loading amplitude, so the batch spans the
/// elastic and the creep regimes. The state before the last step is
/// recorded, so that the last step can be checked on every point
/// against the radial return, and on a sample of points against central
/// differences of the tangent operator.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

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

    // Shared per-point outputs, point-major, written once per point by
    // its owning iteration.
    std::vector<double> eel_before(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> sig(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> Dt(static_cast<std::size_t>(stensor_size) * stensor_size * points);
    std::vector<double> p_before(points), p(points);
    std::vector<int> ok(points);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < points; ++i) {
        double eel[stensor_size] = {};
        double p_i               = 0;
        double sig_i[stensor_size], Dt_i[stensor_size * stensor_size] = {};
        double J[N * N], r[N];
        int piv[N];
        double deto[stensor_size];
        strain_increment(static_cast<unsigned long long>(i), deto);

        bool success = true;
        for (int s = 0; s < steps && success; ++s) {
            if (s == steps - 1) {
                for (int k = 0; k < stensor_size; ++k)
                    eel_before[static_cast<std::size_t>(i) * stensor_size + k] = eel[k];
                p_before[i] = p_i;
            }
            success =
                integrate<true, true, true>(deto, dt, eel, p_i, sig_i, Dt_i, J, 1, piv, 1, r, 1);
        }

        for (int k = 0; k < stensor_size; ++k)
            sig[static_cast<std::size_t>(i) * stensor_size + k] = sig_i[k];
        for (int k = 0; k < stensor_size * stensor_size; ++k)
            Dt[static_cast<std::size_t>(i) * stensor_size * stensor_size + k] = Dt_i[k];
        p[i]  = p_i;
        ok[i] = success ? 1 : 0;
    }

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check = check_last_step(points, dt, false, eel_before.data(), p_before.data(),
                                             sig.data(), p.data(), Dt.data(), ok.data(), 1024);

    std::printf("points = %d, threads = %d, stress deviation from the radial return = %.1e, "
                "tangent deviation = %.1e\n",
                points, omp_get_max_threads(), check.sig_error, check.tangent);
    return check.failures == 0 && check.sig_error < 1e-9 && check.tangent < 1e-5 &&
                   check.p_monotonic
               ? 0
               : 1;
}
