/// \file
/// \brief Example: parameter sweep of Love's integral equation, one
/// dense Nystroem system per plate separation, parallelized with OpenMP
/// on the runtime TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Companion of integral_equation.cpp at the next scale: the capacitor
/// potential is computed for a whole sweep of plate separations d, the
/// canonical embarrassingly parallel workload of parametric studies.
/// Every instance assembles and solves its own dense n x n Nystroem
/// system; one omp parallel for distributes the instances and each
/// thread reuses its own buffers. Every instance writes its outputs
/// (manufactured error, central potential, success flag) into shared
/// per-instance slots (the exact outputs of the GPU companion
/// examples) and the self-checks run serially afterwards; the pragmas
/// thus stay plain OpenMP 2.0 (no min/max reduction, which the default
/// MSVC OpenMP runtime rejects).
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched (here the command line), so the dimension is a runtime
/// value and TiledLUppSolverDynamic applies.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential, giving the curve of the potential against
/// the plate separation.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

#include "love.hpp"

int main(int argc, char** argv) {
    using namespace love;
    // The quadrature resolution comes from outside the program; the
    // default is odd so that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 41;
    if (n < 5 || n > 100) {
        std::printf("usage: %s [quadrature points in [5, 100]]\n", argv[0]);
        return 1;
    }
    constexpr int instances = 20000; // plate separations in the sweep
    const int mid           = (n - 1) / 2;

    // Shared per-instance outputs, written once per instance by its
    // owning iteration.
    std::vector<double> err(instances), u_mid(instances);
    std::vector<int> ok(instances);

#pragma omp parallel
    {
        // Thread-local buffers, reused across the instances of the
        // thread.
        std::vector<double> A(static_cast<std::size_t>(n) * n), g(n), u(n);
        std::vector<int> piv(n);

#pragma omp for schedule(static)
        for (int c = 0; c < instances; ++c) {
            assemble(n, separation(c, instances), A.data(), 1, g.data(), 1);

            // One factorization, two right-hand sides.
            if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) {
                err[c]   = 1.0;
                u_mid[c] = 0.0;
                ok[c]    = 0;
                continue;
            }
            Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
            err[c] = manufactured_error(n, u.data(), 1);

            std::fill(g.begin(), g.end(), 1.0);
            Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
            u_mid[c] = u[mid];
            ok[c]    = 1;
        }
    }

    // Serial self-checks over the gathered outputs, as in the GPU
    // examples.
    int failures = 0;
    double e_max = 0.0, u_lo = 1.0, u_hi = 0.0;
    for (int c = 0; c < instances; ++c) {
        if (!ok[c]) ++failures;
        e_max = std::fmax(e_max, err[c]);
        u_lo  = std::fmin(u_lo, u_mid[c]);
        u_hi  = std::fmax(u_hi, u_mid[c]);
    }

    std::printf("instances = %d, n = %d, threads = %d, manufactured error = %.3e, "
                "u(0): %.6f at d = %.1f -> %.6f at d = %.1f\n",
                instances, n, omp_get_max_threads(), e_max, u_mid.front(), d_min, u_mid.back(),
                d_max);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return failures == 0 && e_max < 1e-12 && u_lo > 0.4 && u_hi < 1.0 ? 0 : 1;
}
