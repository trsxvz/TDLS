/// \file
/// \brief Example: parameter sweep of Love's integral equation on the
/// parallel STL, one dense Nystroem system per plate separation, on the
/// runtime TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched, so the dimension is a runtime value and
/// TiledLUppSolverDynamic applies.
///
/// Why the parallel STL: std::for_each with the par_unseq policy over
/// the instance indices expresses the sweep without naming a backend.
/// The same source runs on the CPU cores (nvc++ -stdpar=multicore, or
/// libstdc++ on oneTBB) and on a GPU (nvc++ -stdpar=gpu), where the
/// lambda becomes a kernel and the vectors it reaches through captured
/// pointers live in managed memory.
///
/// Each instance assembles its system in arrays local to the lambda and
/// solves at unit stride: the batch of matrices never exists in memory,
/// only the per-instance outputs do. Each instance factorizes once and
/// substitutes two right-hand sides: a manufactured one (exact
/// self-check at solver accuracy) and the physical unit potential.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

#include "love.hpp"

namespace {

constexpr int instances = 20000; ///< plate separations in the sweep
constexpr int max_n     = 40;    ///< bound of the runtime resolution

} // namespace

int main(int argc, char** argv) {
    using namespace love;
    // The quadrature resolution comes from outside the program, bounded
    // here by the local array capacity; the default is odd so that
    // x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 33;
    if (n < 5 || n > max_n) {
        std::printf("usage: %s [quadrature points in [5, %d]]\n", argv[0], max_n);
        return 1;
    }

    // Per-instance outputs, reached from the lambda through pointers
    // captured by value.
    std::vector<double> err(instances), u_mid(instances);
    std::vector<int> ok(instances);
    double* err_p   = err.data();
    double* u_mid_p = u_mid.data();
    int* ok_p       = ok.data();

    // The instance indices, materialized: a random-access range that
    // every parallel STL implementation distributes.
    std::vector<int> ids(instances);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int c) {
        // Local storage, unit strides. The arrays are sized for max_n and
        // filled for n; they are value-initialized because gcc cannot
        // follow that and warns, at a cost negligible next to the
        // assembly.
        double A[max_n * max_n] = {}, g[max_n] = {}, u[max_n] = {};
        int piv[max_n] = {};
        assemble(n, separation(c, instances), A, 1, g, 1);

        // One factorization, two right-hand sides.
        if (!Solver::factorize(n, A, 1, piv, 1)) {
            err_p[c]   = 1.0;
            u_mid_p[c] = 0.0;
            ok_p[c]    = 0;
            return;
        }
        Solver::substitute(n, A, 1, piv, 1, g, u, 1);
        err_p[c] = manufactured_error(n, u, 1);

        for (int i = 0; i < n; ++i)
            g[i] = 1.0;
        Solver::substitute(n, A, 1, piv, 1, g, u, 1);
        u_mid_p[c] = u[(n - 1) / 2];
        ok_p[c]    = 1;
    });

    // Serial self-checks over the gathered outputs.
    int failures = 0;
    double e_max = 0.0, u_lo = 1.0, u_hi = 0.0;
    for (int c = 0; c < instances; ++c) {
        if (!ok[c]) ++failures;
        e_max = std::fmax(e_max, err[c]);
        u_lo  = std::fmin(u_lo, u_mid[c]);
        u_hi  = std::fmax(u_hi, u_mid[c]);
    }

    std::printf("instances = %d, n = %d, manufactured error = %.3e, "
                "u(0): %.6f at d = %.1f -> %.6f at d = %.1f\n",
                instances, n, e_max, u_mid.front(), d_min, u_mid.back(), d_max);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return failures == 0 && e_max < 1e-12 && u_lo > 0.4 && u_hi < 1.0 ? 0 : 1;
}
