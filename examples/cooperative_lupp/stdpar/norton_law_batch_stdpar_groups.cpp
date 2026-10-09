/// \file
/// \brief Example: Norton viscoplasticity on a batch of integration
/// points on the parallel STL offloaded to an NVIDIA GPU by nvc++, each
/// Newton system shared by a group of iterations.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why groups of iterations: nvc++ runs every iteration of std::for_each
/// on a GPU lane of its own, iteration i on lane i % 32. Laid out warp by
/// warp, the iterations of a group of lanes solve one point, as in the
/// CUDA example: four rows per lane, two lanes per point, sixteen points
/// per warp. Each lane then holds four rows of the jacobian instead of
/// seven, the registers of one thread per point being the limit of the
/// other parallel STL examples.
///
/// Why this example is specific to nvc++ on a GPU: the C++ standard
/// does not let the iterations of a parallel algorithm wait for each
/// other, and nothing documents how nvc++ maps them to lanes. The
/// barrier deduced by the solver makes the layout safe: on entry, the
/// solver checks that the iterations of each group run together in
/// their warp, and stops the program with a message otherwise, instead
/// of a wrong result or a deadlock. The build only offers this target
/// with nvc++ and its device flags; the other parallel STL examples keep
/// one iteration per system, portable to every implementation.
///
/// The Newton systems are local arrays of the lambda, so each group
/// solves in the registers of its lanes. Unlike the CUDA example, the
/// parallel STL offers no shared memory: the workspaces live in a
/// vector, zero-initialized, as the per-point outputs, structure-of-
/// arrays.

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
    using Solver          = GroupSolver<4, true>; // two lanes per point
    constexpr int threads = Solver::threads_per_system;
    constexpr int groups  = 32 / threads; // groups per 32 lanes

    const int points = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const double dt = 1.0; // time step (s)
    const int steps = 10;  // integrate each point to t = 10 s

    // Per-point outputs and workspaces, structure-of-arrays, reached from
    // the lambda through pointers captured by value.
    std::vector<double> eel_before(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> sig(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> Dt(static_cast<std::size_t>(stensor_size) * stensor_size * points);
    std::vector<double> p_before(points), p(points);
    std::vector<double> workspaces(static_cast<std::size_t>(Solver::workspace_size) * points);
    std::vector<int> ok(points);
    double* eel_before_p = eel_before.data();
    double* sig_p        = sig.data();
    double* Dt_p         = Dt.data();
    double* p_before_p   = p_before.data();
    double* p_p          = p.data();
    double* workspaces_p = workspaces.data();
    int* ok_p            = ok.data();

    // The iterations, 32 per warp: group g of the warp takes lanes
    // g * threads to (g + 1) * threads - 1.
    std::vector<int> ids((points + groups - 1) / groups * 32);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int i) {
        const int lane  = i % 32;
        const int group = lane / threads;
        const int tx    = lane % threads;
        const int t     = i / 32 * groups + group;
        // The lanes past the last group, and the groups past the batch,
        // leave at once: a barrier only involves the lanes of its group.
        if (group >= groups || t >= points) return;
        double* work = workspaces_p + static_cast<std::size_t>(t) * Solver::workspace_size;
        auto sync    = Solver::make_sync(work);
        auto any     = [](const bool flag) { return flag; };

        double eel[stensor_size] = {};
        double p_t               = 0;
        double sig_t[stensor_size], Dt_t[stensor_size * stensor_size] = {};
        double J[Solver::rows_per_thread * N], r[Solver::rows_per_thread];
        int piv[Solver::rows_per_thread];
        double deto[stensor_size];
        strain_increment(static_cast<unsigned long long>(t), deto);

        bool success = true;
        for (int s = 0; s < steps && success; ++s) {
            if (s == steps - 1 && tx == 0) {
                for (int k = 0; k < stensor_size; ++k)
                    eel_before_p[static_cast<std::size_t>(k) * points + t] = eel[k];
                p_before_p[t] = p_t;
            }
            success = integrate_group<true, true, true, Solver>(
                tx, sync, any, deto, dt, eel, p_t, sig_t, Dt_t, J, 1, piv, 1, r, 1, work);
        }

        // Every lane of the group holds the same results.
        if (tx != 0) return;
        for (int k = 0; k < stensor_size; ++k)
            sig_p[static_cast<std::size_t>(k) * points + t] = sig_t[k];
        for (int k = 0; k < stensor_size * stensor_size; ++k)
            Dt_p[static_cast<std::size_t>(k) * points + t] = Dt_t[k];
        p_p[t]  = p_t;
        ok_p[t] = success ? 1 : 0;
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
