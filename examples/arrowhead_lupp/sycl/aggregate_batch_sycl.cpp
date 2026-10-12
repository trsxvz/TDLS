/// \file
/// \brief Example: Berveiller-Zaoui aggregates whose number of grains is
/// given at run time, on a batch of integration points on a SYCL device,
/// every operand in device USM.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at run time: exactly as in the other aggregate
/// examples, the number of grains is a datum of the computation (240 by
/// default, 1452 unknowns), so ArrowheadLUppSolverDynamic takes the
/// shape as its first argument.
///
/// Why SYCL: the kernel is a single-source parallel_for, one work-item
/// per integration point. A work-item has no local array of runtime
/// size: every operand of a point (state, increments, jacobian,
/// workspace, pivots, residual) is its slice of a device USM batch,
/// structure-of-arrays, entry k of point t at k * points + t, walked by
/// the solver with the stride points. A machine without a SYCL device
/// reports the test skipped.
///
/// The self-checks run serially on the host: every point for the
/// deviatoric viscoplastic strain and the dissipation of every grain,
/// and a sample of points against their whole history integrated again
/// on the host.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>

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

    std::optional<sycl::queue> queue;
    try {
        queue.emplace(sycl::property_list{sycl::property::queue::in_order{}});
    } catch (const sycl::exception&) {
        std::printf("no device available, skipping\n");
        return 77;
    }
    sycl::queue& q = *queue;

    // The batch, structure-of-arrays: one allocation per operand, sized
    // from the shape.
    const auto P           = static_cast<std::size_t>(points);
    const std::size_t size = static_cast<std::size_t>(n) * P;
    double* d_state        = sycl::malloc_device<double>(size, q);
    double* d_before       = sycl::malloc_device<double>(size, q);
    double* d_du           = sycl::malloc_device<double>(size, q);
    double* d_r            = sycl::malloc_device<double>(size, q);
    double* d_A    = sycl::malloc_device<double>(static_cast<std::size_t>(Map::size(shape)) * P, q);
    double* d_work = sycl::malloc_device<double>(
        static_cast<std::size_t>(DynamicSolver::workspace_size(shape)) * P, q);
    int* d_piv =
        sycl::malloc_device<int>(static_cast<std::size_t>(DynamicSolver::pivot_size(shape)) * P, q);
    int* d_ok         = sycl::malloc_device<int>(points, q);
    int* d_iterations = sycl::malloc_device<int>(points, q);
    int* d_refused    = sycl::malloc_device<int>(points, q);

    // One integration point per work-item. Grains = 0: the runtime
    // solver, every operand external, walked with the stride points.
    q.parallel_for(sycl::range<1>(P), [=](sycl::id<1> idx) {
        const int t = static_cast<int>(idx[0]);
        int it = 0, rf = 0;
        const bool success = history<0, false, false, false, false>(
            static_cast<unsigned long long>(t), G, steps, dt, d_state + t, d_du + t, points,
            d_before + t, points, d_A + t, points, d_piv + t, points, d_work + t, points, d_r + t,
            points, it, rf);
        d_ok[t]         = success ? 1 : 0;
        d_iterations[t] = it;
        d_refused[t]    = rf;
    });

    std::vector<double> before(size), after(size);
    std::vector<int> ok(points), iterations(points), refused(points);
    q.memcpy(before.data(), d_before, sizeof(double) * size);
    q.memcpy(after.data(), d_state, sizeof(double) * size);
    q.memcpy(ok.data(), d_ok, sizeof(int) * points);
    q.memcpy(iterations.data(), d_iterations, sizeof(int) * points);
    q.memcpy(refused.data(), d_refused, sizeof(int) * points);
    q.wait();
    for (void* ptr :
         {static_cast<void*>(d_state), static_cast<void*>(d_before), static_cast<void*>(d_du),
          static_cast<void*>(d_r), static_cast<void*>(d_A), static_cast<void*>(d_work),
          static_cast<void*>(d_piv), static_cast<void*>(d_ok), static_cast<void*>(d_iterations),
          static_cast<void*>(d_refused)})
        sycl::free(ptr, q);

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), after.data(), 1,
                                             points, ok.data(), 16, false);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, grains = %d, Newton iterations = %lld, refused blocks = %lld\n",
                points, G, total_iterations, total_refused);
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the host = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
