/// \file
/// \brief Example: a Berveiller-Zaoui polycrystal of 12 grains on a
/// batch of integration points on a SYCL device, the Newton systems
/// living in the private memory of each work-item.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at compile time: exactly as in the other
/// polycrystal examples, the number of grains is a property of the
/// behaviour, so the 84 x 84 arrowhead jacobian has a compile-time
/// shape.
///
/// Why SYCL: the kernel is a single-source parallel_for, one work-item
/// per integration point, and the solver headers compile as plain C++
/// inside it. The state, the Newton system and the workspace of the
/// solver are local arrays of the work-item, about 18 KB; device USM
/// holds the states before and after the last step,
/// structure-of-arrays. A machine without a SYCL device reports the
/// test skipped.
///
/// The self-checks run serially on the host: every point for the
/// deviatoric viscoplastic strain and the dissipation of every grain,
/// and a sample of points against their whole history integrated again
/// with a dense LU.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>

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

    std::optional<sycl::queue> queue;
    try {
        queue.emplace(sycl::property_list{sycl::property::queue::in_order{}});
    } catch (const sycl::exception&) {
        std::printf("no device available, skipping\n");
        return 77;
    }
    sycl::queue& q = *queue;

    const std::size_t size = static_cast<std::size_t>(n) * points;
    double* d_before       = sycl::malloc_device<double>(size, q);
    double* d_after        = sycl::malloc_device<double>(size, q);
    int* d_ok              = sycl::malloc_device<int>(points, q);
    int* d_iterations      = sycl::malloc_device<int>(points, q);
    int* d_refused         = sycl::malloc_device<int>(points, q);

    // One integration point per work-item: its loading history with every
    // operand in private memory, then the final state to device USM,
    // entry k of point t at k * points + t.
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(points)), [=](sycl::id<1> idx) {
        const int t = static_cast<int>(idx[0]);
        double state[n], du[n], r[n], A[Solver::storage_size], work[Solver::workspace_size];
        int it = 0, rf = 0;
        const bool success = history<G, true, true, true, true>(
            static_cast<unsigned long long>(t), G, steps, dt, state, du, 1, d_before + t, points, A,
            1, nullptr, 1, work, 1, r, 1, it, rf);
        for (int k = 0; k < n; ++k)
            d_after[static_cast<std::size_t>(k) * points + t] = state[k];
        d_ok[t]         = success ? 1 : 0;
        d_iterations[t] = it;
        d_refused[t]    = rf;
    });

    std::vector<double> before(size), after(size);
    std::vector<int> ok(points), iterations(points), refused(points);
    q.memcpy(before.data(), d_before, sizeof(double) * size);
    q.memcpy(after.data(), d_after, sizeof(double) * size);
    q.memcpy(ok.data(), d_ok, sizeof(int) * points);
    q.memcpy(iterations.data(), d_iterations, sizeof(int) * points);
    q.memcpy(refused.data(), d_refused, sizeof(int) * points);
    q.wait();
    for (void* ptr :
         {static_cast<void*>(d_before), static_cast<void*>(d_after), static_cast<void*>(d_ok),
          static_cast<void*>(d_iterations), static_cast<void*>(d_refused)})
        sycl::free(ptr, q);

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), after.data(), 1,
                                             points, ok.data(), 64, true);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, Newton iterations = %lld, refused blocks = %lld\n", points,
                total_iterations, total_refused);
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the dense LU = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
