/// \file
/// \brief Example: Iwan multisurface plasticity on a batch of
/// integration points on a SYCL device, the Newton systems living in the
/// private memory of each work-item.
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
/// Why SYCL: the kernel is a single-source parallel_for, one work-item
/// per integration point, and the solver headers compile as plain C++
/// inside it. The jacobian, the workspace of the solver and the residual
/// are local arrays of the work-item; the material is captured by value,
/// and device USM holds the per-point outcomes. The batch runs on
/// whatever device the default selector picks. A machine without a SYCL
/// device reports the test skipped.
///
/// Each point runs its own shear cycle (iwan.hpp), and the stress must
/// reach the skeleton at the peak and its opposite at the end.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>

#include "iwan.hpp"

int main(int argc, char** argv) {
    using namespace iwan;
    const int points = argc > 1 ? std::atoi(argv[1]) : 8192;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const Material mat = calibrate();

    // A machine without a SYCL runtime exposes no device: the test
    // reports itself skipped, as the CUDA and HIP examples do. The queue
    // is in order, so the readbacks see the kernel.
    std::optional<sycl::queue> queue;
    try {
        queue.emplace(sycl::property_list{sycl::property::queue::in_order{}});
    } catch (const sycl::exception&) {
        std::printf("no device available, skipping\n");
        return 77;
    }
    sycl::queue& q = *queue;

    double* d_tau_peak = sycl::malloc_device<double>(points, q);
    double* d_tau_end  = sycl::malloc_device<double>(points, q);
    double* d_off      = sycl::malloc_device<double>(points, q);
    int* d_ok          = sycl::malloc_device<int>(points, q);
    int* d_iterations  = sycl::malloc_device<int>(points, q);
    int* d_refused     = sycl::malloc_device<int>(points, q);

    // One integration point per work-item: its shear cycle with the
    // Newton systems in private memory.
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(points)), [=](sycl::id<1> idx) {
        const int t = static_cast<int>(idx[0]);
        double A[Solver::storage_size], work[Solver::workspace_size], r[N];
        double peak = 0, end = 0, o = 0;
        int it = 0, rf = 0;
        d_ok[t] = cycle<true, true, true>(mat, static_cast<unsigned long long>(t), A, 1, work, 1, r,
                                          1, peak, end, o, it, rf)
                      ? 1
                      : 0;
        d_tau_peak[t]   = peak;
        d_tau_end[t]    = end;
        d_off[t]        = o;
        d_iterations[t] = it;
        d_refused[t]    = rf;
    });

    std::vector<double> tau_peak(points), tau_end(points), off(points);
    std::vector<int> ok(points), iterations(points), refused(points);
    q.memcpy(tau_peak.data(), d_tau_peak, sizeof(double) * points);
    q.memcpy(tau_end.data(), d_tau_end, sizeof(double) * points);
    q.memcpy(off.data(), d_off, sizeof(double) * points);
    q.memcpy(ok.data(), d_ok, sizeof(int) * points);
    q.memcpy(iterations.data(), d_iterations, sizeof(int) * points);
    q.memcpy(refused.data(), d_refused, sizeof(int) * points);
    q.wait();
    for (void* ptr : {static_cast<void*>(d_tau_peak), static_cast<void*>(d_tau_end),
                      static_cast<void*>(d_off), static_cast<void*>(d_ok),
                      static_cast<void*>(d_iterations), static_cast<void*>(d_refused)})
        sycl::free(ptr, q);

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
