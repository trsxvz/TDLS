/// \file
/// \brief Example: parameter sweep of Love's integral equation on a
/// SYCL device, one dense Nystroem system per plate separation held in
/// the local memory of a work-group and shared by a group of work-items
/// of the runtime CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the
/// computation is launched, so the dimension is a runtime value and
/// CooperativeLUppSolverDynamic applies.
///
/// Why SYCL: the kernel is a single-source nd_range parallel_for and
/// the solver headers compile as plain C++ inside it: no decoration is
/// needed in this model. With four rows per work-item, an instance
/// takes a group of ceil(n / 4) work-items, nine at the default
/// resolution. A work-group solves as many instances as fit in 32
/// work-items and in its local memory. Each group assembles its rows
/// of the system there and solves at unit stride: the batch of
/// matrices never exists in device memory, only the per-instance
/// outputs do, in device USM. The sweep runs on whatever device the
/// default selector picks, which is why it stays moderate. A machine
/// without a SYCL device reports the test skipped.
///
/// The barrier is the one of the work-group, which every SYCL device
/// provides. The instances of a work-group make the same calls, since
/// the sequence of an instance depends on n alone.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>

#include "love.hpp"

namespace {

constexpr int instances = 20000; ///< plate separations in the sweep
constexpr int max_n     = 40;    ///< bound of the runtime resolution

} // namespace

int main(int argc, char** argv) {
    using namespace love;
    using Solver = GroupSolver<4>; // four rows per work-item
    // The quadrature resolution comes from outside the program, bounded
    // here by the local memory of a work-group; the default is odd so
    // that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 33;
    if (n < 5 || n > max_n) {
        std::printf("usage: %s [quadrature points in [5, %d]]\n", argv[0], max_n);
        return 1;
    }

    // A machine without a SYCL runtime exposes no device: the test
    // reports itself skipped, as the CUDA and HIP examples do. The
    // queue is in order, so the readbacks see the kernel.
    std::optional<sycl::queue> queue;
    try {
        queue.emplace(sycl::property_list{sycl::property::queue::in_order{}});
    } catch (const sycl::exception&) {
        std::printf("no device available, skipping\n");
        return 77;
    }
    sycl::queue& q = *queue;

    // A group keeps A, g, u and its workspace in local memory, plus the
    // pivot. The work-group takes as many groups as fit in 32 work-items
    // and in the local memory of the device.
    const int threads              = Solver::threads_per_system(n);
    const int doubles              = n * n + 2 * n + Solver::workspace_size(n);
    const std::size_t bytes        = sizeof(double) * static_cast<std::size_t>(doubles) +
                                     sizeof(int) * static_cast<std::size_t>(n);
    const std::size_t local_memory = q.get_device().get_info<sycl::info::device::local_mem_size>();
    const int groups               = std::min(32 / threads, static_cast<int>(local_memory / bytes));
    if (groups < 1) {
        std::printf("the local memory of the device cannot hold one system of size %d\n", n);
        return 1;
    }
    const int local = groups * threads;

    double* d_err  = sycl::malloc_device<double>(instances, q);
    double* d_umid = sycl::malloc_device<double>(instances, q);
    int* d_ok      = sycl::malloc_device<int>(instances, q);

    // One instance per group of work-items: assembles its Nystroem
    // system in local memory, factorizes it once, substitutes the
    // manufactured and the physical right-hand sides, and writes the
    // manufactured error and the physical potential. The last
    // work-group may hold groups past the sweep: they solve the last
    // instance again, to keep pace with the others, and write nothing.
    const std::size_t work_groups =
        (static_cast<std::size_t>(instances) + groups - 1) / static_cast<std::size_t>(groups);
    q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<double, 1> systems(sycl::range<1>(groups * doubles), handler);
        sycl::local_accessor<int, 1> pivots(sycl::range<1>(groups * n), handler);
        handler.parallel_for(
            sycl::nd_range<1>(work_groups * local, local), [=](const sycl::nd_item<1> item) {
                const auto wg      = item.get_group();
                const int lid      = static_cast<int>(item.get_local_linear_id());
                const int group    = lid / threads;
                const int tx       = lid % threads;
                const int instance = static_cast<int>(wg.get_group_linear_id()) * groups + group;
                const bool owner   = instance < instances && tx == 0;
                const int c        = instance < instances ? instance : instances - 1;
                auto sync          = [&wg] { sycl::group_barrier(wg); };

                // Local storage of the group, unit strides.
                double* A    = &systems[group * doubles];
                double* g    = A + n * n;
                double* u    = g + n;
                double* work = u + n;
                int* piv     = &pivots[group * n];

                // One factorization, two right-hand sides.
                double err = 0, u_mid = 0;
                const bool ok = capacitor_group<Solver>(n, tx, sync, separation(c, instances), A, 1,
                                                        piv, 1, g, u, 1, work, err, u_mid);

                // Every work-item of the group holds the same results.
                if (!owner) return;
                d_err[c]  = ok ? err : 1.0;
                d_umid[c] = ok ? u_mid : 0.0;
                d_ok[c]   = ok ? 1 : 0;
            });
    });

    std::vector<double> err(instances), u_mid(instances);
    std::vector<int> ok(instances);
    q.memcpy(err.data(), d_err, sizeof(double) * err.size());
    q.memcpy(u_mid.data(), d_umid, sizeof(double) * u_mid.size());
    q.memcpy(ok.data(), d_ok, sizeof(int) * ok.size());
    q.wait();
    sycl::free(d_err, q);
    sycl::free(d_umid, q);
    sycl::free(d_ok, q);

    int failures = 0;
    double e_max = 0.0, u_lo = 1.0, u_hi = 0.0;
    for (int t = 0; t < instances; ++t) {
        if (!ok[t]) ++failures;
        e_max = std::fmax(e_max, err[t]);
        u_lo  = std::fmin(u_lo, u_mid[t]);
        u_hi  = std::fmax(u_hi, u_mid[t]);
    }

    std::printf("instances = %d, n = %d, groups of %d work-items, %d per work-group, "
                "manufactured error = %.3e, u(0): %.6f at d = %.1f -> %.6f at d = %.1f\n",
                instances, n, threads, groups, e_max, u_mid.front(), d_min, u_mid.back(), d_max);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return failures == 0 && e_max < 1e-12 && u_lo > 0.4 && u_hi < 1.0 ? 0 : 1;
}
