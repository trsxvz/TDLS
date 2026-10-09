/// \file
/// \brief Example: Norton viscoplasticity on a batch of integration
/// points on a SYCL device, each Newton system shared by a group of
/// work-items of the compile-time CooperativeLUpp solver.
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
/// Why SYCL: the kernel is a single-source nd_range parallel_for and
/// the solver headers compile as plain C++ inside it: no decoration is
/// needed in this model. With two rows per work-item, a point takes a
/// group of four work-items, and a work-group of 32 work-items solves
/// eight points. Each work-item holds its rows of the jacobian, their
/// pivot entries and their residual entries in private memory: the
/// internal residencies. The workspaces live in the local memory of the
/// work-group, and device USM holds the per-point state and outputs,
/// structure-of-arrays. The batch runs on whatever device the default
/// selector picks, which is why it stays moderate. A machine without a
/// SYCL device reports the test skipped.
///
/// Why a work-group barrier: it is the barrier every SYCL device
/// provides, whatever the size of its sub-groups. Its scope is wider
/// than the group of a point, so the eight points of a work-group make
/// the same calls: the Newton loop and the time loop run while one of
/// them still iterates, a logical or over the work-group.
///
/// The state before the last step is recorded, so that the last step
/// can be checked on every point against the radial return, and on a
/// sample of points against central differences of the tangent
/// operator.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>

#include "norton.hpp"

int main(int argc, char** argv) {
    using namespace norton;
    // Two rows per work-item: a point takes a group of four work-items,
    // and a work-group of 32 work-items solves eight points.
    using Solver          = GroupSolver<2, true>;
    constexpr int threads = Solver::threads_per_system;
    constexpr int groups  = 32 / threads;
    constexpr int local   = groups * threads;

    const int points = argc > 1 ? std::atoi(argv[1]) : 65536;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const double dt = 1.0; // time step (s)
    const int steps = 10;  // integrate each point to t = 10 s

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

    double* d_eel_before =
        sycl::malloc_device<double>(static_cast<std::size_t>(stensor_size) * points, q);
    double* d_p_before = sycl::malloc_device<double>(points, q);
    double* d_sig = sycl::malloc_device<double>(static_cast<std::size_t>(stensor_size) * points, q);
    double* d_p   = sycl::malloc_device<double>(points, q);
    double* d_Dt  = sycl::malloc_device<double>(
        static_cast<std::size_t>(stensor_size) * stensor_size * points, q);
    int* d_ok = sycl::malloc_device<int>(points, q);

    // One point per group of work-items: integrates the loading history
    // with the rows of the Newton systems in private memory, records the
    // state before the last step for the host-side checks, and writes
    // the final stress, tangent operator and success flag. The last
    // work-group may hold groups past the batch: they integrate the last
    // point again, to keep pace with the others, and write nothing.
    const std::size_t work_groups = (static_cast<std::size_t>(points) + groups - 1) / groups;
    q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<double, 1> workspaces(sycl::range<1>(groups * Solver::workspace_size),
                                                   handler);
        handler.parallel_for(
            sycl::nd_range<1>(work_groups * local, local), [=](const sycl::nd_item<1> item) {
                const auto wg    = item.get_group();
                const int lid    = static_cast<int>(item.get_local_linear_id());
                const int group  = lid / threads;
                const int tx     = lid % threads;
                const int point  = static_cast<int>(wg.get_group_linear_id()) * groups + group;
                const bool owner = point < points && tx == 0;
                const int t      = point < points ? point : points - 1;
                double* work     = &workspaces[group * Solver::workspace_size];
                auto sync        = [&wg] { sycl::group_barrier(wg); };
                auto any         = [&wg](const bool flag) { return sycl::any_of_group(wg, flag); };

                double eel[stensor_size] = {};
                double p_t               = 0;
                double sig_t[stensor_size], Dt_t[stensor_size * stensor_size] = {};
                double J[Solver::rows_per_thread * N], r[Solver::rows_per_thread];
                int piv[Solver::rows_per_thread];
                double deto[stensor_size];
                strain_increment(static_cast<unsigned long long>(t), deto);

                bool success = true;
                for (int s = 0; s < steps && any(success); ++s) {
                    if (s == steps - 1 && owner) {
                        for (int k = 0; k < stensor_size; ++k)
                            d_eel_before[static_cast<std::size_t>(k) * points + t] = eel[k];
                        d_p_before[t] = p_t;
                    }
                    const bool step = integrate_group<true, true, true, Solver>(
                        tx, sync, any, deto, dt, eel, p_t, sig_t, Dt_t, J, 1, piv, 1, r, 1, work);
                    success = success && step;
                }

                // Every work-item of the group holds the same results.
                if (!owner) return;
                for (int k = 0; k < stensor_size; ++k)
                    d_sig[static_cast<std::size_t>(k) * points + t] = sig_t[k];
                for (int k = 0; k < stensor_size * stensor_size; ++k)
                    d_Dt[static_cast<std::size_t>(k) * points + t] = Dt_t[k];
                d_p[t]  = p_t;
                d_ok[t] = success ? 1 : 0;
            });
    });

    std::vector<double> eel_before(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> sig(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> Dt(static_cast<std::size_t>(stensor_size) * stensor_size * points);
    std::vector<double> p_before(points), p(points);
    std::vector<int> ok(points);
    q.memcpy(eel_before.data(), d_eel_before, sizeof(double) * eel_before.size());
    q.memcpy(p_before.data(), d_p_before, sizeof(double) * p_before.size());
    q.memcpy(sig.data(), d_sig, sizeof(double) * sig.size());
    q.memcpy(p.data(), d_p, sizeof(double) * p.size());
    q.memcpy(Dt.data(), d_Dt, sizeof(double) * Dt.size());
    q.memcpy(ok.data(), d_ok, sizeof(int) * ok.size());
    q.wait();
    for (void* ptr : {static_cast<void*>(d_eel_before), static_cast<void*>(d_p_before),
                      static_cast<void*>(d_sig), static_cast<void*>(d_p), static_cast<void*>(d_Dt),
                      static_cast<void*>(d_ok)})
        sycl::free(ptr, q);

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
