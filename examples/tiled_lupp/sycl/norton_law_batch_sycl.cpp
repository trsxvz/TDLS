/// \file
/// \brief Example: Norton viscoplasticity on a batch of integration
/// points on a SYCL device, the Newton systems living in the private
/// memory of each work-item.
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
/// Why SYCL: the kernel is a single-source parallel_for, one work-item
/// per integration point, and the solver headers compile as plain C++
/// inside it: no decoration is needed in this model. Every solver
/// operand is a local array of the work-item, so the 7 x 7 Newton
/// systems live in private memory; device USM holds the per-point
/// state and outputs, structure-of-arrays. The batch runs on whatever
/// device the default selector picks, which is why it stays moderate.
/// A machine without a SYCL device reports the test skipped.
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

    // One integration point per work-item: integrates the loading
    // history with the Newton systems in private memory, records the
    // state before the last step for the host-side checks, and writes
    // the final stress, tangent operator and success flag.
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(points)), [=](sycl::id<1> idx) {
        const int t = static_cast<int>(idx[0]);

        double eel[stensor_size] = {};
        double p_t               = 0;
        double sig_t[stensor_size], Dt_t[stensor_size * stensor_size] = {};
        double J[N * N], r[N];
        int piv[N];
        double deto[stensor_size];
        strain_increment(static_cast<unsigned long long>(t), deto);

        bool success = true;
        for (int s = 0; s < steps && success; ++s) {
            if (s == steps - 1) {
                for (int k = 0; k < stensor_size; ++k)
                    d_eel_before[static_cast<std::size_t>(k) * points + t] = eel[k];
                d_p_before[t] = p_t;
            }
            success =
                integrate<true, true, true>(deto, dt, eel, p_t, sig_t, Dt_t, J, 1, piv, 1, r, 1);
        }

        for (int k = 0; k < stensor_size; ++k)
            d_sig[static_cast<std::size_t>(k) * points + t] = sig_t[k];
        for (int k = 0; k < stensor_size * stensor_size; ++k)
            d_Dt[static_cast<std::size_t>(k) * points + t] = Dt_t[k];
        d_p[t]  = p_t;
        d_ok[t] = success ? 1 : 0;
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
