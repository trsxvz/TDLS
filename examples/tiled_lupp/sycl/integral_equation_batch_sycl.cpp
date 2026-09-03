/// \file
/// \brief Example: parameter sweep of Love's integral equation on a
/// SYCL device, one dense Nystroem system per plate separation held in
/// the private memory of each work-item, solved with the runtime
/// TiledLUpp solver at unit stride.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the
/// computation is launched, so the dimension is a runtime value and
/// TiledLUppSolverDynamic applies.
///
/// Why SYCL: the kernel is a single-source parallel_for, one work-item
/// per instance, and the solver headers compile as plain C++ inside
/// it: no decoration is needed in this model. Each instance assembles
/// its system in arrays local to the work-item and solves at unit
/// stride: the batch of matrices never exists in memory, only the
/// per-instance outputs do, in device USM. The sweep runs on whatever
/// device the default selector picks, which is why it stays moderate.
/// A machine without a SYCL device reports the test skipped.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential.

#include <cmath>
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
    // The quadrature resolution comes from outside the program, bounded
    // here by the local array capacity; the default is odd so that
    // x = 0 is a node.
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

    double* d_err  = sycl::malloc_device<double>(instances, q);
    double* d_umid = sycl::malloc_device<double>(instances, q);
    int* d_ok      = sycl::malloc_device<int>(instances, q);

    // One instance per work-item: assembles its Nystroem system in
    // private arrays, factorizes it once, substitutes the manufactured
    // and the physical right-hand sides, and writes the manufactured
    // error and the physical potential.
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(instances)), [=](sycl::id<1> idx) {
        const int t = static_cast<int>(idx[0]);

        // Private storage, unit strides.
        double A[max_n * max_n], g[max_n], u[max_n];
        int piv[max_n];
        assemble(n, separation(t, instances), A, 1, g, 1);

        // One factorization, two right-hand sides.
        if (!Solver::factorize(n, A, 1, piv, 1)) {
            d_err[t]  = 1.0;
            d_umid[t] = 0.0;
            d_ok[t]   = 0;
            return;
        }
        Solver::substitute(n, A, 1, piv, 1, g, u, 1);
        d_err[t] = manufactured_error(n, u, 1);

        for (int i = 0; i < n; ++i)
            g[i] = 1.0;
        Solver::substitute(n, A, 1, piv, 1, g, u, 1);
        d_umid[t] = u[(n - 1) / 2];
        d_ok[t]   = 1;
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

    std::printf("instances = %d, n = %d, manufactured error = %.3e, "
                "u(0): %.6f at d = %.1f -> %.6f at d = %.1f\n",
                instances, n, e_max, u_mid.front(), d_min, u_mid.back(), d_max);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return failures == 0 && e_max < 1e-12 && u_lo > 0.4 && u_hi < 1.0 ? 0 : 1;
}
