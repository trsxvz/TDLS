/// \file
/// \brief Example: parameter sweep of Love's integral equation on GPU,
/// one dense Nystroem system per plate separation held in thread-local
/// storage, solved with the runtime TiledLUpp solver at unit stride.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Companion of the sequential and OpenMP integral_equation examples at
/// the GPU scale: the capacitor potential is computed for a large sweep
/// of plate separations d, one thread per instance.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched, so the dimension is a runtime value and
/// TiledLUppSolverDynamic applies.
///
/// Placement on GPU: the runtime solver has no residency booleans,
/// because a runtime dimension forces runtime indexing; the placement
/// of the operands is instead a property of the pointers and strides
/// handed to it. Here each thread assembles its system in thread-local
/// arrays and solves at unit stride, so the storage footprint is
/// resident threads times one system, not batch times one system: the
/// batch of matrices never exists in device memory, only the per
/// instance results do. Device memory holds nothing but the outputs;
/// the plate separation of an instance is one ramp evaluation away.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "love.hpp"

namespace {

using namespace love;

constexpr int instances = 1 << 17; ///< plate separations in the sweep
constexpr int max_n     = 40;      ///< bound of the runtime resolution

/// \brief One instance per thread: assembles its Nystroem system in
/// thread-local arrays, factorizes it once, substitutes the
/// manufactured and the physical right-hand sides, and writes the
/// manufactured error and the physical potential.
/// \param[in]  n     quadrature resolution
/// \param[out] err   per-instance error against the manufactured solution
/// \param[out] u_mid per-instance potential at the central node
/// \param[out] ok    per-instance success flags
__global__ void capacitor_sweep(const int n, double* err, double* u_mid, int* ok) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= instances) return;

    // Thread-local storage, unit strides.
    double A[max_n * max_n], g[max_n], u[max_n];
    int piv[max_n];
    assemble(n, separation(t, instances), A, 1, g, 1);

    // One factorization, two right-hand sides.
    if (!Solver::factorize(n, A, 1, piv, 1)) {
        err[t]   = 1.0;
        u_mid[t] = 0.0;
        ok[t]    = 0;
        return;
    }
    Solver::substitute(n, A, 1, piv, 1, g, u, 1);
    err[t] = manufactured_error(n, u, 1);

    for (int i = 0; i < n; ++i)
        g[i] = 1.0;
    Solver::substitute(n, A, 1, piv, 1, g, u, 1);
    u_mid[t] = u[(n - 1) / 2];
    ok[t]    = 1;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    // The quadrature resolution comes from outside the program, bounded
    // here by the thread-local array capacity; the default is odd so
    // that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 33;
    if (n < 5 || n > max_n) {
        std::printf("usage: %s [quadrature points in [5, %d]]\n", argv[0], max_n);
        return 1;
    }

    double *d_err = nullptr, *d_umid = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_err, sizeof(double) * instances));
    GPU_CHECK(gpuMalloc(&d_umid, sizeof(double) * instances));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * instances));

    const int block = 256;
    capacitor_sweep<<<(instances + block - 1) / block, block>>>(n, d_err, d_umid, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> err(instances), u_mid(instances);
    std::vector<int> ok(instances);
    GPU_CHECK(gpuMemcpy(err.data(), d_err, sizeof(double) * err.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(
        gpuMemcpy(u_mid.data(), d_umid, sizeof(double) * u_mid.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_err));
    GPU_CHECK(gpuFree(d_umid));
    GPU_CHECK(gpuFree(d_ok));

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
