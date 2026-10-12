/// \file
/// \brief Example: Iwan multisurface plasticity on a batch of
/// integration points on GPU, the Newton systems held by each thread
/// through the internal residencies of the compile-time ArrowheadLUpp
/// solver.
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
/// Why the residencies are internal: each thread hands the solver local
/// arrays for the jacobian, the workspace and the residual, with the
/// residency booleans set to true, so the 30 x 30 Newton system and the
/// back stresses never leave the thread. The material is a kernel
/// argument; device memory only holds the per-point outcomes.
///
/// Each point runs its own shear cycle (iwan.hpp), and the stress must
/// reach the skeleton at the peak and its opposite at the end.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "iwan.hpp"

namespace {

using namespace iwan;

/// \brief One integration point per thread: its shear cycle with the
/// Newton systems local to the thread.
/// \param[in]  mat        material
/// \param[in]  points     number of integration points
/// \param[out] tau_peak   shear stress at the peak
/// \param[out] tau_end    shear stress at the end
/// \param[out] off        stress off the direction at the end
/// \param[out] ok         per-point success flags
/// \param[out] iterations per-point Newton iterations
/// \param[out] refused    per-point refused blocks
__global__ void cycle_batch(const Material mat, const int points, double* tau_peak, double* tau_end,
                            double* off, int* ok, int* iterations, int* refused) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= points) return;
    double A[Solver::storage_size], work[Solver::workspace_size], r[N];
    double peak = 0, end = 0, o = 0;
    int it = 0, rf = 0;
    ok[t] = cycle<true, true, true>(mat, static_cast<unsigned long long>(t), A, 1, work, 1, r, 1,
                                    peak, end, o, it, rf)
                ? 1
                : 0;
    tau_peak[t]   = peak;
    tau_end[t]    = end;
    off[t]        = o;
    iterations[t] = it;
    refused[t]    = rf;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    const int points = argc > 1 ? std::atoi(argv[1]) : 1 << 16;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const Material mat = calibrate();

    double *d_tau_peak = nullptr, *d_tau_end = nullptr, *d_off = nullptr;
    int *d_ok = nullptr, *d_iterations = nullptr, *d_refused = nullptr;
    GPU_CHECK(gpuMalloc(&d_tau_peak, sizeof(double) * points));
    GPU_CHECK(gpuMalloc(&d_tau_end, sizeof(double) * points));
    GPU_CHECK(gpuMalloc(&d_off, sizeof(double) * points));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_iterations, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_refused, sizeof(int) * points));

    const int block = 128;
    cycle_batch<<<(points + block - 1) / block, block>>>(mat, points, d_tau_peak, d_tau_end, d_off,
                                                         d_ok, d_iterations, d_refused);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> tau_peak(points), tau_end(points), off(points);
    std::vector<int> ok(points), iterations(points), refused(points);
    GPU_CHECK(
        gpuMemcpy(tau_peak.data(), d_tau_peak, sizeof(double) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(tau_end.data(), d_tau_end, sizeof(double) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(off.data(), d_off, sizeof(double) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(
        gpuMemcpy(iterations.data(), d_iterations, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(refused.data(), d_refused, sizeof(int) * points, gpuMemcpyDeviceToHost));
    for (void* ptr : {static_cast<void*>(d_tau_peak), static_cast<void*>(d_tau_end),
                      static_cast<void*>(d_off), static_cast<void*>(d_ok),
                      static_cast<void*>(d_iterations), static_cast<void*>(d_refused)})
        GPU_CHECK(gpuFree(ptr));

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
