/// \file
/// \brief Example: Berveiller-Zaoui aggregates whose number of grains is
/// given at run time, on a batch of integration points on GPU, every
/// operand in device memory in SoA layout, solved by the runtime
/// ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Companion of aggregate_batch_gpu.cu: same behaviour, same batch,
/// opposite layout of the batches. Here element k of an operand of
/// point t sits at base[k * points + t], structure-of-arrays. The
/// runtime solver takes a stride for every operand: handed the batch
/// stride, it makes consecutive threads touch consecutive addresses,
/// and every access of a warp is coalesced. Only the (pointer, stride)
/// pairs differ from the AoS variant. Both print the time of their
/// kernel: this one is 6 to 8 % faster, the batch being too small to
/// fill the device (see aggregate_batch_gpu.cu).
///
/// The self-checks run serially on the host: every point for the
/// deviatoric viscoplastic strain and the dissipation of every grain,
/// and a sample of points against their whole history integrated again
/// on the host.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "polycrystal.hpp"

namespace {

using namespace polycrystal;

/// \brief One integration point per thread: its loading history with
/// every operand in the SoA batches.
/// \param[in]  points     number of integration points
/// \param[in]  grains     number of grains
/// \param[in]  steps      number of time steps
/// \param[in]  dt         time step
/// \param[out] state      state batch, the states after the last step
/// \param[out] du         increment batch
/// \param[out] A          jacobian batch
/// \param[out] piv        pivot batch
/// \param[out] work       workspace batch
/// \param[out] r          residual batch
/// \param[out] before     states before the last step
/// \param[out] ok         per-point success flags
/// \param[out] iterations per-point Newton iterations
/// \param[out] refused    per-point refused blocks
__global__ void integrate_batch(const int points, const int grains, const int steps,
                                const double dt, double* state, double* du, double* A, int* piv,
                                double* work, double* r, double* before, int* ok, int* iterations,
                                int* refused) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= points) return;
    int it = 0, rf = 0;
    // Grains = 0: the runtime solver, every operand walked with the
    // batch stride.
    ok[t] = history<0, false, false, false, false>(static_cast<unsigned long long>(t), grains,
                                                   steps, dt, state + t, du + t, points, before + t,
                                                   points, A + t, points, piv + t, points, work + t,
                                                   points, r + t, points, it, rf)
                ? 1
                : 0;
    iterations[t] = it;
    refused[t]    = rf;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    const int points = argc > 1 ? std::atoi(argv[1]) : 2048;
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

    const auto P           = static_cast<std::size_t>(points);
    const std::size_t size = static_cast<std::size_t>(n) * P;
    double *d_state = nullptr, *d_du = nullptr, *d_A = nullptr, *d_work = nullptr, *d_r = nullptr,
           *d_before = nullptr;
    int *d_piv = nullptr, *d_ok = nullptr, *d_iterations = nullptr, *d_refused = nullptr;
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_du, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_A, sizeof(double) * Map::size(shape) * P));
    GPU_CHECK(gpuMalloc(&d_piv, sizeof(int) * DynamicSolver::pivot_size(shape) * P));
    GPU_CHECK(gpuMalloc(&d_work, sizeof(double) * DynamicSolver::workspace_size(shape) * P));
    GPU_CHECK(gpuMalloc(&d_r, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_before, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_iterations, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_refused, sizeof(int) * points));

    const int block = 128;
    const auto t0   = std::chrono::steady_clock::now();
    integrate_batch<<<(points + block - 1) / block, block>>>(points, G, steps, dt, d_state, d_du,
                                                             d_A, d_piv, d_work, d_r, d_before,
                                                             d_ok, d_iterations, d_refused);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<double> before(size), after(size);
    std::vector<int> ok(points), iterations(points), refused(points);
    GPU_CHECK(gpuMemcpy(before.data(), d_before, sizeof(double) * size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(after.data(), d_state, sizeof(double) * size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(
        gpuMemcpy(iterations.data(), d_iterations, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(refused.data(), d_refused, sizeof(int) * points, gpuMemcpyDeviceToHost));
    for (void* ptr :
         {static_cast<void*>(d_state), static_cast<void*>(d_du), static_cast<void*>(d_A),
          static_cast<void*>(d_piv), static_cast<void*>(d_work), static_cast<void*>(d_r),
          static_cast<void*>(d_before), static_cast<void*>(d_ok), static_cast<void*>(d_iterations),
          static_cast<void*>(d_refused)})
        GPU_CHECK(gpuFree(ptr));

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), after.data(), 1,
                                             points, ok.data(), 256, false);
    long long total_iterations = 0, total_refused = 0;
    for (int i = 0; i < points; ++i) {
        total_iterations += iterations[i];
        total_refused += refused[i];
    }

    std::printf("points = %d, grains = %d, Newton iterations = %lld, refused blocks = %lld, "
                "kernel = %.3f s\n",
                points, G, total_iterations, total_refused,
                std::chrono::duration<double>(t1 - t0).count());
    std::printf("failures = %d, largest Sig_eq = %.1f MPa, deviation from the host = %.1e, "
                "|tr Evp| / |Evp| = %.1e, smallest grain dissipation = %.1e Pa\n",
                check.failures, check.seq / 1e6, check.reference, check.trace, check.dissipation);
    return check.failures == 0 && check.reference < 1e-10 && check.trace < 1e-12 &&
                   check.dissipation > -1e-5
               ? 0
               : 1;
}
