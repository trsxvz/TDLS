/// \file
/// \brief Example: a Berveiller-Zaoui polycrystal of 12 grains on a
/// batch of integration points on GPU, with the state and the Newton
/// systems materialized in device memory in SoA layout and solved
/// through the external residencies of the compile-time ArrowheadLUpp
/// solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Companion of polycrystal_batch_gpu.cu: same behaviour, same batch,
/// opposite placement of the operands. Here the state, the increments,
/// the jacobian, the workspace and the residual of every point are
/// slices of device memory batches, structure-of-arrays: element k of
/// point t sits at base[k * points + t]. The solver is called with the
/// residency booleans set to false and the batch stride, so consecutive
/// threads touch consecutive addresses and every access is coalesced.
/// Only the residency booleans and the (pointer, stride) pairs change.
/// The local arrays of the solver (the copy of a block, its solved
/// columns, the Schur complement) stay small. Here both variants run at
/// the same speed, the assembly of the jacobian outweighing the
/// placement of the operands. This placement is the one larger shapes
/// need: the local memory reserved for every thread the device can hold
/// grows with the shape, and exhausts the device first. The price is
/// that the batch occupies device memory, about 18 KB per point, which
/// bounds its size.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "polycrystal.hpp"

namespace {

using namespace polycrystal;
constexpr int G = fixed_grains;
constexpr int n = unknowns(G);
using Solver    = StaticSolver<G>;

/// \brief One integration point per thread: its loading history with
/// every operand in the SoA batches.
/// \param[in]  points     number of integration points
/// \param[in]  steps      number of time steps
/// \param[in]  dt         time step
/// \param[out] state      state batch, the states after the last step
/// \param[out] du         increment batch
/// \param[out] A          jacobian batch
/// \param[out] work       workspace batch
/// \param[out] r          residual batch
/// \param[out] before     states before the last step, SoA
/// \param[out] ok         per-point success flags
/// \param[out] iterations per-point Newton iterations
/// \param[out] refused    per-point refused blocks
__global__ void integrate_batch(const int points, const int steps, const double dt, double* state,
                                double* du, double* A, double* work, double* r, double* before,
                                int* ok, int* iterations, int* refused) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= points) return;
    int it = 0, rf = 0;
    // Every operand is walked with the batch stride.
    ok[t] = history<G, false, false, false, false>(static_cast<unsigned long long>(t), G, steps, dt,
                                                   state + t, du + t, points, before + t, points,
                                                   A + t, points, nullptr, 1, work + t, points,
                                                   r + t, points, it, rf)
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
    const int points = argc > 1 ? std::atoi(argv[1]) : 1 << 14;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const double dt = 1.0; // time step (s)
    const int steps = 20;  // integrate each point to t = 20 s

    const std::size_t size = static_cast<std::size_t>(n) * points;
    double *d_state = nullptr, *d_du = nullptr, *d_A = nullptr, *d_work = nullptr, *d_r = nullptr,
           *d_before = nullptr;
    int *d_ok = nullptr, *d_iterations = nullptr, *d_refused = nullptr;
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_du, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_A, sizeof(double) * Solver::storage_size * points));
    GPU_CHECK(gpuMalloc(&d_work, sizeof(double) * Solver::workspace_size * points));
    GPU_CHECK(gpuMalloc(&d_r, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_before, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_iterations, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_refused, sizeof(int) * points));

    const int block = 128;
    integrate_batch<<<(points + block - 1) / block, block>>>(points, steps, dt, d_state, d_du, d_A,
                                                             d_work, d_r, d_before, d_ok,
                                                             d_iterations, d_refused);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> before(size), after(size);
    std::vector<int> ok(points), iterations(points), refused(points);
    GPU_CHECK(gpuMemcpy(before.data(), d_before, sizeof(double) * size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(after.data(), d_state, sizeof(double) * size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(
        gpuMemcpy(iterations.data(), d_iterations, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(refused.data(), d_refused, sizeof(int) * points, gpuMemcpyDeviceToHost));
    for (void* ptr : {static_cast<void*>(d_state), static_cast<void*>(d_du),
                      static_cast<void*>(d_A), static_cast<void*>(d_work), static_cast<void*>(d_r),
                      static_cast<void*>(d_before), static_cast<void*>(d_ok),
                      static_cast<void*>(d_iterations), static_cast<void*>(d_refused)})
        GPU_CHECK(gpuFree(ptr));

    // Serial checks of the last step over the gathered outputs.
    const BatchCheck check     = check_batch(points, G, steps, dt, before.data(), after.data(), 1,
                                             points, ok.data(), 256, true);
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
