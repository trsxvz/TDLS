/// \file
/// \brief Example: a Berveiller-Zaoui polycrystal of 12 grains on a
/// batch of integration points on GPU, every operand local to its
/// thread through the internal residencies of the compile-time
/// ArrowheadLUpp solver.
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
/// Why the residencies are internal: each thread hands the solver local
/// arrays for the state, the jacobian, the workspace and the residual,
/// with the residency booleans set to true. This is the simplest code.
/// The jacobian alone holds 1512 entries: the arrays live in the local
/// memory of the thread, not in its registers, and the device reserves
/// that memory for every thread it can hold at once. Here this variant
/// runs at the speed of polycrystal_batch_gpu_soa.cu: the assembly of
/// the jacobian, 144 slip systems and as many powers per iteration,
/// outweighs the placement of the operands. Device memory holds the
/// states before and after the last step, structure-of-arrays.
///
/// The self-checks run serially on the host: every point for the
/// deviatoric viscoplastic strain and the dissipation of every grain,
/// and a sample of points against their whole history integrated again
/// with a dense LU.

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
/// every operand local to the thread, then the final state.
/// \param[in]  points     number of integration points
/// \param[in]  steps      number of time steps
/// \param[in]  dt         time step
/// \param[out] before     states before the last step, SoA
/// \param[out] after      states after the last step, SoA
/// \param[out] ok         per-point success flags
/// \param[out] iterations per-point Newton iterations
/// \param[out] refused    per-point refused blocks
__global__ void integrate_batch(const int points, const int steps, const double dt, double* before,
                                double* after, int* ok, int* iterations, int* refused) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= points) return;
    double state[n], du[n], r[n], A[Solver::storage_size], work[Solver::workspace_size];
    int it = 0, rf = 0;
    const bool success = history<G, true, true, true, true>(
        static_cast<unsigned long long>(t), G, steps, dt, state, du, 1, before + t, points, A, 1,
        nullptr, 1, work, 1, r, 1, it, rf);
    for (int k = 0; k < n; ++k)
        after[static_cast<std::size_t>(k) * points + t] = state[k];
    ok[t]         = success ? 1 : 0;
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
    double *d_before = nullptr, *d_after = nullptr;
    int *d_ok = nullptr, *d_iterations = nullptr, *d_refused = nullptr;
    GPU_CHECK(gpuMalloc(&d_before, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_after, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_iterations, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_refused, sizeof(int) * points));

    const int block = 128;
    integrate_batch<<<(points + block - 1) / block, block>>>(points, steps, dt, d_before, d_after,
                                                             d_ok, d_iterations, d_refused);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> before(size), after(size);
    std::vector<int> ok(points), iterations(points), refused(points);
    GPU_CHECK(gpuMemcpy(before.data(), d_before, sizeof(double) * size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(after.data(), d_after, sizeof(double) * size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(
        gpuMemcpy(iterations.data(), d_iterations, sizeof(int) * points, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(refused.data(), d_refused, sizeof(int) * points, gpuMemcpyDeviceToHost));
    for (void* ptr :
         {static_cast<void*>(d_before), static_cast<void*>(d_after), static_cast<void*>(d_ok),
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
