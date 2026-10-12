/// \file
/// \brief Example: Berveiller-Zaoui aggregates whose number of grains is
/// given at run time, on a batch of integration points on GPU, every
/// operand of a point contiguous in device memory (AoS layout), solved
/// by the runtime ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the shape is known at run time: exactly as in the other aggregate
/// examples, the number of grains is a datum of the computation (240 by
/// default, 1452 unknowns), so ArrowheadLUppSolverDynamic takes the
/// shape as its first argument.
///
/// Why device memory: a thread has no local array of runtime size, and
/// the jacobian of a point holds 27504 entries anyway. Every operand of
/// a point (state, increments, jacobian, workspace, pivots, residual)
/// lives in a device memory batch. Here each point owns a contiguous
/// slice of each batch, the layout of an array of structures: the
/// solver walks it with a stride of 1. Consecutive threads then read
/// addresses a whole slice apart, and every access of a warp costs as
/// many memory transactions as it has threads. aggregate_batch_gpu_soa.cu
/// lays the same batches out the other way; both print the time of their
/// kernel. Each thread runs the long sequential work of a whole point,
/// with about 280 KB of operands: the batches that fit in the memory of
/// a 16 GB device hold far fewer points than the device holds threads,
/// and the kernel time hardly changes from 2048 to 8192 points. The
/// coalescing then matters less than usual: the SoA layout is 6 to 8 %
/// faster here.
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

/// \brief Distances between two points in the batches, the entries of
/// one point being contiguous.
struct Slices {
    int n;       ///< unknowns
    int storage; ///< entries of the jacobian
    int work;    ///< entries of the workspace
    int piv;     ///< entries of the pivot array
};

/// \brief One integration point per thread: its loading history with
/// every operand in its slices of the AoS batches.
/// \param[in]  points     number of integration points
/// \param[in]  grains     number of grains
/// \param[in]  steps      number of time steps
/// \param[in]  dt         time step
/// \param[in]  sl         sizes of the slices
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
                                const double dt, const Slices sl, double* state, double* du,
                                double* A, int* piv, double* work, double* r, double* before,
                                int* ok, int* iterations, int* refused) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= points) return;
    const auto T = static_cast<std::size_t>(t);
    int it = 0, rf = 0;
    // Grains = 0: the runtime solver, each operand from the start of the
    // slice of the point, with a stride of 1.
    ok[t] = history<0, false, false, false, false>(
                static_cast<unsigned long long>(t), grains, steps, dt, state + T * sl.n,
                du + T * sl.n, 1, before + T * sl.n, 1, A + T * sl.storage, 1, piv + T * sl.piv, 1,
                work + T * sl.work, 1, r + T * sl.n, 1, it, rf)
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
    const Slices sl{unknowns(G), Map::size(shape), DynamicSolver::workspace_size(shape),
                    DynamicSolver::pivot_size(shape)};
    const double dt = 1.0; // time step (s)
    const int steps = 20;  // integrate each point to t = 20 s

    const auto P           = static_cast<std::size_t>(points);
    const std::size_t size = static_cast<std::size_t>(sl.n) * P;
    double *d_state = nullptr, *d_du = nullptr, *d_A = nullptr, *d_work = nullptr, *d_r = nullptr,
           *d_before = nullptr;
    int *d_piv = nullptr, *d_ok = nullptr, *d_iterations = nullptr, *d_refused = nullptr;
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_du, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_A, sizeof(double) * sl.storage * P));
    GPU_CHECK(gpuMalloc(&d_piv, sizeof(int) * sl.piv * P));
    GPU_CHECK(gpuMalloc(&d_work, sizeof(double) * sl.work * P));
    GPU_CHECK(gpuMalloc(&d_r, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_before, sizeof(double) * size));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_iterations, sizeof(int) * points));
    GPU_CHECK(gpuMalloc(&d_refused, sizeof(int) * points));

    const int block = 128;
    const auto t0   = std::chrono::steady_clock::now();
    integrate_batch<<<(points + block - 1) / block, block>>>(
        points, G, steps, dt, sl, d_state, d_du, d_A, d_piv, d_work, d_r, d_before, d_ok,
        d_iterations, d_refused);
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
    const BatchCheck check = check_batch(points, G, steps, dt, before.data(), after.data(),
                                         static_cast<std::size_t>(sl.n), 1, ok.data(), 256, false);
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
