/// \file
/// \brief Example: Norton viscoplasticity on a batch of integration
/// points on GPU, each Newton system shared by a group of lanes and
/// held in their registers through the internal residencies of the
/// compile-time CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: exactly as in the
/// sequential and OpenMP norton_law examples, N = 7 is fixed by the law
/// and by the modelling hypothesis. This is the shape of the problem
/// MFront-generated behaviours pose to TDLS.
///
/// Why groups of lanes: with four rows per lane, a point takes a group of
/// two lanes, and 32 lanes solve sixteen points. Each lane holds four
/// rows of the jacobian instead of seven, the last slot of the second
/// lane being a phantom row, so the registers of a lane hold a slice of
/// the system, not the whole of it. The barrier of a group, deduced by
/// the solver and returned by make_sync for the exchanges of the
/// example, is a warp barrier on its lanes only, so the groups of a
/// warp iterate their Newton loops independently.
///
/// Why the residencies are internal: each lane hands the solver local
/// arrays for its rows of the jacobian, their pivot entries and their
/// residual entries, with the residency booleans set to true, so the
/// Newton system lives in the registers of the group. The workspaces
/// live in shared memory. Device memory only holds the per-point state
/// and outputs, structure-of-arrays so that every access is coalesced.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "norton.hpp"

namespace {

using namespace norton;

constexpr int block = 256; ///< threads per block

/// \brief One integration point per group of lanes: integrates the
/// loading history with the rows of the Newton systems in registers,
/// records the state before the last step for the host-side checks, and
/// writes the final stress, tangent operator and success flag.
/// \tparam Solver a GroupSolver whose group fits in 32 lanes
/// \param[in]  points     number of integration points
/// \param[in]  steps      number of time steps
/// \param[in]  dt         time step
/// \param[out] eel_before elastic strain before the last step, SoA
/// \param[out] p_before   viscoplastic multiplier before the last step
/// \param[out] sig        final stress, SoA
/// \param[out] p          final viscoplastic multiplier
/// \param[out] Dt         final tangent operator, SoA of 36 components
/// \param[out] ok         per-point success flags
template<typename Solver>
__global__ void integrate_batch(const int points, const int steps, const double dt,
                                double* eel_before, double* p_before, double* sig, double* p,
                                double* Dt, int* ok) {
    constexpr int threads = Solver::threads_per_system;
    constexpr int groups  = 32 / threads; // groups per 32 lanes
    __shared__ double workspaces[block / 32 * groups * Solver::workspace_size];
    const int lane  = static_cast<int>(threadIdx.x) % 32;
    const int pack  = static_cast<int>(threadIdx.x) / 32;
    const int group = lane / threads;
    const int tx    = lane % threads;
    const int t     = (static_cast<int>(blockIdx.x) * (block / 32) + pack) * groups + group;
    // The lanes past the last group, and the groups past the batch,
    // leave at once: a barrier only involves the lanes of its group.
    if (group >= groups || t >= points) return;
    double* work = workspaces + (pack * groups + group) * Solver::workspace_size;
    auto sync    = Solver::make_sync(work);
    auto any     = [](const bool flag) { return flag; };

    double eel[stensor_size] = {};
    double p_t               = 0;
    double sig_t[stensor_size], Dt_t[stensor_size * stensor_size] = {};
    double J[Solver::rows_per_thread * N], r[Solver::rows_per_thread];
    int piv[Solver::rows_per_thread];
    double deto[stensor_size];
    strain_increment(static_cast<unsigned long long>(t), deto);

    bool success = true;
    for (int s = 0; s < steps && success; ++s) {
        if (s == steps - 1 && tx == 0) {
            for (int k = 0; k < stensor_size; ++k)
                eel_before[static_cast<std::size_t>(k) * points + t] = eel[k];
            p_before[t] = p_t;
        }
        success = integrate_group<true, true, true, Solver>(tx, sync, any, deto, dt, eel, p_t,
                                                            sig_t, Dt_t, J, 1, piv, 1, r, 1, work);
    }

    // Every lane of the group holds the same results.
    if (tx != 0) return;
    for (int k = 0; k < stensor_size; ++k)
        sig[static_cast<std::size_t>(k) * points + t] = sig_t[k];
    for (int k = 0; k < stensor_size * stensor_size; ++k)
        Dt[static_cast<std::size_t>(k) * points + t] = Dt_t[k];
    p[t]  = p_t;
    ok[t] = success ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    // Four rows per lane: a point takes a group of two lanes.
    using Solver = GroupSolver<4, true>;

    // Large batch: the Newton systems never leave the registers, device
    // memory holds 50 doubles per point of state and outputs.
    const int points = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    if (points < 1) {
        std::printf("usage: %s [number of integration points >= 1]\n", argv[0]);
        return 1;
    }
    const double dt = 1.0; // time step (s)
    const int steps = 10;  // integrate each point to t = 10 s

    double *d_eel_before = nullptr, *d_p_before = nullptr, *d_sig = nullptr, *d_p = nullptr,
           *d_Dt = nullptr;
    int* d_ok    = nullptr;
    GPU_CHECK(gpuMalloc(&d_eel_before, sizeof(double) * stensor_size * points));
    GPU_CHECK(gpuMalloc(&d_p_before, sizeof(double) * points));
    GPU_CHECK(gpuMalloc(&d_sig, sizeof(double) * stensor_size * points));
    GPU_CHECK(gpuMalloc(&d_p, sizeof(double) * points));
    GPU_CHECK(gpuMalloc(&d_Dt, sizeof(double) * stensor_size * stensor_size * points));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * points));

    const int per_block = block / 32 * (32 / Solver::threads_per_system);
    integrate_batch<Solver><<<(points + per_block - 1) / per_block, block>>>(
        points, steps, dt, d_eel_before, d_p_before, d_sig, d_p, d_Dt, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> eel_before(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> sig(static_cast<std::size_t>(stensor_size) * points);
    std::vector<double> Dt(static_cast<std::size_t>(stensor_size) * stensor_size * points);
    std::vector<double> p_before(points), p(points);
    std::vector<int> ok(points);
    GPU_CHECK(gpuMemcpy(eel_before.data(), d_eel_before, sizeof(double) * eel_before.size(),
                        gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(p_before.data(), d_p_before, sizeof(double) * p_before.size(),
                        gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(sig.data(), d_sig, sizeof(double) * sig.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(p.data(), d_p, sizeof(double) * p.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(Dt.data(), d_Dt, sizeof(double) * Dt.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    for (void* ptr : {static_cast<void*>(d_eel_before), static_cast<void*>(d_p_before),
                      static_cast<void*>(d_sig), static_cast<void*>(d_p), static_cast<void*>(d_Dt),
                      static_cast<void*>(d_ok)})
        GPU_CHECK(gpuFree(ptr));

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
