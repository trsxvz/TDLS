/// \file
/// \brief Example: parameter sweep of Love's integral equation on GPU,
/// one dense Nystroem system per plate separation held in shared memory
/// and solved by a group of lanes with the runtime CooperativeLUpp
/// solver at unit stride.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Companion of the sequential and OpenMP integral_equation examples at
/// the GPU scale: the capacitor potential is computed for a large sweep
/// of plate separations d, one group of lanes per instance.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched, so the dimension is a runtime value and
/// CooperativeLUppSolverDynamic applies.
///
/// Why groups of lanes: with four rows per lane, an instance takes a
/// group of ceil(n / 4) lanes, nine at the default resolution, and a
/// warp solves as many instances as fit in its 32 lanes. The barrier of
/// a group, deduced by the solver and returned by make_sync for the
/// exchanges of the example, is a warp barrier on its lanes only.
///
/// Placement on GPU: the runtime solver has no residency booleans,
/// because a runtime dimension forces runtime indexing; the placement
/// of the operands is instead a property of the pointers and strides
/// handed to it. Here each group assembles its system in shared memory,
/// one row per lane at a time, and solves at unit stride. The storage
/// footprint is resident groups times one system, not batch times one
/// system: the batch of matrices never exists in device memory, only
/// the per instance results do. The plate separation of an instance is
/// one ramp evaluation away.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "love.hpp"

namespace {

using namespace love;

constexpr int instances = 1 << 17; ///< plate separations in the sweep
constexpr int max_n     = 40;      ///< bound of the runtime resolution

/// \brief One instance per group of lanes, one warp per block:
/// assembles its Nystroem system in shared memory, factorizes it once,
/// substitutes the manufactured and the physical right-hand sides, and
/// writes the manufactured error and the physical potential.
/// \tparam Solver a GroupSolver whose group fits in 32 lanes
/// \param[in]  n     quadrature resolution
/// \param[out] err   per-instance error against the manufactured solution
/// \param[out] u_mid per-instance potential at the central node
/// \param[out] ok    per-instance success flags
template<typename Solver>
__global__ void capacitor_sweep(const int n, double* err, double* u_mid, int* ok) {
    extern __shared__ double shared[];
    const int threads = Solver::threads_per_system(n);
    const int groups  = 32 / threads;
    const int group   = static_cast<int>(threadIdx.x) / threads;
    const int tx      = static_cast<int>(threadIdx.x) % threads;
    const int t       = static_cast<int>(blockIdx.x) * groups + group;
    // The lanes past the last group, and the groups past the sweep,
    // leave at once: a barrier only involves the lanes of its group.
    if (group >= groups || t >= instances) return;

    // Shared storage of the group, unit strides: A, g, u and the
    // workspace, then the pivots after those of every group.
    const int doubles = n * n + 2 * n + Solver::workspace_size(n);
    double* A         = shared + group * doubles;
    double* g         = A + n * n;
    double* u         = g + n;
    double* work      = u + n;
    int* piv          = reinterpret_cast<int*>(shared + groups * doubles) + group * n;
    auto sync         = Solver::make_sync(n, work);

    // One factorization, two right-hand sides.
    double err_t = 0, u_mid_t = 0;
    const bool ok_t = capacitor_group<Solver>(n, tx, sync, separation(t, instances), A, 1, piv, 1,
                                              g, u, 1, work, err_t, u_mid_t);

    // Every lane of the group holds the same results.
    if (tx != 0) return;
    err[t]   = ok_t ? err_t : 1.0;
    u_mid[t] = ok_t ? u_mid_t : 0.0;
    ok[t]    = ok_t ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    // Four rows per lane: an instance takes ceil(n / 4) lanes.
    using Solver = GroupSolver<4>;

    // The quadrature resolution comes from outside the program, bounded
    // here by the shared memory of a block; the default is odd so that
    // x = 0 is a node.
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

    // One warp per block, as many groups as fit in it. At max_n, the
    // three systems of a block take 43 KB of shared memory, under the
    // 48 KB every device grants without opt-in.
    const int groups = 32 / Solver::threads_per_system(n);
    const std::size_t shared =
        groups * (sizeof(double) * (n * n + 2 * n + Solver::workspace_size(n)) + sizeof(int) * n);
    capacitor_sweep<Solver>
        <<<(instances + groups - 1) / groups, 32, shared>>>(n, d_err, d_umid, d_ok);
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
