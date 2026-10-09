/// \file
/// \brief Example: reaction substep of an operator-splitting scheme on
/// GPU, one independent stiff implicit step chain per cell, each linear
/// system shared by a group of lanes and held in their registers through
/// the internal residencies of the compile-time CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: exactly as in the
/// sequential and OpenMP implicit_ode examples, the Newton systems of a
/// Radau IIA step have size N = stages x species, fixed by the method
/// and by the chemical mechanism. N = 9 is a property of the program.
///
/// Why groups of lanes: with three rows per lane, a cell takes a group
/// of three lanes, and 32 lanes solve ten cells. Each lane holds three
/// rows of the Newton matrix instead of nine. The barrier of a group is
/// a warp barrier on its lanes only, so the groups of a warp iterate
/// their Newton loops independently.
///
/// Why the residencies are internal: each lane passes residency
/// booleans set to true, so the solver indexes its rows of the matrix,
/// their pivot entries and their right-hand-side entries with
/// compile-time strides, and the system lives in the registers of the
/// group. The workspaces live in shared memory. Device memory only
/// holds the per-cell inputs (a temperature factor) and outputs (the
/// cell state): the batch of matrices is never materialized, so the
/// batch size is bounded by compute capacity, not by device memory.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "hash01.hpp"
#include "robertson.hpp"

namespace {

using namespace robertson;

constexpr int block = 256; ///< threads per block

/// \brief One cell per group of lanes: reads the cell inputs,
/// integrates the chemistry with the rows of the Newton systems in
/// registers, writes the final state (SoA layout, coalesced) and a
/// success flag.
/// \tparam Solver a GroupSolver whose group fits in 32 lanes
/// \param[in]  cells number of cells
/// \param[in]  steps number of Radau steps per cell
/// \param[in]  h     time step
/// \param[in]  theta per-cell temperature factors
/// \param[out] state final states, species-major SoA of extent cells
/// \param[out] ok    per-cell success flags
template<typename Solver>
__global__ void reaction_substep(const int cells, const int steps, const double h,
                                 const double* theta, double* state, int* ok) {
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
    if (group >= groups || t >= cells) return;
    double* work = workspaces + (pack * groups + group) * Solver::workspace_size;
    auto sync    = [=] { gpu_group_sync(group * threads, threads); };
    auto any     = [](const bool flag) { return flag; };

    double butcher[stages][stages];
    radau_butcher(butcher);

    // Per-cell inputs: fresh mixture and the cell temperature factor.
    double y[species] = {1.0, 0.0, 0.0};

    // The rows of the lane of the Newton systems, in registers.
    double M[Solver::rows_per_thread * N], r[Solver::rows_per_thread], dz[Solver::rows_per_thread];
    int piv[Solver::rows_per_thread];
    bool success = true;
    for (int s = 0; s < steps && success; ++s)
        success = radau_step_group<true, true, true, Solver>(tx, sync, any, butcher, theta[t], h, y,
                                                             M, 1, piv, 1, r, dz, 1, work);

    // Every lane of the group holds the same state.
    if (tx != 0) return;

    for (int a = 0; a < species; ++a)
        state[static_cast<std::size_t>(a) * cells + t] = y[a];
    ok[t] = success ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    // Three rows per lane: a cell takes a group of three lanes.
    using Solver = GroupSolver<3, true>;

    // Large batch: device memory only holds 4 doubles per cell, the
    // systems themselves never leave the registers.
    const int cells = argc > 1 ? std::atoi(argv[1]) : 1 << 21;
    if (cells < 1) {
        std::printf("usage: %s [number of cells >= 1]\n", argv[0]);
        return 1;
    }
    const double h  = 1e-3; // time step of the reaction substep
    const int steps = 2;    // integrate each cell to t = 0.002

    // Per-cell temperature factors in [0.5, 2], log-uniform across the
    // batch, synthesized on the host as the input data of the substep.
    std::vector<double> theta(cells);
    for (int c = 0; c < cells; ++c)
        theta[c] = 0.5 * std::pow(4.0, hash01(static_cast<unsigned long long>(c)));

    double *d_theta = nullptr, *d_state = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_theta, sizeof(double) * cells));
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * species * cells));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * cells));
    GPU_CHECK(
        gpuMemcpy(d_theta, theta.data(), sizeof(double) * theta.size(), gpuMemcpyHostToDevice));

    const int per_block = block / 32 * (32 / Solver::threads_per_system);
    reaction_substep<Solver>
        <<<(cells + per_block - 1) / per_block, block>>>(cells, steps, h, d_theta, d_state, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> state(static_cast<std::size_t>(species) * cells);
    std::vector<int> ok(cells);
    GPU_CHECK(
        gpuMemcpy(state.data(), d_state, sizeof(double) * state.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_theta));
    GPU_CHECK(gpuFree(d_state));
    GPU_CHECK(gpuFree(d_ok));

    // The Robertson system conserves the total mass exactly, and the
    // trajectory stays a physical concentration vector.
    int failures = 0, unphysical = 0;
    double mass_err = 0.0;
    for (int c = 0; c < cells; ++c) {
        if (!ok[c]) ++failures;
        const double y0 = state[c];
        const double y1 = state[static_cast<std::size_t>(cells) + c];
        const double y2 = state[static_cast<std::size_t>(2) * cells + c];
        mass_err        = std::fmax(mass_err, std::fabs(y0 + y1 + y2 - 1.0));
        if (!(y0 > 0.0 && y0 < 1.0 && y1 > 0.0 && y1 < 1e-3 && y2 > 0.0 && y2 < 1.0)) ++unphysical;
    }

    std::printf("cells = %d, max mass error = %.3e\n", cells, mass_err);
    return failures == 0 && unphysical == 0 && mass_err < 1e-10 ? 0 : 1;
}
