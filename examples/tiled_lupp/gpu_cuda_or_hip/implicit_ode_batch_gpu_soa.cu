/// \file
/// \brief Example: reaction substep of an operator-splitting scheme on
/// GPU, with the linear systems materialized in device memory in SoA
/// layout and solved through the external residencies of the
/// compile-time TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Companion of implicit_ode_batch_gpu.cu: same physics, same method,
/// opposite residency choice. Here the residency booleans are false,
/// so the matrices, pivots and right-hand sides live in device memory
/// and the solver walks them with a runtime element stride. The batch
/// is stored species-of-arrays: element k of system t sits at
/// base[k * cells + t], so consecutive threads touch consecutive
/// addresses and every access is coalesced.
///
/// External residency is the right choice when the systems cannot or
/// should not stay in registers: larger dimensions, register pressure
/// throttling occupancy, or matrices produced by a separate assembly
/// kernel. The price is that the batch now occupies device memory,
/// which bounds its size; the register variant has no such bound.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu_runtime.hpp"
#include "hash01.hpp"
#include "robertson.hpp"

namespace {

using namespace robertson;

/// \brief One cell per thread: integrates the chemistry with every
/// solver operand in the SoA batch (external residencies), writes the
/// final state and a success flag.
/// \param[in]  cells number of cells
/// \param[in]  steps number of Radau steps per cell
/// \param[in]  h     time step
/// \param[in]  theta per-cell temperature factors
/// \param[out] state final states, species-major SoA of extent cells
/// \param[out] M     matrix batch, element stride cells
/// \param[out] piv   pivot batch, element stride cells
/// \param[out] r     residual batch, element stride cells
/// \param[out] dz    correction batch, element stride cells
/// \param[out] ok    per-cell success flags
__global__ void reaction_substep(const int cells, const int steps, const double h,
                                 const double* theta, double* state, double* M, int* piv, double* r,
                                 double* dz, int* ok) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= cells) return;

    double butcher[stages][stages];
    radau_butcher(butcher);

    // Per-cell inputs: fresh mixture and the cell temperature factor;
    // the state fits in registers, the linear algebra stays in the batch.
    double y[species] = {1.0, 0.0, 0.0};

    bool success = true;
    for (int s = 0; s < steps && success; ++s)
        success = radau_step<false, false, false>(butcher, theta[t], h, y, M + t, cells, piv + t,
                                                  cells, r + t, dz + t, cells);

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

    // The batch is materialized in device memory (about 850 bytes per
    // cell), so its size is bounded by the device capacity.
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

    double *d_theta = nullptr, *d_state = nullptr, *d_M = nullptr, *d_r = nullptr, *d_dz = nullptr;
    int *d_piv = nullptr, *d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_theta, sizeof(double) * cells));
    GPU_CHECK(
        gpuMemcpy(d_theta, theta.data(), sizeof(double) * theta.size(), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * species * cells));
    GPU_CHECK(gpuMalloc(&d_M, sizeof(double) * N * N * cells));
    GPU_CHECK(gpuMalloc(&d_r, sizeof(double) * N * cells));
    GPU_CHECK(gpuMalloc(&d_dz, sizeof(double) * N * cells));
    GPU_CHECK(gpuMalloc(&d_piv, sizeof(int) * N * cells));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * cells));

    const int block = 256;
    reaction_substep<<<(cells + block - 1) / block, block>>>(cells, steps, h, d_theta, d_state, d_M,
                                                             d_piv, d_r, d_dz, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> state(static_cast<std::size_t>(species) * cells);
    std::vector<int> ok(cells);
    GPU_CHECK(
        gpuMemcpy(state.data(), d_state, sizeof(double) * state.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    for (void* p : {static_cast<void*>(d_theta), static_cast<void*>(d_state),
                    static_cast<void*>(d_M), static_cast<void*>(d_r), static_cast<void*>(d_dz),
                    static_cast<void*>(d_piv), static_cast<void*>(d_ok)})
        GPU_CHECK(gpuFree(p));

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
