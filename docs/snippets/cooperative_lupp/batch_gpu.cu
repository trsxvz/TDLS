/// \file
/// \brief Documentation snippet: groups of lanes inside a GPU kernel.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <cstdio>

#include <tdls/tdls.hpp>

#include "batch.hpp"
#include "check.hpp"
#include "gpu_runtime.hpp"

// snippet begin
constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;
constexpr int threads = Solver::threads_per_system; // 2 lanes per system

// one group of 2 lanes per system of the SoA batch, as many groups per warp as fit
__global__ void solve_batch(const int count, double* A, double* y, int* ok) {
    // the workspace of every group of the block, in shared memory
    __shared__ double work[64][Solver::workspace_size];
    const int groups     = warpSize / threads;
    const int lane       = static_cast<int>(threadIdx.x) % warpSize;
    const int group      = lane / threads;
    const int tx         = lane % threads; // rank in the group
    const int block_slot = static_cast<int>(threadIdx.x) / warpSize * groups + group;
    const int s = static_cast<int>(blockIdx.x) * (blockDim.x / warpSize) * groups + block_slot;
    if (group >= groups || s >= count) return; // a whole group leaves together

    // matrix and right-hand side in the batch, pivot slice of 2 rows in registers
    int piv[Solver::rows_per_thread];
    // no barrier argument: the solver deduces the barrier of the 2 lanes
    const bool solved = Solver::solve_inplace<false, true, false>(tx, A + s, count, piv, 1, y + s,
                                                                  count, work[block_slot]);
    if (tx == 0) ok[s] = solved ? 1 : 0;
}
// snippet end

int main() {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    using namespace snippets;
    double A[N * N * count], y[N * count];
    int ok[count];
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A[k * count + s] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y[i * count + s] = rhs(s, i);
    }

    double *d_A = nullptr, *d_y = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_A, sizeof(A)));
    GPU_CHECK(gpuMalloc(&d_y, sizeof(y)));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(ok)));
    GPU_CHECK(gpuMemcpy(d_A, A, sizeof(A), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_y, y, sizeof(y), gpuMemcpyHostToDevice));

    solve_batch<<<1, 64>>>(count, d_A, d_y, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(y, d_y, sizeof(y), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok, d_ok, sizeof(ok), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_A));
    GPU_CHECK(gpuFree(d_y));
    GPU_CHECK(gpuFree(d_ok));

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s] ? snippets::check(y + s, count, expected, N) : 1;
    return status;
}
