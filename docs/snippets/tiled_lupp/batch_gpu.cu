/// \file
/// \brief Documentation snippet: inside a GPU kernel, in the common CUDA/HIP dialect.
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
using Solver =
    tdls::TiledLUppSolverStatic<double, 4, tdls::TiledLUppConfig<double>{.tile_size = 2}>;

// one thread per system of the SoA batch
__global__ void solve_batch(const int count, double* A, double* y, int* ok) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= count) return;

    // matrix and right-hand side in the batch, pivot in registers
    int piv[4];
    ok[s] = Solver::solve_inplace<false, true, false>(A + s, count, piv, 1, y + s, count) ? 1 : 0;
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

    solve_batch<<<1, 32>>>(count, d_A, d_y, d_ok);
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
