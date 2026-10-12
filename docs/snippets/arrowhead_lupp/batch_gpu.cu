/// \file
/// \brief Documentation snippet: a batch on GPU, one thread per system.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <cstdio>

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"
#include "gpu_runtime.hpp"

// snippet begin
constexpr auto config = tdls::ArrowheadLUppConfig<double>{
    .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
using Solver = tdls::ArrowheadLUppSolverStatic<double, snippets::arrow::shape, config>;

// one thread per system of the SoA batch; the workspaces in GPU memory too, SoA
__global__ void solve_batch(const int count, double* M, double* work, double* y, int* ok) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= count) return;

    // every operand in the batch, at the stride of the batch size
    ok[s] = Solver::solve_inplace<false, false, false>(M + s, count, work + s, count, y + s, count)
                ? 1
                : 0;
}
// snippet end

int main() {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    using namespace snippets::arrow;
    constexpr int size = Solver::storage_size;
    // entry e of system s at e * count + s
    double A_batch[size * count] = {}, y[n * count] = {};
    int ok[count];
    for (int s = 0; s < count; ++s) {
        double full[n * n], stored[size];
        for (int k = 0; k < n * n; ++k)
            full[k] = matrix(s, k);
        store<Solver::storage_map>(full, stored);
        for (int e = 0; e < size; ++e)
            A_batch[e * count + s] = stored[e];
        for (int i = 0; i < n; ++i)
            y[i * count + s] = rhs(s, i);
    }
    double* d_work = nullptr;
    GPU_CHECK(gpuMalloc(&d_work, sizeof(double) * Solver::workspace_size * count));

    double *d_A = nullptr, *d_y = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_A, sizeof(A_batch)));
    GPU_CHECK(gpuMalloc(&d_y, sizeof(y)));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(ok)));
    GPU_CHECK(gpuMemcpy(d_A, A_batch, sizeof(A_batch), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_y, y, sizeof(y), gpuMemcpyHostToDevice));

    solve_batch<<<1, 64>>>(count, d_A, d_work, d_y, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(y, d_y, sizeof(y), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok, d_ok, sizeof(ok), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_A));
    GPU_CHECK(gpuFree(d_y));
    GPU_CHECK(gpuFree(d_ok));
    GPU_CHECK(gpuFree(d_work));

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s] ? snippets::check(y + s, count, expected, n) : 1;
    return status;
}
