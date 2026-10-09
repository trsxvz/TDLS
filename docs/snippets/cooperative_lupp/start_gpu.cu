/// \file
/// \brief Documentation snippet: a group of 2 GPU threads solves one
/// system.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <cstdio>

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "gpu_runtime.hpp"

// snippet begin
// 2 rows per thread for a 4 x 4 system: a group of 2 GPU threads solves it
constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

// A, y and ok in GPU memory; one block, whose 2 threads form the group
__global__ void solve(double* A, double* y, int* ok) {
    __shared__ double work[Solver::workspace_size]; // shared by the group
    __shared__ int piv[4];
    const int tx = static_cast<int>(threadIdx.x); // rank of the thread in the group: 0 or 1

    // <false, false, false>: y, piv and A are whole objects, read and written by both
    // threads. No barrier argument: the solver deduces the one of the 2 threads
    const bool solved = Solver::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work);
    if (tx == 0) *ok = solved ? 1 : 0;
}

// on the host: one block of Solver::threads_per_system threads
void launch(double* A, double* y, int* ok) {
    solve<<<1, Solver::threads_per_system>>>(A, y, ok);
}
// snippet end

int main() {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    int ok                   = 0;

    double *d_A = nullptr, *d_y = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_A, sizeof(A)));
    GPU_CHECK(gpuMalloc(&d_y, sizeof(y)));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(ok)));
    GPU_CHECK(gpuMemcpy(d_A, A, sizeof(A), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_y, y, sizeof(y), gpuMemcpyHostToDevice));

    launch(d_A, d_y, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(y, d_y, sizeof(y), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(&ok, d_ok, sizeof(ok), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_A));
    GPU_CHECK(gpuFree(d_y));
    GPU_CHECK(gpuFree(d_ok));
    return ok ? snippets::check(y, expected, 4) : 1;
}
