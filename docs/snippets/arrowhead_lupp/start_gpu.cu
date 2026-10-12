/// \file
/// \brief Documentation snippet: one GPU thread per system.
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
constexpr auto shape = tdls::ArrowheadShape{.border = 2, .block = 2, .blocks = 2};
using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, tdls::ArrowheadLUppConfig<double>{}>;

// A, y and ok in GPU memory, system s at A + s * 36 and y + s * 6; one thread per system
__global__ void solve(const int count, double* A, double* y, int* ok) {
    const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= count) return;

    // <false, true, false>: y and A in GPU memory, the workspace in a local array
    double work[Solver::workspace_size];
    ok[s] = Solver::solve_inplace<false, true, false>(A + s * 36, 1, work, 1, y + s * 6, 1) ? 1 : 0;
}

// on the host: one thread per system, in blocks of 64
void launch(const int count, double* A, double* y, int* ok) {
    solve<<<(count + 63) / 64, 64>>>(count, A, y, ok);
}
// snippet end

int main() {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    using namespace snippets::arrow;
    // system s at A + s * n * n and y + s * n, one after the other
    double A_batch[n * n * count], y[n * count];
    int ok[count];
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < n * n; ++k)
            A_batch[s * n * n + k] = matrix(s, k);
        for (int i = 0; i < n; ++i)
            y[s * n + i] = rhs(s, i);
    }

    double *d_A = nullptr, *d_y = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_A, sizeof(A_batch)));
    GPU_CHECK(gpuMalloc(&d_y, sizeof(y)));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(ok)));
    GPU_CHECK(gpuMemcpy(d_A, A_batch, sizeof(A_batch), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_y, y, sizeof(y), gpuMemcpyHostToDevice));

    launch(count, d_A, d_y, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(y, d_y, sizeof(y), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok, d_ok, sizeof(ok), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_A));
    GPU_CHECK(gpuFree(d_y));
    GPU_CHECK(gpuFree(d_ok));

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s] ? snippets::check(y + s * n, expected, n) : 1;
    return status;
}
