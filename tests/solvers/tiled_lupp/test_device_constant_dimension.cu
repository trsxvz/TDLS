/// \file
/// \brief Device suite: TiledLUppSolverDynamic called from a kernel with
/// dimensions known at compile time.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// nvcc used to miscompile the runtime solver for Blackwell (sm_100 and
/// later) when the dimension passed by the caller is a compile-time
/// constant that propagates into its loops (TDLS issue #16): wrong
/// solutions, for single and multiple right-hand sides, the operands in
/// thread-local memory. Each kernel factors and solves one system with
/// constant dimensions through every entry point and compares the
/// factors, the pivots and the solutions with the same calls on the host,
/// both layouts, both layouts of a block of right-hand sides. Built with
/// the GPU example toolchain; returns 77 without a device.

#include <cmath>
#include <cstdio>

#include <tdls/tdls.hpp>

#include "gpu_runtime.hpp"

namespace {

/// \brief Configuration of one layout.
/// \tparam L layout of the matrix
/// \return the configuration
template<tdls::MatrixLayout L>
TDLS_HOST_DEVICE constexpr tdls::TiledLUppConfig<double> layout_config() {
    tdls::TiledLUppConfig<double> c;
    c.layout = L;
    return c;
}

/// \brief Entry (i, j) of the matrix, diagonally dominant.
/// \param[in] i row
/// \param[in] j column
/// \param[in] n dimension
/// \return the entry
TDLS_HOST_DEVICE double entry(const int i, const int j, const int n) {
    return (i == j ? n + 2.0 : 0.0) + 1.0 / (1 + i + 2 * j) + 0.1 * ((i * 7 + j * 3) % 5);
}

/// \brief Entry i of right-hand side w.
/// \param[in] i entry
/// \param[in] w right-hand side
/// \return the entry
TDLS_HOST_DEVICE double rhs(const int i, const int w) {
    return 0.3 * ((i * 3 + w * 5) % 7) - 1.0;
}

/// \brief Number of result slots of a case, n * nrhs entries each.
constexpr int slots = 11;

/// \brief Copies a block of right-hand sides into one result slot.
/// \param[out] out  the slot, column after column
/// \param[in]  x    the block
/// \param[in]  n    dimension
/// \param[in]  nrhs number of right-hand sides
/// \param[in]  rs   element stride of the block
/// \param[in]  xs   stride between the columns of the block
TDLS_HOST_DEVICE void store(double* out, const double* x, const int n, const int nrhs, const int rs,
                            const std::size_t xs) {
    for (int w = 0; w < nrhs; ++w)
        for (int i = 0; i < n; ++i)
            out[w * n + i] = x[i * rs + w * xs];
}

/// \brief Loads the right-hand sides into a block.
/// \param[out] b    the block
/// \param[in]  n    dimension
/// \param[in]  nrhs number of right-hand sides
/// \param[in]  rs   element stride of the block
/// \param[in]  xs   stride between the columns of the block
TDLS_HOST_DEVICE void load(double* b, const int n, const int nrhs, const int rs,
                           const std::size_t xs) {
    for (int w = 0; w < nrhs; ++w)
        for (int i = 0; i < n; ++i)
            b[i * rs + w * xs] = rhs(i, w);
}

/// \brief Copies n entries.
/// \param[out] to   destination
/// \param[in]  from source
/// \param[in]  n    number of entries
TDLS_HOST_DEVICE void copy(double* to, const double* from, const int n) {
    for (int e = 0; e < n; ++e)
        to[e] = from[e];
}

/// \brief The results of one case, written to out: the slots of the
/// single-column entry points (substitute, substitute_canonical,
/// substitute_inplace, solve, solve_inplace), column after column, then
/// those of the multi-right-hand-side ones (substitute_multirhs,
/// substitute_canonical_multirhs, substitute_inplace_multirhs, the two
/// stages, solve_multirhs, solve_inplace_multirhs), then the factors,
/// the pivots and 1, the mark of a complete run.
/// \tparam L     layout of the matrix
/// \tparam n     dimension
/// \tparam nrhs  number of right-hand sides
/// \tparam rows  block of right-hand sides by rows (true) or by columns
/// \param[out] out slots * n * nrhs + n * n + n + 1 results
template<tdls::MatrixLayout L, int n, int nrhs, bool rows>
TDLS_HOST_DEVICE void solve_case(double* out) {
    using D                  = tdls::TiledLUppSolverDynamic<double, layout_config<L>()>;
    constexpr int rs         = rows ? nrhs : 1;
    constexpr std::size_t xs = rows ? 1 : n;
    constexpr int block      = n * nrhs;
    double A0[n * n], A[n * n], F[n * n], b[block], x[block], col[n], bc[n];
    int piv[n], fpiv[n];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A0[i * n + j] = entry(i, j, n);
    out[slots * block + n * n + n] = 0.0;
    copy(F, A0, n * n);
    if (!D::factorize(n, F, 1, piv, 1)) return;
    for (int w = 0; w < nrhs; ++w) {
        double* slot = out + w * n;
        for (int i = 0; i < n; ++i)
            bc[i] = rhs(i, w);
        D::substitute(n, F, 1, piv, 1, bc, col, 1);
        copy(slot, col, n);
        D::substitute_canonical(n, F, 1, piv, 1, w, col, 1);
        copy(slot + block, col, n);
        D::substitute_inplace(n, F, 1, piv, 1, bc, 1);
        copy(slot + 2 * block, bc, n);
        for (int i = 0; i < n; ++i)
            bc[i] = rhs(i, w);
        copy(A, A0, n * n);
        if (!D::solve(n, A, 1, fpiv, 1, bc, col, 1)) return;
        copy(slot + 3 * block, col, n);
        copy(A, A0, n * n);
        if (!D::solve_inplace(n, A, 1, fpiv, 1, bc, 1)) return;
        copy(slot + 4 * block, bc, n);
    }
    load(b, n, nrhs, rs, xs);
    D::substitute_multirhs(n, nrhs, F, 1, piv, 1, b, x, rs, xs);
    store(out + 5 * block, x, n, nrhs, rs, xs);
    D::substitute_canonical_multirhs(n, nrhs, F, 1, piv, 1, 0, x, rs, xs);
    store(out + 6 * block, x, n, nrhs, rs, xs);
    D::substitute_inplace_multirhs(n, nrhs, F, 1, piv, 1, b, rs, xs);
    store(out + 7 * block, b, n, nrhs, rs, xs);
    load(b, n, nrhs, rs, xs);
    D::forward_inplace_multirhs(n, nrhs, F, 1, piv, 1, b, rs, xs);
    D::backward_inplace_multirhs(n, nrhs, F, 1, piv, 1, b, rs, xs);
    store(out + 8 * block, b, n, nrhs, rs, xs);
    load(b, n, nrhs, rs, xs);
    copy(A, A0, n * n);
    if (!D::solve_multirhs(n, nrhs, A, 1, fpiv, 1, b, x, rs, xs)) return;
    store(out + 9 * block, x, n, nrhs, rs, xs);
    copy(A, A0, n * n);
    if (!D::solve_inplace_multirhs(n, nrhs, A, 1, fpiv, 1, b, rs, xs)) return;
    store(out + 10 * block, b, n, nrhs, rs, xs);
    copy(out + slots * block, F, n * n);
    for (int i = 0; i < n; ++i)
        out[slots * block + n * n + i] = piv[i];
    out[slots * block + n * n + n] = 1.0;
}

/// \brief One thread solves one case on the device.
/// \tparam L    layout of the matrix
/// \tparam n    dimension
/// \tparam nrhs number of right-hand sides
/// \tparam rows block of right-hand sides by rows or by columns
/// \param[out] out the results
template<tdls::MatrixLayout L, int n, int nrhs, bool rows>
__global__ void device_case(double* out) {
    solve_case<L, n, nrhs, rows>(out);
}

/// \brief Runs one case on the device and on the host, and compares.
/// \tparam L    layout of the matrix
/// \tparam n    dimension
/// \tparam nrhs number of right-hand sides
/// \tparam rows block of right-hand sides by rows or by columns
/// \return the number of entries off by more than 1e-12
template<tdls::MatrixLayout L, int n, int nrhs, bool rows>
int run_case() {
    constexpr int size = slots * n * nrhs + n * n + n + 1;
    double* d_out      = nullptr;
    GPU_CHECK(gpuMalloc(&d_out, sizeof(double) * size));
    device_case<L, n, nrhs, rows><<<1, 1>>>(d_out);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    double device[size], host[size];
    GPU_CHECK(gpuMemcpy(device, d_out, sizeof device, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_out));
    solve_case<L, n, nrhs, rows>(host);
    int wrong = device[size - 1] == 1.0 && host[size - 1] == 1.0 ? 0 : 1;
    for (int e = 0; e < size; ++e)
        if (!(std::fabs(device[e] - host[e]) <= 1e-12)) ++wrong;
    if (wrong > 0)
        std::printf("FAIL %s n=%d nrhs=%d block by %s: %d wrong entries\n",
                    L == tdls::MatrixLayout::RowMajor ? "row-major" : "column-major", n, nrhs,
                    rows ? "rows" : "columns", wrong);
    return wrong;
}

/// \brief Every case of one layout.
/// \tparam L layout of the matrix
/// \return the number of wrong entries
template<tdls::MatrixLayout L>
int run_layout() {
    return run_case<L, 4, 2, true>() + run_case<L, 5, 2, true>() + run_case<L, 5, 3, false>() +
           run_case<L, 7, 2, false>() + run_case<L, 13, 4, true>() + run_case<L, 13, 4, false>();
}

} // namespace

int main() {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }
    const int wrong =
        run_layout<tdls::MatrixLayout::RowMajor>() + run_layout<tdls::MatrixLayout::ColMajor>();
    std::printf("%s\n", wrong == 0 ? "device constant dimension: all cases match the host"
                                   : "device constant dimension: FAILED");
    return wrong == 0 ? 0 : 1;
}
