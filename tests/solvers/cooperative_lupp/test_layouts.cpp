/// \file
/// \brief Bridge suite: matrix layouts and batched layouts.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// In external mode, the CooperativeLUpp solver sees every batched layout
/// through the same (pre-offset pointer, element stride) pair: AoS is
/// stride 1, SoA is stride count, AoSoA is stride W. Under one matrix
/// layout, the three batched layouts and both solvers must therefore
/// produce bitwise-identical factored matrices, row positions and
/// solutions, for the matrix, the right-hand side and the pivot
/// simultaneously; static SoA is the baseline of each matrix layout.
///
/// Both matrix layouts factor S, the stored matrix read column-major, and
/// solve S x = b (column-major, S = A) or S^T x = b (row-major, S = A^T).
/// A matrix stored row-major and its transpose stored column-major are the
/// same array: the two configurations must produce the same factored
/// array and the same row positions, bit for bit, while their solutions
/// solve A x = b and A^T x = b, each checked by its backward error. The
/// batches are solved by groups of CPU threads, through solve_inplace and
/// through factorize + substitute, which reloads the factored rows with
/// the same addressing.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Identifies one batched layout.
enum class Batch { soa, aos, aosoa };

/// \brief Buffer index of (element, system) under the given batched layout.
/// \param[in] layout      the batched layout
/// \param[in] element     flat element index inside the object
/// \param[in] system      system index
/// \param[in] object_size elements per object
/// \param[in] count       systems in the buffer
/// \return the buffer index
std::size_t batch_index(const Batch layout, const std::size_t element, const std::size_t system,
                        const std::size_t object_size, const std::size_t count) {
    switch (layout) {
    case Batch::soa:
        return tdls_tests::soa_index(element, system, count);
    case Batch::aos:
        return tdls_tests::aos_index(element, system, object_size);
    default:
        return tdls_tests::aosoa_index(element, system, object_size, 4);
    }
}

/// \brief Element stride of the given batched layout.
/// \param[in] layout the batched layout
/// \param[in] count  systems in the buffer
/// \return the stride
int batch_stride(const Batch layout, const int count) {
    switch (layout) {
    case Batch::soa:
        return count;
    case Batch::aos:
        return 1;
    default:
        return 4;
    }
}

/// \brief Offset of the first element of a system under the given layout.
/// \param[in] layout      the batched layout
/// \param[in] system      system index
/// \param[in] object_size elements per object
/// \param[in] count       systems in the buffer
/// \return the offset the caller pre-applies to the base pointer
std::size_t batch_offset(const Batch layout, const std::size_t system,
                         const std::size_t object_size, const std::size_t count) {
    return batch_index(layout, 0, system, object_size, count);
}

/// \brief Outputs of a whole batch.
struct BatchResult {
    std::vector<double> A; ///< factored matrices, as stored (flat index)
    std::vector<int> piv;  ///< row positions
    std::vector<double> x; ///< solutions
    std::vector<int> ok;   ///< verdicts
};

/// \brief Solves every system of a batch stored under one matrix layout
/// and one batched layout, and gathers the outputs.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam layout          matrix layout
/// \tparam dynamic         solve with the runtime solver instead of the
///         compile-time one
/// \param[in] batch     the systems, contiguous row-major
/// \param[in] blay      batched layout of every operand
/// \param[in] fused     solve_inplace when true, factorize + substitute
///            otherwise
/// \param[in] transpose solve with the transpose of each matrix
/// \return the outputs
template<int N, int rows_per_thread, tdls::MatrixLayout layout, bool dynamic = false>
BatchResult solve_batch(const tdls_tests::SystemBatch<double>& batch, const Batch blay,
                        const bool fused, const bool transpose) {
    constexpr auto config = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = rows_per_thread,
        .unroll_loops    = tdls_tests::test_unroll<N, rows_per_thread>,
        .layout          = layout};
    using Solver     = tdls::CooperativeLUppSolverStatic<double, N, config>;
    using Dynamic    = tdls::CooperativeLUppSolverDynamic<double, config>;
    const int count  = batch.count;
    const int padded = (count + 3) / 4 * 4; // full AoSoA blocks
    const auto flat  = [](const int r, const int c) {
        return layout == tdls::MatrixLayout::RowMajor ? std::size_t(r) * N + c
                                                      : std::size_t(c) * N + r;
    };

    std::vector<double> A(static_cast<std::size_t>(padded) * N * N, 0.0);
    std::vector<double> b(static_cast<std::size_t>(padded) * N, 0.0);
    std::vector<double> x(static_cast<std::size_t>(padded) * N, 0.0);
    std::vector<int> piv(static_cast<std::size_t>(padded) * N, 0);
    for (int s = 0; s < count; ++s) {
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c)
                A[batch_index(blay, flat(r, c), s, N * N, padded)] =
                    transpose ? batch.matrix(s)[c * N + r] : batch.matrix(s)[r * N + c];
            b[batch_index(blay, r, s, N, padded)] = batch.rhs(s)[r];
            x[batch_index(blay, r, s, N, padded)] = batch.rhs(s)[r];
        }
    }

    const int A_stride   = batch_stride(blay, padded);
    const int vec_stride = batch_stride(blay, padded);
    BatchResult out{std::vector<double>(static_cast<std::size_t>(count) * N * N),
                    std::vector<int>(static_cast<std::size_t>(count) * N),
                    std::vector<double>(static_cast<std::size_t>(count) * N),
                    std::vector<int>(count)};
    std::vector<double> work(Solver::workspace_size);
    for (int s = 0; s < count; ++s) {
        double* As = A.data() + batch_offset(blay, s, N * N, padded);
        double* bs = b.data() + batch_offset(blay, s, N, padded);
        double* xs = x.data() + batch_offset(blay, s, N, padded);
        int* ps    = piv.data() + batch_offset(blay, s, N, padded);
        tdls_tests::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
            bool ok = false;
            if constexpr (dynamic) {
                if (fused) {
                    ok = Dynamic::solve_inplace(N, tx, As, A_stride, ps, vec_stride, xs, vec_stride,
                                                work.data(), sync);
                } else {
                    ok = Dynamic::factorize(N, tx, As, A_stride, ps, vec_stride, work.data(), sync);
                    Dynamic::substitute(N, tx, As, A_stride, ps, vec_stride, bs, xs, vec_stride,
                                        work.data(), sync);
                }
            } else if (fused) {
                ok = Solver::template solve_inplace<false, false, false>(
                    tx, As, A_stride, ps, vec_stride, xs, vec_stride, work.data(), sync);
            } else {
                ok = Solver::template factorize<false, false>(tx, As, A_stride, ps, vec_stride,
                                                              work.data(), sync);
                Solver::template substitute<false, false, false>(
                    tx, As, A_stride, ps, vec_stride, bs, xs, vec_stride, work.data(), sync);
            }
            if (tx == 0) out.ok[s] = ok ? 1 : 0;
        });
        for (int e = 0; e < N * N; ++e)
            out.A[static_cast<std::size_t>(s) * N * N + e] =
                A[batch_index(blay, e, s, N * N, padded)];
        for (int r = 0; r < N; ++r) {
            out.piv[static_cast<std::size_t>(s) * N + r] = piv[batch_index(blay, r, s, N, padded)];
            out.x[static_cast<std::size_t>(s) * N + r]   = x[batch_index(blay, r, s, N, padded)];
        }
    }
    return out;
}

/// \brief Requires two batch results to agree bit for bit.
/// \param[in] base  reference result
/// \param[in] other compared result
void check_same(const BatchResult& base, const BatchResult& other) {
    TDLS_CHECK_BITWISE(base.ok.data(), other.ok.data(), base.ok.size());
    TDLS_CHECK_BITWISE(base.A.data(), other.A.data(), base.A.size());
    TDLS_CHECK_BITWISE(base.piv.data(), other.piv.data(), base.piv.size());
    TDLS_CHECK_BITWISE(base.x.data(), other.x.data(), base.x.size());
}

/// \brief Compares every (solver, batched layout) combination of one
/// matrix layout to its static SoA baseline, and returns the baseline.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam layout          matrix layout
/// \param[in] batch     the systems
/// \param[in] fused     solve_inplace when true, factorize + substitute
///            otherwise
/// \param[in] transpose solve with the transpose of each matrix
/// \return the baseline
template<int N, int rows_per_thread, tdls::MatrixLayout layout>
BatchResult batched_layouts(const tdls_tests::SystemBatch<double>& batch, const bool fused,
                            const bool transpose) {
    const auto base = solve_batch<N, rows_per_thread, layout>(batch, Batch::soa, fused, transpose);
    int solved      = 0;
    for (const int v : base.ok)
        solved += v;
    // Floor: every system of the baseline solved.
    TDLS_CHECK(solved == batch.count);
    for (const auto blay : {Batch::soa, Batch::aos, Batch::aosoa}) {
        if (blay != Batch::soa)
            check_same(base,
                       solve_batch<N, rows_per_thread, layout>(batch, blay, fused, transpose));
        check_same(base,
                   solve_batch<N, rows_per_thread, layout, true>(batch, blay, fused, transpose));
    }
    return base;
}

/// \brief Runs both bridges on one reproducible batch: the batched layouts
/// under each matrix layout, and the row-major solve of A against the
/// column-major solve of A^T, which factor the same array.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<int N, int rows_per_thread>
void layouts_case(const int count, const std::uint64_t seed) {
    const auto batch = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    for (const bool fused : {true, false}) {
        const auto row =
            batched_layouts<N, rows_per_thread, tdls::MatrixLayout::RowMajor>(batch, fused, false);
        const auto col =
            batched_layouts<N, rows_per_thread, tdls::MatrixLayout::ColMajor>(batch, fused, true);
        TDLS_CHECK_BITWISE(row.ok.data(), col.ok.data(), row.ok.size());
        TDLS_CHECK_BITWISE(row.A.data(), col.A.data(), row.A.size());
        TDLS_CHECK_BITWISE(row.piv.data(), col.piv.data(), row.piv.size());
        double be_row = 0.0, be_col = 0.0;
        std::vector<double> At(static_cast<std::size_t>(N) * N);
        for (int s = 0; s < count; ++s) {
            const double* A0 = batch.matrix(s);
            for (int r = 0; r < N; ++r)
                for (int c = 0; c < N; ++c)
                    At[static_cast<std::size_t>(c) * N + r] = A0[r * N + c];
            be_row = std::max(
                be_row, tdls_tests::backward_error(
                            A0, row.x.data() + static_cast<std::size_t>(s) * N, batch.rhs(s), N));
            be_col = std::max(be_col, tdls_tests::backward_error(
                                          At.data(), col.x.data() + static_cast<std::size_t>(s) * N,
                                          batch.rhs(s), N));
        }
        TDLS_CHECK_LE(be_row, 1e-13);
        TDLS_CHECK_LE(be_col, 1e-13);
    }
}

} // namespace

TDLS_TEST_CASE("cooperativelupp/bridge/layouts/double/N=12,rows_per_thread=12") {
    layouts_case<12, 12>(30, 650112);
}
TDLS_TEST_CASE("cooperativelupp/bridge/layouts/double/N=12,rows_per_thread=3") {
    layouts_case<12, 3>(10, 650203);
}
TDLS_TEST_CASE("cooperativelupp/bridge/layouts/double/N=7,rows_per_thread=3") {
    layouts_case<7, 3>(10, 650703);
}
TDLS_TEST_CASE("cooperativelupp/bridge/layouts/double/N=13,rows_per_thread=5") {
    layouts_case<13, 5>(10, 650305);
}

TDLS_TEST_MAIN
