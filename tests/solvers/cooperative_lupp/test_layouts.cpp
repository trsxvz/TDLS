/// \file
/// \brief Bridge suite: matrix layouts and batched layouts are
/// equivalent.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// In external mode, the CooperativeLUpp solver sees every batched layout
/// through the same (pre-offset pointer, element stride) pair: AoS is
/// stride 1, SoA is stride count, AoSoA is stride W. Config.layout only
/// remaps the flat index of a matrix element. On identical inputs, every
/// combination of the two matrix layouts and the three batched layouts
/// must therefore produce bitwise-identical factored rows, row positions
/// and solutions, for the matrix, the right-hand side and the pivot
/// simultaneously, with both solvers. Row-major SoA under the static
/// solver is the baseline. The batches are solved by groups of CPU
/// threads, through solve_inplace and through factorize + substitute,
/// which reloads the factored rows with the same addressing.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

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

/// \brief Outputs of a whole batch in canonical storage.
struct BatchResult {
    std::vector<double> A; ///< factored rows, row-major, physical order
    std::vector<int> piv;  ///< row positions
    std::vector<double> x; ///< solutions
    std::vector<int> ok;   ///< verdicts
};

/// \brief Solves every system of a batch stored under one matrix layout
/// and one batched layout, and gathers the outputs canonically.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam layout          matrix layout
/// \tparam dynamic         solve with the runtime solver instead of the
///         compile-time one
/// \param[in] batch  the systems, contiguous row-major
/// \param[in] blay   batched layout of every operand
/// \param[in] fused  solve_inplace when true, factorize + substitute
///            otherwise
/// \return the outputs
template<int N, int rows_per_thread, tdls::MatrixLayout layout, bool dynamic = false>
BatchResult solve_batch(const tdls_tests::SystemBatch<double>& batch, const Batch blay,
                        const bool fused) {
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
                A[batch_index(blay, flat(r, c), s, N * N, padded)] = batch.matrix(s)[r * N + c];
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
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c)
                out.A[static_cast<std::size_t>(s) * N * N + r * N + c] =
                    A[batch_index(blay, flat(r, c), s, N * N, padded)];
            out.piv[static_cast<std::size_t>(s) * N + r] = piv[batch_index(blay, r, s, N, padded)];
            out.x[static_cast<std::size_t>(s) * N + r]   = x[batch_index(blay, r, s, N, padded)];
        }
    }
    return out;
}

/// \brief Compares every (solver, matrix layout, batched layout)
/// combination to the static row-major SoA baseline on one reproducible
/// batch.
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<int N, int rows_per_thread>
void layouts_case(const int count, const std::uint64_t seed) {
    const auto batch   = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    constexpr auto row = tdls::MatrixLayout::RowMajor;
    constexpr auto col = tdls::MatrixLayout::ColMajor;
    for (const bool fused : {true, false}) {
        const auto base = solve_batch<N, rows_per_thread, row>(batch, Batch::soa, fused);
        int solved      = 0;
        for (const int v : base.ok)
            solved += v;
        // Floor: every system of the baseline solved.
        TDLS_CHECK(solved == count);
        for (const auto blay : {Batch::soa, Batch::aos, Batch::aosoa}) {
            for (const bool colmajor : {false, true}) {
                if (blay == Batch::soa && !colmajor) continue;
                const auto other = colmajor
                                       ? solve_batch<N, rows_per_thread, col>(batch, blay, fused)
                                       : solve_batch<N, rows_per_thread, row>(batch, blay, fused);
                TDLS_CHECK_BITWISE(base.ok.data(), other.ok.data(), base.ok.size());
                TDLS_CHECK_BITWISE(base.A.data(), other.A.data(), base.A.size());
                TDLS_CHECK_BITWISE(base.piv.data(), other.piv.data(), base.piv.size());
                TDLS_CHECK_BITWISE(base.x.data(), other.x.data(), base.x.size());
            }
            for (const bool colmajor : {false, true}) {
                const auto other =
                    colmajor ? solve_batch<N, rows_per_thread, col, true>(batch, blay, fused)
                             : solve_batch<N, rows_per_thread, row, true>(batch, blay, fused);
                TDLS_CHECK_BITWISE(base.ok.data(), other.ok.data(), base.ok.size());
                TDLS_CHECK_BITWISE(base.A.data(), other.A.data(), base.A.size());
                TDLS_CHECK_BITWISE(base.piv.data(), other.piv.data(), base.piv.size());
                TDLS_CHECK_BITWISE(base.x.data(), other.x.data(), base.x.size());
            }
        }
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
