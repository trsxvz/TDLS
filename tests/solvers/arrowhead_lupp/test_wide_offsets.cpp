/// \file
/// \brief Bridge suite: element offsets beyond 2^32.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The offset of a strided element, index times stride, is computed in
/// 64-bit arithmetic: a batch may hold more than 2^32 elements. Each cell
/// spreads one system over tens of gigabytes of a lazily backed arena
/// (huge_arena.hpp), with strides that take the offsets of the matrix,
/// the right-hand side, the workspace and the pivots past 2^32, a block
/// made singular so that the second pass uses the workspace, and checks
/// the results bitwise against the same system stored contiguously. A
/// 32-bit product would wrap these offsets around onto other elements.
/// Both solvers, both layouts. The runtime solver passes TiledLUpp the
/// strides between the columns of its blocks, block or border times the
/// workspace stride: with blocks of 3 and a workspace stride of 2^31 - 1,
/// both pass 2^32, through the multi right-hand-side path. Both solvers
/// take the column stride of a block of right-hand sides past 2^32, a
/// std::size_t.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"
#include "huge_arena.hpp"

namespace {

constexpr tdls::ArrowheadShape shape{4, 2, 3, 1};
/// Stride of the matrix and of the workspace: their last entries land
/// past 2^32.
const int big_stride = tdls_tests::opaque((1 << 27) + 1);
/// Stride of the right-hand side and of the pivots: the entries from
/// index 3 on land past 2^32.
const int vector_stride = tdls_tests::opaque(2147483647);

/// \brief Element e of a strided arena.
template<typename T>
T& at(const tdls_tests::HugeArena<T>& arena, const int e, const int stride) {
    return arena.data()[static_cast<std::size_t>(e) * static_cast<std::size_t>(stride)];
}

/// \brief One system, contiguous then spread, with both solvers.
/// \tparam col column-major layout
/// \param[in] seed generator seed
template<bool col>
void wide_case(const std::uint64_t seed) {
    constexpr auto config = [] {
        tdls::ArrowheadLUppConfig<double> c;
        c.coupling = tdls::ArrowheadCoupling::Diagonal;
        c.storage  = tdls::ArrowheadStorage::Packed;
        c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
        return c;
    }();
    using Static       = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Dynamic      = tdls::ArrowheadLUppSolverDynamic<double, config>;
    constexpr int size = Static::storage_size, n = Static::dimension;
    const int wsize = Dynamic::workspace_size(shape), psize = Dynamic::pivot_size(shape);

    auto p = tdls_tests::make_arrow_problem<double>(shape, true, tdls_tests::ArrowBlocks::Dominant,
                                                    seed);
    tdls_tests::zero_block_unknown(p, 1, 0);
    const auto M0 = tdls_tests::store_arrow<typename Static::storage_map>(p);

    for (int round = 0; round < 2; ++round) {
        tdls_tests::HugeArena<double> A(static_cast<std::size_t>(size - 1) * big_stride + 1);
        tdls_tests::HugeArena<double> y(static_cast<std::size_t>(n - 1) * vector_stride + 1);
        tdls_tests::HugeArena<double> w(static_cast<std::size_t>(wsize - 1) * big_stride + 1);
        tdls_tests::HugeArena<int> piv(static_cast<std::size_t>(psize - 1) * vector_stride + 1);
        if (A.data() == nullptr || y.data() == nullptr || w.data() == nullptr ||
            piv.data() == nullptr) {
            std::printf("       not run: no lazily backed arena on this system\n");
            return;
        }
        auto Mr = M0;
        auto yr = p.b;
        std::vector<double> wr(static_cast<std::size_t>(wsize));
        std::vector<int> pr(static_cast<std::size_t>(psize));
        for (int e = 0; e < size; ++e)
            at(A, e, big_stride) = M0[static_cast<std::size_t>(e)];
        for (int i = 0; i < n; ++i)
            at(y, i, vector_stride) = p.b[static_cast<std::size_t>(i)];
        int rr = 0, rw = 0;
        bool okr = false, okw = false;
        if (round == 0) {
            okr = Static::template solve_inplace<true, true, true>(Mr.data(), 1, wr.data(), 1,
                                                                   yr.data(), 1, rr);
            okw = Static::template solve_inplace<false, false, false>(
                A.data(), big_stride, w.data(), big_stride, y.data(), vector_stride, rw);
        } else {
            okr = Dynamic::solve_inplace(shape, Mr.data(), 1, pr.data(), 1, wr.data(), 1, yr.data(),
                                         1, rr);
            okw = Dynamic::solve_inplace(shape, A.data(), big_stride, piv.data(), vector_stride,
                                         w.data(), big_stride, y.data(), vector_stride, rw);
        }
        TDLS_CHECK(okr && okw);
        TDLS_CHECK(rr == 1 && rw == 1);
        for (int e = 0; e < size; ++e)
            TDLS_CHECK_BITWISE(&Mr[static_cast<std::size_t>(e)], &at(A, e, big_stride), 1);
        for (int i = 0; i < n; ++i)
            TDLS_CHECK_BITWISE(&yr[static_cast<std::size_t>(i)], &at(y, i, vector_stride), 1);
    }
}

/// \brief The runtime solver with its workspace spread, the column
/// strides of its TiledLUpp calls past 2^32, two right-hand sides.
/// \tparam col column-major layout
/// \param[in] seed generator seed
template<bool col>
void wide_columns_case(const std::uint64_t seed) {
    constexpr tdls::ArrowheadShape shape3{4, 3, 2, 1};
    constexpr auto config = [] {
        tdls::ArrowheadLUppConfig<double> c;
        c.coupling = tdls::ArrowheadCoupling::Diagonal;
        c.storage  = tdls::ArrowheadStorage::Packed;
        c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
        return c;
    }();
    using Dynamic      = tdls::ArrowheadLUppSolverDynamic<double, config>;
    constexpr int nrhs = 2;
    const int n = shape3.n(), wsize = Dynamic::workspace_size(shape3, nrhs);

    auto p = tdls_tests::make_arrow_problem<double>(shape3, true, tdls_tests::ArrowBlocks::Dominant,
                                                    seed);
    tdls_tests::zero_block_unknown(p, 1, 0);
    const auto M0 = tdls_tests::store_arrow<typename Dynamic::storage_map>(p);
    std::vector<double> Y0(static_cast<std::size_t>(nrhs * n));
    for (int m = 0; m < nrhs; ++m)
        for (int i = 0; i < n; ++i)
            Y0[static_cast<std::size_t>(m * n + i)] = p.b[static_cast<std::size_t>(i)] * (m + 1);

    tdls_tests::HugeArena<double> w(static_cast<std::size_t>(wsize - 1) * vector_stride + 1);
    if (w.data() == nullptr) {
        std::printf("       not run: no lazily backed arena on this system\n");
        return;
    }
    auto Mr = M0, Mw = M0;
    auto Yr = Y0, Yw = Y0;
    std::vector<double> wr(static_cast<std::size_t>(wsize));
    std::vector<int> pr(static_cast<std::size_t>(Dynamic::pivot_size(shape3))), pw = pr;
    int rr = 0, rw = 0;
    const bool okr = Dynamic::solve_inplace_multirhs(shape3, nrhs, Mr.data(), 1, pr.data(), 1,
                                                     wr.data(), 1, Yr.data(), 1, n, rr);
    const bool okw = Dynamic::solve_inplace_multirhs(shape3, nrhs, Mw.data(), 1, pw.data(), 1,
                                                     w.data(), vector_stride, Yw.data(), 1, n, rw);
    TDLS_CHECK(okr && okw);
    TDLS_CHECK(rr == 1 && rw == 1);
    TDLS_CHECK(tdls_tests::same_bits(Mr, Mw));
    TDLS_CHECK(tdls_tests::same_bits(Yr, Yw));
}

/// \brief Both solvers on a block of right-hand sides whose columns lie
/// 5 * 2^30 entries apart, past 2^32: the column stride of the caller, a
/// std::size_t, as in an SoA batch of many systems.
/// \tparam col column-major layout
/// \param[in] seed generator seed
template<bool col>
void wide_rhs_columns_case(const std::uint64_t seed) {
    constexpr tdls::ArrowheadShape shape3{4, 3, 2, 1};
    constexpr auto config = [] {
        tdls::ArrowheadLUppConfig<double> c;
        c.coupling = tdls::ArrowheadCoupling::Diagonal;
        c.storage  = tdls::ArrowheadStorage::Packed;
        c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
        return c;
    }();
    using Static       = tdls::ArrowheadLUppSolverStatic<double, shape3, config>;
    using Dynamic      = tdls::ArrowheadLUppSolverDynamic<double, config>;
    constexpr int nrhs = 2;
    constexpr int n    = shape3.n();
    const std::size_t column_stride =
        static_cast<std::size_t>(tdls_tests::opaque(5)) * (std::size_t(1) << 30);

    auto p = tdls_tests::make_arrow_problem<double>(shape3, true, tdls_tests::ArrowBlocks::Dominant,
                                                    seed);
    const auto M0 = tdls_tests::store_arrow<typename Dynamic::storage_map>(p);
    std::vector<double> Y0(static_cast<std::size_t>(nrhs * n));
    for (int m = 0; m < nrhs; ++m)
        for (int i = 0; i < n; ++i)
            Y0[static_cast<std::size_t>(m * n + i)] = p.b[static_cast<std::size_t>(i)] * (m + 1);

    tdls_tests::HugeArena<double> Y(column_stride * (nrhs - 1) + n);
    if (Y.data() == nullptr) {
        std::printf("       not run: no lazily backed arena on this system\n");
        return;
    }
    for (int round = 0; round < 2; ++round) {
        const bool dynamic = round == 1;
        auto Mr = M0, Mw = M0;
        auto Yr = Y0;
        for (int m = 0; m < nrhs; ++m)
            for (int i = 0; i < n; ++i)
                Y.data()[column_stride * m + i] = Y0[static_cast<std::size_t>(m * n + i)];
        std::vector<double> w(
            static_cast<std::size_t>(dynamic ? Dynamic::workspace_size(shape3, nrhs)
                                             : Static::template multirhs_workspace_size<nrhs>));
        std::vector<int> piv(static_cast<std::size_t>(Dynamic::pivot_size(shape3)));
        int rr = 0, rw = 0;
        bool okr = false, okw = false;
        if (dynamic) {
            okr = Dynamic::solve_inplace_multirhs(shape3, nrhs, Mr.data(), 1, piv.data(), 1,
                                                  w.data(), 1, Yr.data(), 1, n, rr);
            okw = Dynamic::solve_inplace_multirhs(shape3, nrhs, Mw.data(), 1, piv.data(), 1,
                                                  w.data(), 1, Y.data(), 1, column_stride, rw);
        } else {
            okr = Static::template solve_inplace_multirhs<nrhs, false, false, false>(
                Mr.data(), 1, w.data(), 1, Yr.data(), 1, n, rr);
            okw = Static::template solve_inplace_multirhs<nrhs, false, false, false>(
                Mw.data(), 1, w.data(), 1, Y.data(), 1, column_stride, rw);
        }
        TDLS_CHECK(okr && okw);
        TDLS_CHECK(rr == rw);
        TDLS_CHECK(tdls_tests::same_bits(Mr, Mw));
        for (int m = 0; m < nrhs; ++m)
            TDLS_CHECK_BITWISE(&Yr[static_cast<std::size_t>(m * n)], &Y.data()[column_stride * m],
                               n);
    }
}

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/rowmajor") {
    wide_case<false>(1200100);
}
TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/colmajor") {
    wide_case<true>(1200200);
}
TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/columns_rowmajor") {
    wide_columns_case<false>(1200300);
}
TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/columns_colmajor") {
    wide_columns_case<true>(1200400);
}
TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/rhs_columns_rowmajor") {
    wide_rhs_columns_case<false>(1200500);
}
TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/rhs_columns_colmajor") {
    wide_rhs_columns_case<true>(1200600);
}

TDLS_TEST_MAIN
