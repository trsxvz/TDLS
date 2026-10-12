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
/// Both solvers, both layouts.

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
constexpr int big_stride = (1 << 27) + 1;
/// Stride of the right-hand side and of the pivots: the entries from
/// index 3 on land past 2^32.
constexpr int vector_stride = 2147483647;

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

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/rowmajor") {
    wide_case<false>(1200100);
}
TDLS_TEST_CASE("arrowheadlupp/bridge/wide_offsets/colmajor") {
    wide_case<true>(1200200);
}

TDLS_TEST_MAIN
