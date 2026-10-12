/// \file
/// \brief Documentation snippet: a streamed solve with the runtime solver,
/// the provider of the compile-time solver unchanged.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"

int main() {
    using namespace snippets::arrow;

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;
    using Map    = Solver::storage_map;

    const tdls::ArrowheadShape runtime_shape{.border = 2, .block = 2, .blocks = 2};

    std::vector<double> M(Map::size(runtime_shape)), y(n);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c)
            M[Map::abb(runtime_shape, r, c)] = A[r * n + c];
        y[r] = b[r];
    }

    // the same accessors as the compile-time solver: a generic lambda
    // serves both
    auto block = [](const int k, auto& blk) {
        for (int i = 0; i < 2; ++i) {
            const int row = 2 + 2 * k + i;
            for (int j = 0; j < 2; ++j)
                blk.d(i, j) = A[row * n + 2 + 2 * k + j];
            for (int c = 0; c < 2; ++c)
                blk.akb(i, c) = A[row * n + c];
            blk.coupling(i) = A[i * n + row];
            blk.y(i)        = b[row];
        }
    };

    // the block is written in the workspace, which grows accordingly
    std::vector<double> work(Solver::streamed_workspace_size(runtime_shape));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));
    int refused   = 0;
    const bool ok = Solver::solve_inplace_streamed(runtime_shape, block, M.data(), 1, piv.data(), 1,
                                                   work.data(), 1, y.data(), 1, refused);
    // snippet end

    return ok ? snippets::check(y.data(), expected, n) : 1;
}
