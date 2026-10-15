/// \file
/// \brief Documentation snippet: the runtime solver, the blocks recomputed,
/// nothing of the matrix stored.
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
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling = tdls::ArrowheadCoupling::Diagonal,
                                          .storage  = tdls::ArrowheadStorage::Packed,
                                          .block_row_interchange = tdls::RowInterchange::Physical};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;
    using RO     = Solver::RecomputedOutcome;

    const tdls::ArrowheadShape runtime_shape{.border = 2, .block = 2, .blocks = 2};

    // the border block alone, by rows, and the border rows of y
    std::vector<double> Abb(2 * 2), y(n);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c)
            Abb[r * 2 + c] = A[r * n + c];
        y[r] = b[r];
    }

    // the accessors of the compile-time solver: a generic lambda serves both
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

    // the workspace holds the areas of one block and the store of refused
    // blocks: it does not grow with the number of blocks
    const int capacity = 2;
    std::vector<double> work(Solver::recomputed_workspace_size(runtime_shape, capacity));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));
    int refused = 0;
    const RO out =
        Solver::solve_inplace_recomputed(runtime_shape, capacity, block, Abb.data(), 1, piv.data(),
                                         1, work.data(), 1, y.data(), 1, refused);
    // snippet end

    return out == RO::solved ? snippets::check(y.data(), expected, n) : 1;
}
