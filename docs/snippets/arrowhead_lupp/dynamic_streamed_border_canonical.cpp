/// \file
/// \brief Documentation snippet: the border block of the inverse with the
/// runtime solver, the blocks streamed.
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

    std::vector<double> M(Map::size(runtime_shape));
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            M[Map::abb(runtime_shape, r, c)] = A[r * n + c];

    auto block = [](const int k, auto& blk) {
        for (int i = 0; i < 2; ++i) {
            const int row = 2 + 2 * k + i;
            for (int j = 0; j < 2; ++j)
                blk.d(i, j) = A[row * n + 2 + 2 * k + j];
            for (int c = 0; c < 2; ++c)
                blk.akb(i, c) = A[row * n + c];
            blk.coupling(i) = A[i * n + row];
        }
    };

    // the workspace of the streamed solve with 2 columns
    std::vector<double> work(Solver::streamed_workspace_size(runtime_shape, 2));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));
    std::vector<double> x(4); // both columns of the border block of A^-1
    int refused   = 0;
    const bool ok = Solver::solve_border_canonical_multirhs_streamed(
        runtime_shape, 2, block, M.data(), 1, piv.data(), 1, work.data(), 1, 0, x.data(), 1, 2,
        refused);
    // snippet end

    return ok ? snippets::check(x.data(), inverse_border, 4) : 1;
}
