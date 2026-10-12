/// \file
/// \brief Documentation snippet: the Newton pass of the runtime solver,
/// streamed, then the tangent once converged.
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

    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;
    using Map    = Solver::storage_map;

    const tdls::ArrowheadShape runtime_shape{.border = 2, .block = 2, .blocks = 2};

    std::vector<double> M(Map::size(runtime_shape)), y(n);
    auto border = [&] {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c)
                M[Map::abb(runtime_shape, r, c)] = A[r * n + c];
            y[r] = b[r];
        }
    };
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

    // snippet begin
    std::vector<double> work(Solver::streamed_workspace_size(runtime_shape, 2));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));
    std::vector<double> x(4);
    int refused = 0;

    // a pass that has not converged: the correction, in y
    border();
    const auto first = Solver::solve_inplace_streamed_or_tangent(
        runtime_shape, 2, block, [] { return false; }, M.data(), 1, piv.data(), 1, work.data(), 1,
        y.data(), 1, 0, x.data(), 1, 2, refused);

    const std::vector<double> correction = y;

    // the pass that converged: the tangent, in x (y then holds
    // intermediate values)
    border();
    const auto last = Solver::solve_inplace_streamed_or_tangent(
        runtime_shape, 2, block, [] { return true; }, M.data(), 1, piv.data(), 1, work.data(), 1,
        y.data(), 1, 0, x.data(), 1, 2, refused);
    // snippet end

    // the correction of the first pass is the solution: x = 1 .. 6
    using O = Solver::StreamedOutcome;
    if (first != O::solved || last != O::tangent) return 1;
    return snippets::check(correction.data(), expected, n) +
           snippets::check(x.data(), inverse_border, 4);
}
