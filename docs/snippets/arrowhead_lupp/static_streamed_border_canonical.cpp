/// \file
/// \brief Documentation snippet: the border block of the inverse, the
/// blocks streamed.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"

int main() {
    using namespace snippets::arrow;

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Map    = Solver::storage_map;

    // only A_bb is written beforehand: the right-hand sides are the unit
    // columns of the border
    double M[Solver::storage_size] = {};
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            M[Map::abb(shape, r, c)] = A[r * n + c];

    // the blocks, without right-hand side (blk.y is not read)
    auto block = [](const int k, Solver::StreamedBlock& blk) {
        for (int i = 0; i < 2; ++i) {
            const int row = 2 + 2 * k + i;
            for (int j = 0; j < 2; ++j)
                blk.d(i, j) = A[row * n + 2 + 2 * k + j];
            for (int c = 0; c < 2; ++c)
                blk.akb(i, c) = A[row * n + c];
            blk.coupling(i) = A[i * n + row];
        }
    };

    double work[Solver::multirhs_workspace_size<2>];
    double x[4]; // both columns of the border block of A^-1
    int refused   = 0;
    const bool ok = Solver::solve_border_canonical_multirhs_streamed<2, true, true, true>(
        block, M, 1, work, 1, 0, x, 1, 2, refused);
    // snippet end

    return ok ? snippets::check(x, inverse_border, 4) : 1;
}
