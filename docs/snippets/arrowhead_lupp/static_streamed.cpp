/// \file
/// \brief Documentation snippet: a solve whose blocks are streamed, given
/// by the caller when the solver condenses them.
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

    // only the border is written beforehand: A_bb and the border rows of y
    double M[Solver::storage_size] = {};
    double y[n]                    = {};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c)
            M[Map::abb(shape, r, c)] = A[r * n + c];
        y[r] = b[r];
    }

    // block k, written when the solver asks for it (here read from A)
    auto block = [](const int k, Solver::StreamedBlock& blk) {
        for (int i = 0; i < 2; ++i) {
            const int row = 2 + 2 * k + i;
            for (int j = 0; j < 2; ++j)
                blk.d(i, j) = A[row * n + 2 + 2 * k + j]; // D_k
            for (int c = 0; c < 2; ++c)
                blk.akb(i, c) = A[row * n + c]; // A_kb
            blk.coupling(i) = A[i * n + row];   // A_bk(i, i)
            blk.y(i)        = b[row];           // y_k
        }
    };

    double work[Solver::workspace_size];
    int refused = 0;
    const bool ok =
        Solver::solve_inplace_streamed<true, true, true>(block, M, 1, work, 1, y, 1, refused);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
