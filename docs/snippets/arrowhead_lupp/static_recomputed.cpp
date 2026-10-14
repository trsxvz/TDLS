/// \file
/// \brief Documentation snippet: a solve whose blocks are recomputed, given
/// by the caller twice, nothing of the matrix stored.
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
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling        = tdls::ArrowheadCoupling::Diagonal,
                                          .storage         = tdls::ArrowheadStorage::Packed,
                                          .register_blocks = true};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using RO     = Solver::RecomputedOutcome;

    // the border block alone, by rows, and the border rows of y
    double Abb[2 * 2] = {};
    double y[n]       = {};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c)
            Abb[r * 2 + c] = A[r * n + c];
        y[r] = b[r];
    }

    // block k, written when the solver asks for it: once to condense it,
    // once more (blk.recovery set) to recover its unknowns
    auto block = [](const int k, Solver::RecomputedBlock& blk) {
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

    // room for every block refused by the pivot check: never deferred
    constexpr int capacity = 2;
    double work[Solver::recomputed_workspace_size<capacity>];
    int refused  = 0;
    const RO out = Solver::solve_inplace_recomputed<capacity, true, true, true>(block, Abb, 1, work,
                                                                                1, y, 1, refused);
    // snippet end

    return out == RO::solved ? snippets::check(y, expected, n) : 1;
}
