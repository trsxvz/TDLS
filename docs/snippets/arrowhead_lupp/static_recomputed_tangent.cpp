/// \file
/// \brief Documentation snippet: one recomputed pass of a Newton
/// iteration, the tangent when converged, the correction otherwise.
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

    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling        = tdls::ArrowheadCoupling::Diagonal,
                                          .storage         = tdls::ArrowheadStorage::Packed,
                                          .register_blocks = true};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using RO     = Solver::RecomputedOutcome;

    double Abb[2 * 2] = {};
    double y[n]       = {};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c)
            Abb[r * 2 + c] = A[r * n + c];
        y[r] = b[r];
    }
    auto block = [](const int k, Solver::RecomputedBlock& blk) {
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
    // called once, after the last block: converged, the tangent replaces
    // the correction, and the blocks are not asked for again
    auto converged = [] { return true; };

    constexpr int capacity = 2;
    double work[Solver::recomputed_workspace_size<capacity, 2>];
    double x[4]; // the 2 columns of the border block of A^-1
    int refused      = 0;
    const RO outcome = Solver::solve_inplace_recomputed_or_tangent<2, capacity, true, true, true>(
        block, converged, Abb, 1, work, 1, y, 1, 0, x, 1, 2, refused);
    // outcome: tangent (x written, y untouched), solved when converged()
    // returns false (y holds the correction), singular, or deferred
    // snippet end

    return outcome == RO::tangent ? snippets::check(x, inverse_border, 4) : 1;
}
