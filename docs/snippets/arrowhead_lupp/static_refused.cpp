/// \file
/// \brief Documentation snippet: a refused block and the second pass.
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
    double y[n] = {};
    for (int i = 0; i < n; ++i)
        y[i] = b_singular_block[i];

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A_singular_block, M);

    // the second block is singular: the pivot check refuses it, and the second pass
    // eliminates it with the border rows, in the workspace
    double work[Solver::workspace_size];
    int refused   = 0;
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1, refused);
    // ok: the matrix is invertible; refused == 1
    // snippet end

    return ok && refused == 1 ? snippets::check(y, expected, n) : 1;
}
