/// \file
/// \brief Documentation snippet: no unroll pragma.
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
        y[i] = b[i];

    // snippet begin
    // no forced unrolling, here and in the TiledLUpp solvers of the blocks and of the
    // border: faster compiles
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling     = tdls::ArrowheadCoupling::Diagonal,
                                          .storage      = tdls::ArrowheadStorage::Packed,
                                          .unroll_loops = false};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A, M);
    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
