/// \file
/// \brief Documentation snippet: the right-hand sides of a block in passes.
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
    // the 3 columns [A_kb | y_k] of each block solved 2 at a time: fewer values alive at
    // once, the faster choice on GPU under the row-major layout with a large border
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling       = tdls::ArrowheadCoupling::Diagonal,
                                          .storage        = tdls::ArrowheadStorage::Packed,
                                          .rhs_pass_width = 2};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A, M);
    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
