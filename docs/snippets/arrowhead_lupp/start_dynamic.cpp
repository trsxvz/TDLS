/// \file
/// \brief Documentation snippet: the first call with a runtime shape.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    std::vector<double> A    = {6, 1, 1, 0, 2, 0, 1, 7, 0, 1, 0, 1, 1, 0, 5, 1, 0, 0,
                                0, 2, 1, 4, 0, 0, 2, 1, 0, 0, 6, 2, 0, 1, 0, 0, 1, 5};
    std::vector<double> y    = {21, 25, 20, 23, 46, 37};
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    // the shape is a runtime argument
    const tdls::ArrowheadShape shape{.border = 2, .block = 2, .blocks = 2};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, tdls::ArrowheadLUppConfig<double>{}>;

    // every temporary of the solve in the workspace, the pivots in an int array
    std::vector<double> work(Solver::workspace_size(shape));
    std::vector<int> piv(Solver::pivot_size(shape));

    // the shape first; every operand is a pointer plus a stride
    const bool ok =
        Solver::solve_inplace(shape, A.data(), 1, piv.data(), 1, work.data(), 1, y.data(), 1);
    // snippet end

    return ok ? snippets::check(y.data(), expected, 6) : 1;
}
