/// \file
/// \brief Documentation snippet: the first call, on CPU.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[6 * 6]          = {6, 1, 1, 0, 2, 0, 1, 7, 0, 1, 0, 1, 1, 0, 5, 1, 0, 0,
                                0, 2, 1, 4, 0, 0, 2, 1, 0, 0, 6, 2, 0, 1, 0, 0, 1, 5};
    double y[6]              = {21, 25, 20, 23, 46, 37};
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    // the shape: a border of 2 unknowns, then 2 blocks of 2 unknowns
    constexpr auto shape = tdls::ArrowheadShape{.border = 2, .block = 2, .blocks = 2};
    // the default configuration reads the full 6 x 6 matrix, row-major
    using Solver =
        tdls::ArrowheadLUppSolverStatic<double, shape, tdls::ArrowheadLUppConfig<double>{}>;

    double work[Solver::workspace_size]; // touched only when a block is refused

    // <true, true, true>: y, work and A are local arrays (their strides, here 1, are ignored). On
    // exit, y holds the solution and A the recovery data; false means a singular matrix
    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 6) : 1;
}
