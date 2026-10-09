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
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    // the solver of 4 x 4 systems, cut into tiles of 2 x 2
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4]; // pivots, on exit

    // <true, true, true>: y, piv and A are local arrays, their strides the 1s. On exit, A
    // holds the factors and y the solution; false means a singular matrix
    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 4) : 1;
}
