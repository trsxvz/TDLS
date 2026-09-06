/// \file
/// \brief Documentation snippet: the singular verdict.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

int main() {
    double A[4 * 4] = {4, 1, 0, 2, 1, 5, 0, 0, 0, 1, 0, 1, 2, 0, 0, 7};
    double y[4]     = {14, 14, 24, 33};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];

    // column 2 is zero: false, and y is left partially updated
    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    // snippet end

    return ok ? 1 : 0;
}
