/// \file
/// \brief Documentation snippet: a tile larger than the dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[3 * 3]          = {3, 1, 0, 1, 4, 1, 0, 1, 5};
    double y[3]              = {5, 12, 17};
    const double expected[3] = {1, 2, 3};

    // snippet begin
    // the grid is a single partial tile: 3 of its 8 x 8 slots are used, the others never touched
    constexpr auto wide = tdls::TiledLUppConfig<double>{.tile_size = 8};
    using Solver        = tdls::TiledLUppSolverStatic<double, 3, wide>;

    static_assert(Solver::F == 0 && Solver::TAIL == 3);

    int piv[3];

    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 3) : 1;
}
