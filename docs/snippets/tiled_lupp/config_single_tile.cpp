/// \file
/// \brief Documentation snippet: a single tile.
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
    // one tile holds the whole matrix: no trailing update at all
    constexpr auto one_tile = tdls::TiledLUppConfig<double>{.tile_size = 4};
    using Solver            = tdls::TiledLUppSolverStatic<double, 4, one_tile>;

    static_assert(Solver::F == 1 && Solver::TAIL == 0);

    int piv[4];

    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 4) : 1;
}
