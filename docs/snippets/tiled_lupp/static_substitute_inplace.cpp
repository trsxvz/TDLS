/// \file
/// \brief Documentation snippet: factorize, then substitute in place.
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
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];

    if (!Solver::factorize<true, true>(A, 1, piv, 1)) return 1;

    // y holds b on entry and x on exit: one buffer, permuted in place
    Solver::substitute_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    // snippet end

    return snippets::check(y, expected, 4);
}
