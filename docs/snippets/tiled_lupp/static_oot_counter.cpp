/// \file
/// \brief Documentation snippet: the out-of-tile counter.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]          = {1e-12, 1, 0, 2, 2e-12, 5, 1, 0, 3, 1, 6, 1, 2, 0, 1, 7};
    const double b[4]        = {10 + 1e-12, 13 + 2e-12, 27, 33};
    const double expected[4] = {1, 2, 3, 4};
    double x[4];

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];

    // one more argument: the columns whose best in-tile pivot fell below the threshold
    int oot_count;
    if (!Solver::factorize<true, true>(A, 1, piv, 1, oot_count)) return 1;

    const bool searched_below_the_tile = oot_count == 1 && piv[0] == 2;
    // snippet end

    if (!searched_below_the_tile) return 1;
    Solver::substitute<true, true, true>(A, 1, piv, 1, b, x, 1);
    return snippets::check(x, expected, 4);
}
