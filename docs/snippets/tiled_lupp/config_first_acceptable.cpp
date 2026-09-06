/// \file
/// \brief Documentation snippet: the out-of-tile search strategy.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A_first[4 * 4]    = {1e-12, 1, 0, 2, 2e-12, 5, 1, 0, 1e-6, 1, 6, 1, 3, 0, 1, 7};
    double A_best[4 * 4]     = {1e-12, 1, 0, 2, 2e-12, 5, 1, 0, 1e-6, 1, 6, 1, 3, 0, 1, 7};
    const double b[4]        = {10 + 1e-12, 13 + 2e-12, 24 + 1e-6, 34};
    const double expected[4] = {1, 2, 3, 4};
    double x_first[4], x_best[4];

    // snippet begin
    constexpr auto first = tdls::TiledLUppConfig<double>{.tile_size = 2};
    constexpr auto best =
        tdls::TiledLUppConfig<double>{.tile_size = 2, .oot_first_acceptable = false};

    using First = tdls::TiledLUppSolverStatic<double, 4, first>;
    using Best  = tdls::TiledLUppSolverStatic<double, 4, best>;

    int piv_first[4], piv_best[4];

    // both in-tile candidates of column 0 are below 1e-10: the search goes below the tile
    const bool ok_first = First::factorize<true, true>(A_first, 1, piv_first, 1);
    const bool ok_best  = Best::factorize<true, true>(A_best, 1, piv_best, 1);

    // first acceptable: row 2, whose 1e-6 reaches the threshold; full scan: row 3, the largest
    const bool rows = piv_first[0] == 2 && piv_best[0] == 3;
    // snippet end

    if (!ok_first || !ok_best || !rows) return 1;
    First::substitute<true, true, true>(A_first, 1, piv_first, 1, b, x_first, 1);
    Best::substitute<true, true, true>(A_best, 1, piv_best, 1, b, x_best, 1);
    return snippets::check(x_first, expected, 4, 1e-8) | snippets::check(x_best, expected, 4, 1e-8);
}
