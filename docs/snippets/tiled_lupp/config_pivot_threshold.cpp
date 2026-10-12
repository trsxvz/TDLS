/// \file
/// \brief Documentation snippet: the relative pivot threshold.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double A0[4 * 4]   = {3, 1, 0, 0, 4, 6, 1, 0, 0, 1, 5, 1, 0, 0, 1, 4};
    const double b0[4]       = {5, 19, 21, 19};
    const double expected[4] = {1, 2, 3, 4};
    double A1[4 * 4], A2[4 * 4], y1[4], y2[4];
    for (int k = 0; k < 4 * 4; ++k)
        A1[k] = A2[k] = A0[k];
    for (int i = 0; i < 4; ++i)
        y1[i] = y2[i] = b0[i];

    // snippet begin
    // column 0 of the first tile holds 3 in place and 4 below it: a
    // threshold of 1 takes 4; the default threshold 0.1 keeps 3, since
    // 3 >= 0.1 * 4
    constexpr auto largest =
        tdls::TiledLUppConfig<double>{.tile_size = 2, .relative_pivot_threshold = 1.0};
    constexpr auto relaxed = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Largest          = tdls::TiledLUppSolverStatic<double, 4, largest>;
    using Relaxed          = tdls::TiledLUppSolverStatic<double, 4, relaxed>;

    int piv1[4], piv2[4];
    const bool ok1 = Largest::solve_inplace<true, true, true>(A1, 1, piv1, 1, y1, 1);
    const bool ok2 = Relaxed::solve_inplace<true, true, true>(A2, 1, piv2, 1, y2, 1);

    // a threshold of 1 swaps rows 0 and 1; under 0.1 every row stays in
    // place, and the in-tile multipliers stay below 1 / 0.1 = 10
    const bool swapped  = piv1[0] == 1 && piv1[1] == 0;
    const bool in_place = piv2[0] == 0 && piv2[1] == 1 && piv2[2] == 2 && piv2[3] == 3;
    // snippet end

    if (!ok1 || !ok2 || !swapped || !in_place) return 1;
    return snippets::check(y1, expected, 4) | snippets::check(y2, expected, 4);
}
