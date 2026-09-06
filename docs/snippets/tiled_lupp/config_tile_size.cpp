/// \file
/// \brief Documentation snippet: the tile size.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <algorithm>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double A0[6 * 6]   = {8, 1, 0, 0, 2, 0, 1, 8, 1, 0, 0, 2, 0, 1, 8, 1, 0, 0,
                                0, 0, 1, 8, 1, 0, 2, 0, 0, 1, 8, 1, 0, 2, 0, 0, 1, 8};
    const double b0[6]       = {20, 32, 30, 40, 52, 57};
    const double expected[6] = {1, 2, 3, 4, 5, 6};
    double A4[6 * 6], A3[6 * 6], y4[6], y3[6];
    std::copy(A0, A0 + 36, A4);
    std::copy(A0, A0 + 36, A3);
    std::copy(b0, b0 + 6, y4);
    std::copy(b0, b0 + 6, y3);

    // snippet begin
    constexpr auto tiles_of_4 = tdls::TiledLUppConfig<double>{.tile_size = 4};
    constexpr auto tiles_of_3 = tdls::TiledLUppConfig<double>{.tile_size = 3};

    // 6 x 6: one full tile of 4 and a trailing tile of 2, or a 2 x 2 grid of full tiles
    using Tiles4 = tdls::TiledLUppSolverStatic<double, 6, tiles_of_4>;
    using Tiles3 = tdls::TiledLUppSolverStatic<double, 6, tiles_of_3>;

    static_assert(Tiles4::F == 1 && Tiles4::TAIL == 2);
    static_assert(Tiles3::F == 2 && Tiles3::TAIL == 0);

    int piv[6];

    const bool ok4 = Tiles4::solve_inplace<true, true, true>(A4, 1, piv, 1, y4, 1);
    const bool ok3 = Tiles3::solve_inplace<true, true, true>(A3, 1, piv, 1, y3, 1);
    // snippet end

    if (!ok4 || !ok3) return 1;
    return snippets::check(y4, expected, 6) | snippets::check(y3, expected, 6);
}
