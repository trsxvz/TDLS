/// \file
/// \brief Documentation snippet: the unroll policy.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A_unrolled[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A_rolled[4 * 4]   = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y_unrolled[4]     = {14, 14, 24, 33};
    double y_rolled[4]       = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    constexpr auto unrolled = tdls::TiledLUppConfig<double>{.tile_size = 2};
    constexpr auto rolled   = tdls::TiledLUppConfig<double>{.tile_size = 2, .unroll_inner = false};

    using Unrolled = tdls::TiledLUppSolverStatic<double, 4, unrolled>;
    using Rolled   = tdls::TiledLUppSolverStatic<double, 4, rolled>;

    int piv[4];

    // no unroll pragma: same values, faster builds
    const bool ok_unrolled =
        Unrolled::solve_inplace<true, true, true>(A_unrolled, 1, piv, 1, y_unrolled, 1);
    const bool ok_rolled =
        Rolled::solve_inplace<true, true, true>(A_rolled, 1, piv, 1, y_rolled, 1);
    // snippet end

    if (!ok_unrolled || !ok_rolled) return 1;
    return snippets::check(y_unrolled, expected, 4) |
           snippets::check_bitwise(y_unrolled, y_rolled, 4) |
           snippets::check_bitwise(A_unrolled, A_rolled, 16);
}
