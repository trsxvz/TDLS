/// \file
/// \brief Documentation snippet: the pivot thresholds.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 0, 1, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1e-8};
    double A2[4 * 4]         = {4, 1, 0, 0, 1, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1e-8};
    double y[4]              = {6, 11, 18, 4e-8};
    double y2[4]             = {6, 11, 18, 4e-8};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    constexpr auto lenient = tdls::TiledLUppConfig<double>{.tile_size = 2};
    constexpr auto strict  = tdls::TiledLUppConfig<double>{
        .tile_size = 2, .oot_threshold = 1e-6, .singular_floor = 1e-6};

    using Lenient = tdls::TiledLUppSolverStatic<double, 4, lenient>;
    using Strict  = tdls::TiledLUppSolverStatic<double, 4, strict>;

    int piv[4];

    // the last pivot, 1e-8, reaches the default threshold: accepted inside its tile
    const bool ok_lenient = Lenient::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);

    // below 1e-6 the search finds no row left, and the floor declares the matrix singular
    const bool ok_strict = Strict::solve_inplace<true, true, true>(A2, 1, piv, 1, y2, 1);
    // snippet end

    if (!ok_lenient || ok_strict) return 1;
    return snippets::check(y, expected, 4);
}
