/// \file
/// \brief Documentation snippet: factorize, then substitute.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]           = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double b1[4]        = {14, 14, 24, 33};
    const double b2[4]        = {7, 7, 8, 10};
    const double expected1[4] = {1, 2, 3, 4};
    const double expected2[4] = {1, 1, 1, 1};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];
    double x1[4], x2[4];

    // A becomes L and U in place, piv the row permutation; false means a singular matrix
    if (!Solver::factorize<true, true>(A, 1, piv, 1)) return 1;

    // two right-hand sides on the same factorization
    Solver::substitute<true, true, true>(A, 1, piv, 1, b1, x1, 1);
    Solver::substitute<true, true, true>(A, 1, piv, 1, b2, x2, 1);
    // snippet end

    return snippets::check(x1, expected1, 4) | snippets::check(x2, expected2, 4);
}
