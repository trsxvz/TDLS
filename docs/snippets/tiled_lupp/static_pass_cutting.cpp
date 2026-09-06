/// \file
/// \brief Documentation snippet: pass cutting.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]       = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double B[5 * 4] = {14, 14, 24, 33, 7, 7, 8, 10, 8, 3, 6, 5, 3, 5, 2, 7, 4, 1, 0, 2};
    const double expected[5 * 4] = {1, 2, 3, 4, 1, 1, 1, 1, 2, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];

    if (!Solver::factorize<true, true>(A, 1, piv, 1)) return 1;

    // five columns in passes of two, then the remaining one
    double X[5 * 4];
    Solver::substitute_multirhs<5, true, true, true, 2>(A, 1, piv, 1, B, X, 1, 0);
    // snippet end

    return snippets::check(X, expected, 20);
}
