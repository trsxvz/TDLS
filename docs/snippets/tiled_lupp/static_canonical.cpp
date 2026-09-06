/// \file
/// \brief Documentation snippet: one canonical column.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double A0[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A[4 * 4]        = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double e2[4]     = {0, 0, 1, 0};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];
    double x[4];

    if (!Solver::factorize<true, true>(A, 1, piv, 1)) return 1;

    // x is column 2 of A^-1
    Solver::substitute_canonical<true, true, true>(A, 1, piv, 1, 2, x, 1);
    // snippet end

    return snippets::check_product(A0, x, e2, 4);
}
