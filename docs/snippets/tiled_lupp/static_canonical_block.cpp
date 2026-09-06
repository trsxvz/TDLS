/// \file
/// \brief Documentation snippet: a block of canonical columns.
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
    const double e[3][4]   = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];

    if (!Solver::factorize<true, true>(A, 1, piv, 1)) return 1;

    // the first three columns of A^-1 in one sweep: column w of X starts at X + w * 4
    double X[3 * 4];
    Solver::substitute_canonical_multirhs<3, true, true, true>(A, 1, piv, 1, 0, X, 1, 0);
    // snippet end

    int status = 0;
    for (int w = 0; w < 3; ++w)
        status |= snippets::check_product(A0, X + w * 4, e[w], 4);
    return status;
}
