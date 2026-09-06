/// \file
/// \brief Documentation snippet: a block of canonical columns, on TFEL objects.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double A0[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double e[3][4]   = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::tmatrix<4, 3, double> X;
    tfel::math::fsarray<4, int> piv;

    if (!tdls::factorize(A, piv)) return 1;

    // a matrix-like x: its columns receive the solutions of e_0, e_1 and e_2
    tdls::substitute_canonical(A, piv, 0, X);
    // snippet end

    int status = 0;
    for (int w = 0; w < 3; ++w) {
        double column[4];
        for (int i = 0; i < 4; ++i)
            column[i] = X(i, w);
        status |= snippets::check_product(A0, column, e[w], 4);
    }
    return status;
}
