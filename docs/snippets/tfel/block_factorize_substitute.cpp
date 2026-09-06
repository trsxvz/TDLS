/// \file
/// \brief Documentation snippet: factorize, then substitute a block, on TFEL objects.
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
    const double expected[4 * 3] = {1, 1, 2, 2, 1, 0, 3, 1, 1, 4, 1, 0};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const tfel::math::tmatrix<4, 3, double> B{14, 7, 8, 14, 7, 3, 24, 8, 6, 33, 10, 5};
    tfel::math::tmatrix<4, 3, double> X;
    tfel::math::fsarray<4, int> piv;

    if (!tdls::factorize(A, piv)) return 1;

    // a matrix-like right-hand side: one system per column, solved together
    tdls::substitute(A, piv, B, X);
    // snippet end

    return snippets::check(X.data(), expected, 12);
}
