/// \file
/// \brief Documentation snippet: factorize, then substitute, on fixed-size TFEL objects.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double expected1[4] = {1, 2, 3, 4};
    const double expected2[4] = {1, 1, 1, 1};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const tfel::math::tvector<4, double> b1{14, 14, 24, 33}, b2{7, 7, 8, 10};
    tfel::math::tvector<4, double> x1, x2;
    tfel::math::fsarray<4, int> piv;

    // the dimension, the residencies and the strides are deduced from the objects
    if (!tdls::factorize(A, piv)) return 1;

    tdls::substitute(A, piv, b1, x1);
    tdls::substitute(A, piv, b2, x2);
    // snippet end

    return snippets::check(x1.data(), expected1, 4) | snippets::check(x2.data(), expected2, 4);
}
