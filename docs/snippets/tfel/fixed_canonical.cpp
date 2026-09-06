/// \file
/// \brief Documentation snippet: one canonical column, on fixed-size TFEL objects.
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
    const double A0[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double e2[4]     = {0, 0, 1, 0};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::tvector<4, double> x;
    tfel::math::fsarray<4, int> piv;

    if (!tdls::factorize(A, piv)) return 1;

    // x is column 2 of A^-1
    tdls::substitute_canonical(A, piv, 2, x);
    // snippet end

    return snippets::check_product(A0, x.data(), e2, 4);
}
