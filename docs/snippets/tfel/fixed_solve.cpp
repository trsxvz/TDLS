/// \file
/// \brief Documentation snippet: solve in one call, on fixed-size TFEL objects.
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
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const tfel::math::tvector<4, double> b{14, 14, 24, 33};
    tfel::math::tvector<4, double> x;
    tfel::math::fsarray<4, int> piv;

    const bool ok = tdls::solve(A, piv, b, x);
    // snippet end

    return ok ? snippets::check(x.data(), expected, 4) : 1;
}
