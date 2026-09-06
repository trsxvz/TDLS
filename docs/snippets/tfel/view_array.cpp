/// \file
/// \brief Documentation snippet: the elements of a views array.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/Array/ViewsArray.hxx>
#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double aos[3 * 4]            = {14, 14, 24, 33, 7, 7, 8, 10, 8, 3, 6, 5};
    const double expected[3 * 4] = {1, 2, 3, 4, 1, 1, 1, 1, 2, 0, 1, 0};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::fsarray<4, int> piv;

    if (!tdls::factorize(A, piv)) return 1;

    // an array of views on three contiguous right-hand sides, each passed on its own
    auto views = tfel::math::map_array<tfel::math::tvector<3, tfel::math::tvector<4, double>>>(aos);
    for (int s = 0; s < 3; ++s) {
        auto y = views[s];
        tdls::substitute_inplace(A, piv, y);
    }
    // snippet end

    return snippets::check(aos, expected, 12);
}
