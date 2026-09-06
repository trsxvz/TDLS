/// \file
/// \brief Documentation snippet: the counting overloads, on fixed-size TFEL objects.
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
    tfel::math::tmatrix<4, 4, double> A{1e-12, 1, 0, 2, 2e-12, 5, 1, 0, 3e-12, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::tvector<4, double> y{10 + 1e-12, 13 + 2e-12, 24 + 3e-12, 33};
    tfel::math::fsarray<4, int> piv;

    // the trailing int&: the columns whose best in-tile pivot fell below the threshold
    int oot_count;
    const bool ok = tdls::solve_inplace(A, piv, y, oot_count);

    const bool searched_below_the_tile = oot_count == 1 && piv[0] == 3;
    // snippet end

    if (!ok || !searched_below_the_tile) return 1;
    return snippets::check(y.data(), expected, 4);
}
