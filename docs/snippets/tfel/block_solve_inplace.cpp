/// \file
/// \brief Documentation snippet: solve a block in place, on TFEL objects.
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
    tfel::math::tmatrix<4, 3, double> Y{14, 7, 8, 14, 7, 3, 24, 8, 6, 33, 10, 5};
    tfel::math::fsarray<4, int> piv;

    // Y holds the three right-hand sides on entry and the three solutions on exit
    const bool ok = tdls::solve_inplace(A, piv, Y);
    // snippet end

    return ok ? snippets::check(Y.data(), expected, 12) : 1;
}
