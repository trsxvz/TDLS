/// \file
/// \brief Documentation snippet: pass cutting, on TFEL objects.
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
    const double expected[4 * 5] = {1, 1, 2, 0, 1, 2, 1, 0, 1, 0, 3, 1, 1, 0, 0, 4, 1, 0, 1, 0};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const tfel::math::tmatrix<4, 5, double> B{14, 7, 8, 3, 4, 14, 7,  3, 5, 1,
                                              24, 8, 6, 2, 0, 33, 10, 5, 7, 2};
    tfel::math::tmatrix<4, 5, double> X;
    tfel::math::fsarray<4, int> piv;

    // five columns in passes of two: the pass width is the template argument
    const bool ok = tdls::solve<2>(A, piv, B, X);
    // snippet end

    return ok ? snippets::check(X.data(), expected, 20) : 1;
}
