/// \file
/// \brief Documentation snippet, must not compile: an extent mismatch.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

int main() {
    tfel::math::tmatrix<4, 4, double> A{};
    int piv[4];

    // snippet begin
    // a 3-vector against a 4 x 4 matrix: both extents are known at compile time
    tfel::math::tvector<3, double> b{}, x;

    return tdls::solve(A, piv, b, x) ? 0 : 1;
    // snippet end
}
