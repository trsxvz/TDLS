/// \file
/// \brief Documentation snippet, must not compile: a sub-matrix view.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>

#include <tdls/tdls.hpp>

int main() {
    tfel::math::tmatrix<4, 4, double> J{};
    int piv[2];

    // snippet begin
    // a 2 x 2 block of a 4 x 4 matrix: its rows are 4 elements apart, its columns 1
    auto S = J.submatrix_view<0, 0, 2, 2>();

    return tdls::factorize(S, piv) ? 0 : 1;
    // snippet end
}
