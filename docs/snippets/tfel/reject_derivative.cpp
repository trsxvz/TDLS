/// \file
/// \brief Documentation snippet, must not compile: a derivative view.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

int main() {
    tfel::math::tmatrix<4, 4, double> J{};
    int piv[2];

    // snippet begin
    // a 2 x 2 derivative block inside a 4 x 4 jacobian: rows 4 elements apart
    using vector2 = tfel::math::tvector<2, double>;
    auto D        = tfel::math::map_derivative<0, 0, vector2, vector2>(J);

    return tdls::factorize(D, piv) ? 0 : 1;
    // snippet end
}
