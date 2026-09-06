/// \file
/// \brief Documentation snippet, must not compile: a column-major configuration.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>

#include <tdls/tdls.hpp>

int main() {
    tfel::math::tmatrix<4, 4, double> A{};
    int piv[4];

    // snippet begin
    // TFEL stores matrices row-major: the layout knob cannot say otherwise here
    constexpr auto colmajor = tdls::TiledLUppConfig<double>{.layout = tdls::MatrixLayout::ColMajor};

    return tdls::factorize<colmajor>(A, piv) ? 0 : 1;
    // snippet end
}
