/// \file
/// \brief Documentation snippet, must not compile: a const matrix.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>

#include <tdls/tdls.hpp>

int main() {
    int piv[4];

    // snippet begin
    // factorize writes the factors into its matrix argument
    const tfel::math::tmatrix<4, 4, double> A{};

    return tdls::factorize(A, piv) ? 0 : 1;
    // snippet end
}
