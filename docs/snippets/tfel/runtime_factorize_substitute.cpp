/// \file
/// \brief Documentation snippet: factorize, then substitute, on runtime-sized TFEL objects.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/matrix.hxx>
#include <TFEL/Math/vector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double expected1[5] = {1, 2, 3, 4, 5};
    const double expected2[5] = {1, 1, 1, 1, 1};

    // snippet begin
    tfel::math::matrix<double> A = {
        {5, 1, 0, 0, 1}, {1, 6, 1, 0, 0}, {0, 1, 7, 1, 0}, {0, 0, 1, 8, 1}, {1, 0, 0, 1, 9}};
    const tfel::math::vector<double> b1 = {12, 16, 27, 40, 50}, b2 = {7, 8, 9, 10, 11};
    tfel::math::vector<double> x1(5), x2(5);
    tfel::math::vector<int> piv(5);

    // the dimension is read from the objects at run time, the runtime solver is resolved
    if (!tdls::factorize(A, piv)) return 1;

    tdls::substitute(A, piv, b1, x1);
    tdls::substitute(A, piv, b2, x2);
    // snippet end

    return snippets::check(x1.data(), expected1, 5) | snippets::check(x2.data(), expected2, 5);
}
