/// \file
/// \brief Documentation snippet: one canonical column, on runtime-sized TFEL objects.
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
    const double A0[5 * 5] = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                              1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    const double e2[5]     = {0, 0, 1, 0, 0};

    // snippet begin
    tfel::math::matrix<double> A = {
        {5, 1, 0, 0, 1}, {1, 6, 1, 0, 0}, {0, 1, 7, 1, 0}, {0, 0, 1, 8, 1}, {1, 0, 0, 1, 9}};
    tfel::math::vector<double> x(5);
    tfel::math::vector<int> piv(5);

    if (!tdls::factorize(A, piv)) return 1;

    // x is column 2 of A^-1
    tdls::substitute_canonical(A, piv, 2, x);
    // snippet end

    return snippets::check_product(A0, x.data(), e2, 5);
}
