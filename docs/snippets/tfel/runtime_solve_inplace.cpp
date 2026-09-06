/// \file
/// \brief Documentation snippet: solve in place, on runtime-sized TFEL objects.
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
    const double expected[5] = {1, 2, 3, 4, 5};

    // snippet begin
    tfel::math::matrix<double> A = {
        {5, 1, 0, 0, 1}, {1, 6, 1, 0, 0}, {0, 1, 7, 1, 0}, {0, 0, 1, 8, 1}, {1, 0, 0, 1, 9}};
    tfel::math::vector<double> y = {12, 16, 27, 40, 50};
    tfel::math::vector<int> piv(5);

    // y holds b on entry and x on exit
    const bool ok = tdls::solve_inplace(A, piv, y);
    // snippet end

    return ok ? snippets::check(y.data(), expected, 5) : 1;
}
