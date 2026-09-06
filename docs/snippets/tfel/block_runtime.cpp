/// \file
/// \brief Documentation snippet: a runtime-sized block, on TFEL objects.
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
    const double expected[5 * 3] = {1, 1, 1, 2, 1, 0, 3, 1, 1, 4, 1, 0, 5, 1, 1};

    // snippet begin
    tfel::math::matrix<double> A = {
        {5, 1, 0, 0, 1}, {1, 6, 1, 0, 0}, {0, 1, 7, 1, 0}, {0, 0, 1, 8, 1}, {1, 0, 0, 1, 9}};
    const tfel::math::matrix<double> B = {
        {12, 7, 6}, {16, 8, 2}, {27, 9, 7}, {40, 10, 2}, {50, 11, 10}};
    tfel::math::matrix<double> X(5, 3);
    tfel::math::vector<int> piv(5);

    // the column count is read from the block at run time
    const bool ok = tdls::solve(A, piv, B, X);
    // snippet end

    return ok ? snippets::check(X.data(), expected, 15) : 1;
}
