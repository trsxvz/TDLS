/// \file
/// \brief Documentation snippet: views on pointers.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/Array/View.hxx>
#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double raw_A[4 * 4]      = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double raw_b[4]    = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    // a view on a plain pointer, a read-only one on a const pointer: internal residency
    auto A = tfel::math::map<tfel::math::tmatrix<4, 4, double>>(raw_A);
    auto b = tfel::math::map<const tfel::math::tvector<4, double>>(raw_b);

    tfel::math::tvector<4, double> x;
    tfel::math::fsarray<4, int> piv;

    const bool ok = tdls::solve(A, piv, b, x);
    // snippet end

    return ok ? snippets::check(x.data(), expected, 4) : 1;
}
