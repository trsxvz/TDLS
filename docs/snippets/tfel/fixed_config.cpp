/// \file
/// \brief Documentation snippet: a configuration value, on fixed-size TFEL objects.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::tvector<4, double> y{14, 14, 24, 33};
    tfel::math::fsarray<4, int> piv;

    // the configuration value of the matrix scalar type, as first template argument
    constexpr auto config =
        tdls::TiledLUppConfig<double>{.tile_size = 2, .schedule = tdls::Schedule::LeftLooking};
    const bool ok = tdls::solve_inplace<config>(A, piv, y);
    // snippet end

    return ok ? snippets::check(y.data(), expected, 4) : 1;
}
