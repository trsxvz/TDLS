/// \file
/// \brief Documentation snippet: a TFEL jacobian as the Full storage of an arrowhead system.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    // the jacobian of a behaviour: a border of 2 unknowns, then two blocks of 2
    tfel::math::tmatrix<6, 6, double> J{6, 1, 1, 0, 2, 0, //
                                        1, 7, 0, 1, 0, 1, //
                                        1, 0, 5, 1, 0, 0, //
                                        0, 2, 1, 4, 0, 0, //
                                        2, 1, 0, 0, 6, 2, //
                                        0, 1, 0, 0, 1, 5};
    tfel::math::tvector<6, double> y{21, 25, 20, 23, 46, 37};

    // stored row-major and contiguously: the Full storage, the internal residency
    constexpr auto shape  = tdls::ArrowheadShape{.border = 2, .block = 2, .blocks = 2};
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Full};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    double work[Solver::workspace_size];

    const bool ok = Solver::solve_inplace<true, true, true>(J.data(), 1, work, 1, y.data(), 1);
    // snippet end

    if (!ok) return 1;
    return snippets::check(y.data(), expected, 6);
}
