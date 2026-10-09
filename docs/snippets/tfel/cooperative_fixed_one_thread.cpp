/// \file
/// \brief Documentation snippet: fixed-size TFEL objects passed to the
/// CooperativeLUpp solver, one thread per system.
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

    // one thread per system: its slices are the whole objects, stored row-major,
    // so the TFEL storage is the internal residency of the raw interface
    constexpr auto config =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4, .unroll_loops = false};
    using Solver = tdls::CooperativeLUppSolverStatic<double, 4, config>;
    double work[Solver::workspace_size];

    const bool ok =
        Solver::solve_inplace<true, true, true>(0, A.data(), 1, piv.data(), 1, y.data(), 1, work);
    // snippet end

    if (!ok) return 1;
    return snippets::check(y.data(), expected, 4);
}
