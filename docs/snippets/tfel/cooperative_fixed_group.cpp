/// \file
/// \brief Documentation snippet: fixed-size TFEL objects shared by the
/// group of threads of the CooperativeLUpp solver.
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
#include "group.hpp"

int main() {
    const double expected[4] = {1, 2, 3, 4};
    int ok[2]                = {};

    // snippet begin
    tfel::math::tmatrix<4, 4, double> A{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::tvector<4, double> y{14, 14, 24, 33};
    tfel::math::fsarray<4, int> piv;

    // 2 threads of 2 rows share the objects: they are external operands, row-major with
    // stride 1, as the default configuration expects
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;
    double work[Solver::workspace_size];

    snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
        ok[tx] = Solver::solve_inplace<false, false, false>(tx, A.data(), 1, piv.data(), 1,
                                                            y.data(), 1, work, sync);
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(y.data(), expected, 4);
}
