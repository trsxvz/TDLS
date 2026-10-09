/// \file
/// \brief Documentation snippet: more rows per thread than the dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[5 * 5] = {4, 1, 0, 0, 1, 1, 5, 1, 0, 0, 0, 1, 6, 1, 0, 0, 0, 1, 7, 1, 1, 0, 0, 1, 8};
    double y[5]     = {11, 14, 24, 36, 45};
    const double expected[5] = {1, 2, 3, 4, 5};

    // snippet begin
    // 8 rows per thread on a 5 x 5 system act as 5: one thread per system
    constexpr auto config =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 8, .unroll_loops = false};
    using Solver = tdls::CooperativeLUppSolverStatic<double, 5, config>;
    static_assert(Solver::rows_per_thread == 5 && Solver::threads_per_system == 1);

    double work[Solver::workspace_size];
    int piv[5];
    const bool ok = Solver::solve_inplace<true, true, true>(0, A, 1, piv, 1, y, 1, work);
    // snippet end

    if (!ok) return 1;
    return snippets::check(y, expected, 5);
}
