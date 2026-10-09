/// \file
/// \brief Documentation snippet: one thread per system.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    // 4 rows per thread on a 4 x 4 system: one thread per system, the sequential
    // setting, without forced unrolling as recommended on CPU
    constexpr auto config =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4, .unroll_loops = false};
    using Solver = tdls::CooperativeLUppSolverStatic<double, 4, config>;
    static_assert(Solver::threads_per_system == 1);

    double work[Solver::workspace_size];
    int piv[4];

    // rank 0 and no barrier argument: the deduced barrier does nothing for one thread; the
    // slices of the only thread are the whole objects, so every operand can be a local array
    const bool ok = Solver::solve_inplace<true, true, true>(0, A, 1, piv, 1, y, 1, work);
    // snippet end

    if (!ok) return 1;
    return snippets::check(y, expected, 4);
}
