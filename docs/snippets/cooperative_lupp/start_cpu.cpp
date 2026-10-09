/// \file
/// \brief Documentation snippet: the first call, one thread on CPU.
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
    // the solver of 4 x 4 systems with 4 rows per thread: one thread solves a system
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size]; // scratch space of the solver
    int piv[4];                          // pivots, on exit

    // <true, true, true>: y, piv and A are local arrays of the thread, their strides the 1s;
    // 0 is the rank of the thread, the only one. On exit, A holds the factors and y the
    // solution; false means a singular matrix
    const bool ok = Solver::solve_inplace<true, true, true>(0, A, 1, piv, 1, y, 1, work);
    // snippet end

    return ok ? snippets::check(y, expected, 4) : 1;
}
