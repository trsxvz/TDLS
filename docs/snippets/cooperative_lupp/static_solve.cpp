/// \file
/// \brief Documentation snippet: solve in one call, with one thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double b[4]        = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    // 4 rows per thread for a 4 x 4 system: one thread solves it
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];
    int piv[4];
    double x[4];

    // factorize and substitute in one call; A holds the factors afterwards
    const bool ok = Solver::solve<true, true, true>(0, A, 1, piv, 1, b, x, 1, work);
    // snippet end

    return ok ? snippets::check(x, expected, 4) : 1;
}
