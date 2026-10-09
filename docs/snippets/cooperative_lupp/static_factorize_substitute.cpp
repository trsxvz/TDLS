/// \file
/// \brief Documentation snippet: factorize, then substitute two
/// right-hand sides, with one thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]           = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double b1[4]        = {14, 14, 24, 33};
    const double b2[4]        = {7, 7, 8, 10};
    const double expected1[4] = {1, 2, 3, 4};
    const double expected2[4] = {1, 1, 1, 1};

    // snippet begin
    // 4 rows per thread for a 4 x 4 system: one thread solves it
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size]; // scratch space of the solver
    int piv[4];
    double x1[4], x2[4];

    // 0 is the rank of the thread, the only one; A becomes L and U in place, piv the
    // position of each row in the pivoted order; false means a singular matrix
    if (!Solver::factorize<true, true>(0, A, 1, piv, 1, work)) return 1;

    // two right-hand sides on the same factorization
    Solver::substitute<true, true, true>(0, A, 1, piv, 1, b1, x1, 1, work);
    Solver::substitute<true, true, true>(0, A, 1, piv, 1, b2, x2, 1, work);
    // snippet end

    return snippets::check(x1, expected1, 4) | snippets::check(x2, expected2, 4);
}
