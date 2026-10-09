/// \file
/// \brief Documentation snippet: factorize, then substitute, compile-time dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    double A[4 * 4]           = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double b1[4]        = {14, 14, 24, 33};
    const double b2[4]        = {7, 7, 8, 10};
    const double expected1[4] = {1, 2, 3, 4};
    const double expected2[4] = {1, 1, 1, 1};
    double x1[4], x2[4];
    int ok[2] = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];
    int piv[4];

    snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
        // A holds the factors on exit, piv the position of each row in the pivoted order
        ok[tx] = Solver::factorize<false, false>(tx, A, 1, piv, 1, work, sync);
        // one factorization, two right-hand sides
        Solver::substitute<false, false, false>(tx, A, 1, piv, 1, b1, x1, 1, work, sync);
        Solver::substitute<false, false, false>(tx, A, 1, piv, 1, b2, x2, 1, work, sync);
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(x1, expected1, 4) | snippets::check(x2, expected2, 4);
}
