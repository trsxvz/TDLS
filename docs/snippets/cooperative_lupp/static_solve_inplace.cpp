/// \file
/// \brief Documentation snippet: solve in place, compile-time dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    int ok[2]                = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];
    int piv[4];

    // the forward pass folded into the factorization: y holds b on entry, x on exit
    snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
        ok[tx] = Solver::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work, sync);
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(y, expected, 4);
}
