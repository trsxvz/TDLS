/// \file
/// \brief Documentation snippet: phantom rows.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    double A[5 * 5] = {4, 1, 0, 0, 1, 1, 5, 1, 0, 0, 0, 1, 6, 1, 0, 0, 0, 1, 7, 1, 1, 0, 0, 1, 8};
    double y[5]     = {11, 14, 24, 36, 45};
    const double expected[5] = {1, 2, 3, 4, 5};
    int ok[3]                = {};

    // snippet begin
    // 5 rows on threads of 2 rows: 3 threads; slot 1 holds rows 3 and 4 on threads 0
    // and 1, and a phantom row on thread 2, skipped at compile time
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 5, config>;
    static_assert(Solver::threads_per_system == 3);
    static_assert(Solver::slot_is_full(0) && !Solver::slot_is_full(1) && Solver::slot_has_rows(1));

    double work[Solver::workspace_size] = {};
    int piv[5];
    snippets::run_group<Solver::threads_per_system>([&](const int tx) {
        ok[tx] = Solver::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work);
    });
    // snippet end

    if (!ok[0] || !ok[1] || !ok[2]) return 1;
    return snippets::check(y, expected, 5);
}
