/// \file
/// \brief Documentation snippet: one canonical column, compile-time dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    const double A0[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A[4 * 4]        = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double e2[4]     = {0, 0, 1, 0};
    double x[4];
    int ok[2] = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];
    int piv[4];

    snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
        ok[tx] = Solver::factorize<false, false>(tx, A, 1, piv, 1, work, sync);
        // x := A^-1 e_2, the right-hand side generated on the fly: no buffer for it
        Solver::substitute_canonical<false, false, false>(tx, A, 1, piv, 1, 2, x, 1, work, sync);
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check_product(A0, x, e2, 4);
}
