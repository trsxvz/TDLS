/// \file
/// \brief Documentation snippet: one column of the inverse, with one
/// thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double A0[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A[4 * 4]        = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double e2[4]     = {0, 0, 1, 0};

    // snippet begin
    // 4 rows per thread for a 4 x 4 system: one thread solves it
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];
    int piv[4];
    double x[4];

    if (!Solver::factorize<true, true>(0, A, 1, piv, 1, work)) return 1;
    // x := A^-1 e_2, the right-hand side generated on the fly: no buffer for it
    Solver::substitute_canonical<true, true, true>(0, A, 1, piv, 1, 2, x, 1, work);
    // snippet end

    return snippets::check_product(A0, x, e2, 4);
}
