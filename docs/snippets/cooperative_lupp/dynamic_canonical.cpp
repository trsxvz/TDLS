/// \file
/// \brief Documentation snippet: one column of the inverse, runtime
/// dimension, one thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const int n                  = 5;
    std::vector<double> A        = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                    1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    const std::vector<double> A0 = A;
    const double e2[5]           = {0, 0, 1, 0, 0};

    // snippet begin
    // 8 rows per thread: one thread solves every dimension up to 8
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 8};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

    std::vector<double> work(Solver::workspace_size(n)), x(n);
    std::vector<int> piv(n);

    if (!Solver::factorize(n, 0, A.data(), 1, piv.data(), 1, work.data())) return 1;
    // x := A^-1 e_2, the right-hand side generated on the fly
    Solver::substitute_canonical(n, 0, A.data(), 1, piv.data(), 1, 2, x.data(), 1, work.data());
    // snippet end

    return snippets::check_product(A0.data(), x.data(), e2, n);
}
