/// \file
/// \brief Documentation snippet: factorize, then substitute two
/// right-hand sides, runtime dimension, one thread.
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
    const std::vector<double> b1 = {12, 16, 27, 40, 50};
    const std::vector<double> b2 = {7, 8, 9, 10, 11};
    const double expected1[5]    = {1, 2, 3, 4, 5};
    const double expected2[5]    = {1, 1, 1, 1, 1};

    // snippet begin
    // 8 rows per thread: one thread solves every dimension up to 8
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 8};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

    std::vector<double> work(Solver::workspace_size(n)), x1(n), x2(n);
    std::vector<int> piv(n);

    // n comes first, then the rank of the thread, 0: every operand is a pointer plus a stride
    if (!Solver::factorize(n, 0, A.data(), 1, piv.data(), 1, work.data())) return 1;

    // two right-hand sides on the same factorization
    Solver::substitute(n, 0, A.data(), 1, piv.data(), 1, b1.data(), x1.data(), 1, work.data());
    Solver::substitute(n, 0, A.data(), 1, piv.data(), 1, b2.data(), x2.data(), 1, work.data());
    // snippet end

    return snippets::check(x1.data(), expected1, n) | snippets::check(x2.data(), expected2, n);
}
