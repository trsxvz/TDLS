/// \file
/// \brief Documentation snippet: factorize, then substitute, runtime dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    const int n                  = 5;
    std::vector<double> A        = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                    1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    const std::vector<double> b1 = {12, 16, 27, 40, 50};
    const std::vector<double> b2 = {7, 8, 9, 10, 11};
    const double expected1[5]    = {1, 2, 3, 4, 5};
    const double expected2[5]    = {1, 1, 1, 1, 1};
    std::vector<int> ok(3, 0);

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

    // n = 5 on threads of 2 rows: 3 threads, and a workspace of 3 * n elements
    std::vector<double> work(Solver::workspace_size(n)), x1(n), x2(n);
    std::vector<int> piv(n);

    // n comes first, then the rank: every operand is a pointer plus a stride
    snippets::run_group(Solver::threads_per_system(n), [&](const int tx) {
        ok[tx] = Solver::factorize(n, tx, A.data(), 1, piv.data(), 1, work.data());
        // two right-hand sides on the same factorization
        Solver::substitute(n, tx, A.data(), 1, piv.data(), 1, b1.data(), x1.data(), 1, work.data());
        Solver::substitute(n, tx, A.data(), 1, piv.data(), 1, b2.data(), x2.data(), 1, work.data());
    });
    // snippet end

    if (!ok[0] || !ok[1] || !ok[2]) return 1;
    return snippets::check(x1.data(), expected1, n) | snippets::check(x2.data(), expected2, n);
}
