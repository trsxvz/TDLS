/// \file
/// \brief Documentation snippet: the group of a runtime dimension.
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
    const int n              = 5;
    std::vector<double> A    = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    std::vector<double> y    = {12, 16, 27, 40, 50};
    const double expected[5] = {1, 2, 3, 4, 5};
    std::vector<int> ok(3, 0);

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

    // the group of a runtime dimension: 3 threads for n = 5, the last one with a phantom row
    const int threads = Solver::threads_per_system(n);
    std::vector<double> work(Solver::workspace_size(n)); // 2 + 3 n elements, zero
    std::vector<int> piv(n);

    snippets::run_group(threads, [&](const int tx) {
        ok[tx] = Solver::solve_inplace(n, tx, A.data(), 1, piv.data(), 1, y.data(), 1, work.data());
    });
    // snippet end

    if (threads != 3 || !ok[0] || !ok[1] || !ok[2]) return 1;
    return snippets::check(y.data(), expected, n);
}
