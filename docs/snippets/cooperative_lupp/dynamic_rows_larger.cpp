/// \file
/// \brief Documentation snippet: more rows per thread than the dimension, runtime dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const int n              = 5;
    std::vector<double> A    = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    std::vector<double> y    = {12, 16, 27, 40, 50};
    const double expected[5] = {1, 2, 3, 4, 5};

    // snippet begin
    // 8 rows per thread: one thread solves every n up to 8, its slots beyond n phantom
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 8};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

    std::vector<double> work(Solver::workspace_size(n));
    std::vector<int> piv(n);

    // one thread: rank 0 and no barrier, the default tdls::NoSync; the runtime solver cannot
    // check this at compile time, n <= rows_per_thread is the caller's to keep
    const bool ok =
        Solver::threads_per_system(n) == 1 &&
        Solver::solve_inplace(n, 0, A.data(), 1, piv.data(), 1, y.data(), 1, work.data());
    // snippet end

    if (!ok) return 1;
    return snippets::check(y.data(), expected, n);
}
