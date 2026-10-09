/// \file
/// \brief Documentation snippet: the first call of the runtime solver,
/// one thread on CPU.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const int n              = 4;
    std::vector<double> A    = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    std::vector<double> y    = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    // the dimension n is a runtime argument; 8 rows per thread: one thread up to n = 8
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 8};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

    std::vector<double> work(Solver::workspace_size(n));
    std::vector<int> piv(n);

    // n, then the rank 0 of the thread; every operand is a pointer plus a stride
    const bool ok =
        Solver::solve_inplace(n, 0, A.data(), 1, piv.data(), 1, y.data(), 1, work.data());
    // snippet end

    return ok ? snippets::check(y.data(), expected, n) : 1;
}
