/// \file
/// \brief Documentation snippet: the first call of the runtime solver,
/// on CPU.
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
    // the dimension n is a runtime argument
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    std::vector<int> piv(n);

    // n first; every operand is a pointer plus a stride
    const bool ok = Solver::solve_inplace(n, A.data(), 1, piv.data(), 1, y.data(), 1);
    // snippet end

    return ok ? snippets::check(y.data(), expected, n) : 1;
}
