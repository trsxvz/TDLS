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

int main() {
    const int n                         = 5;
    std::vector<double> A               = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                           1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    const std::vector<double> b1        = {12, 16, 27, 40, 50};
    const std::vector<double> b2        = {7, 8, 9, 10, 11};
    const std::vector<double> expected1 = {1, 2, 3, 4, 5};
    const std::vector<double> expected2 = {1, 1, 1, 1, 1};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    std::vector<int> piv(n);
    std::vector<double> x1(n), x2(n);

    // n is the first argument, every operand a pointer plus a stride; false means singular
    if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) return 1;

    // two right-hand sides on the same factorization
    Solver::substitute(n, A.data(), 1, piv.data(), 1, b1.data(), x1.data(), 1);
    Solver::substitute(n, A.data(), 1, piv.data(), 1, b2.data(), x2.data(), 1);
    // snippet end

    return snippets::check(x1.data(), expected1.data(), n) |
           snippets::check(x2.data(), expected2.data(), n);
}
