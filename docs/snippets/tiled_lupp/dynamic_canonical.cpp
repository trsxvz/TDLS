/// \file
/// \brief Documentation snippet: one canonical column, runtime dimension.
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
    const std::vector<double> A0 = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                    1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    std::vector<double> A        = A0;
    const std::vector<double> e2 = {0, 0, 1, 0, 0};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    std::vector<int> piv(n);
    std::vector<double> x(n);

    if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) return 1;

    // x is column 2 of A^-1
    Solver::substitute_canonical(n, A.data(), 1, piv.data(), 1, 2, x.data(), 1);
    // snippet end

    return snippets::check_product(A0.data(), x.data(), e2.data(), n);
}
