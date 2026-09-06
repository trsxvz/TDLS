/// \file
/// \brief Documentation snippet: a tile larger than the dimension, runtime dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const int n                        = 3;
    std::vector<double> A              = {3, 1, 0, 1, 4, 1, 0, 1, 5};
    std::vector<double> y              = {5, 12, 17};
    const std::vector<double> expected = {1, 2, 3};

    // snippet begin
    // the grid is a single partial tile: 3 of its 8 x 8 slots are used, the others never touched
    constexpr auto wide = tdls::TiledLUppConfig<double>{.tile_size = 8};
    using Solver        = tdls::TiledLUppSolverDynamic<double, wide>;

    std::vector<int> piv(n);
    const bool ok = Solver::solve_inplace(n, A.data(), 1, piv.data(), 1, y.data(), 1);
    // snippet end

    return ok ? snippets::check(y.data(), expected.data(), n) : 1;
}
