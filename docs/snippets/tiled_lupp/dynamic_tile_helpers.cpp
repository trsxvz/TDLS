/// \file
/// \brief Documentation snippet: the tile helpers.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const int n                        = 5;
    std::vector<double> A              = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                          1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    std::vector<double> y              = {12, 16, 27, 40, 50};
    const std::vector<double> expected = {1, 2, 3, 4, 5};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    // the grid of a runtime dimension: three tiles per dimension for n = 5, the last of size 1
    const int tiles = Solver::num_tiles(n);
    const int last  = Solver::tile_size_at((tiles - 1) * Solver::tile_size, n);

    std::vector<int> piv(n);
    const bool ok = Solver::solve_inplace(n, A.data(), 1, piv.data(), 1, y.data(), 1);
    // snippet end

    if (tiles != 3 || last != 1 || !ok) return 1;
    return snippets::check(y.data(), expected.data(), n);
}
