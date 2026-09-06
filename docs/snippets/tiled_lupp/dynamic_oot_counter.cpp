/// \file
/// \brief Documentation snippet: the out-of-tile counter, runtime dimension.
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
    std::vector<double> A              = {1e-12, 1, 0, 0, 1, 2e-12, 6, 1, 0, 0, 3, 1, 7,
                                          1,     0, 0, 0, 1, 8,     1, 1, 0, 0, 1, 9};
    const std::vector<double> b        = {7 + 1e-12, 15 + 2e-12, 30, 40, 50};
    const std::vector<double> expected = {1, 2, 3, 4, 5};
    std::vector<double> x(n);

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    std::vector<int> piv(n);

    // one more argument: the columns whose best in-tile pivot fell below the threshold
    int oot_count;
    if (!Solver::factorize(n, A.data(), 1, piv.data(), 1, oot_count)) return 1;

    const bool searched_below_the_tile = oot_count == 1 && piv[0] == 2;
    // snippet end

    if (!searched_below_the_tile) return 1;
    Solver::substitute(n, A.data(), 1, piv.data(), 1, b.data(), x.data(), 1);
    return snippets::check(x.data(), expected.data(), n);
}
