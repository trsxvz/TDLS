/// \file
/// \brief Documentation snippet: pass cutting, runtime dimension.
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
    const std::vector<double> B        = {12, 16, 27, 40, 50, 7, 8, 9,  10, 11, 6, 2, 7,
                                          2,  10, 1,  6,  2,  8, 1, 10, 2,  0,  0, 2};
    const std::vector<double> expected = {1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 0, 1,
                                          0, 1, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    std::vector<int> piv(n);

    if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) return 1;

    // passes of two: nrhs at run time, the pass width as template argument
    const int nrhs = 5;
    std::vector<double> X(nrhs * n);
    Solver::substitute_multirhs<2>(n, nrhs, A.data(), 1, piv.data(), 1, B.data(), X.data(), 1, n);
    // snippet end

    return snippets::check(X.data(), expected.data(), nrhs * n);
}
