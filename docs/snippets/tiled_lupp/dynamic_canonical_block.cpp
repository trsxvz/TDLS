/// \file
/// \brief Documentation snippet: a block of canonical columns, runtime dimension.
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
    const double e[3][5]         = {{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    std::vector<int> piv(n);

    if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) return 1;

    // column w of X starts at X + w * n: element stride 1, column stride n
    const int nrhs = 3;
    std::vector<double> X(nrhs * n);
    Solver::substitute_canonical_multirhs(n, nrhs, A.data(), 1, piv.data(), 1, 0, X.data(), 1, n);
    // snippet end

    int status = 0;
    for (int w = 0; w < nrhs; ++w)
        status |= snippets::check_product(A0.data(), X.data() + w * n, e[w], n);
    return status;
}
