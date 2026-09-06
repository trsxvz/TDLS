/// \file
/// \brief Documentation snippet: the runtime solver reproduces the compile-time one.
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
    std::vector<double> A_dynamic      = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                          1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    double A_static[5 * 5]             = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                          1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    std::vector<double> y_dynamic      = {12, 16, 27, 40, 50};
    double y_static[5]                 = {12, 16, 27, 40, 50};
    const std::vector<double> expected = {1, 2, 3, 4, 5};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Dynamic         = tdls::TiledLUppSolverDynamic<double, config>;
    using Static          = tdls::TiledLUppSolverStatic<double, 5, config>;

    std::vector<int> piv_dynamic(n);
    int piv_static[5];

    // same shape, same configuration: the same arithmetic, bit for bit
    const bool ok_dynamic =
        Dynamic::solve_inplace(n, A_dynamic.data(), 1, piv_dynamic.data(), 1, y_dynamic.data(), 1);
    const bool ok_static =
        Static::solve_inplace<true, true, true>(A_static, 1, piv_static, 1, y_static, 1);
    // snippet end

    if (!ok_dynamic || !ok_static) return 1;
    return snippets::check(y_dynamic.data(), expected.data(), n) |
           snippets::check_bitwise(y_dynamic.data(), y_static, n) |
           snippets::check_bitwise(A_dynamic.data(), A_static, n * n) |
           snippets::check_bitwise(piv_dynamic.data(), piv_static, n);
}
