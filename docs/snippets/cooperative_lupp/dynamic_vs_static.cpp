/// \file
/// \brief Documentation snippet: the runtime solver against the
/// compile-time one, bitwise, one thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const int n                   = 5;
    std::vector<double> A_dynamic = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                     1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    double A_static[5 * 5];
    for (int k = 0; k < 5 * 5; ++k)
        A_static[k] = A_dynamic[k];
    std::vector<double> y_dynamic = {12, 16, 27, 40, 50};
    double y_static[5]            = {12, 16, 27, 40, 50};

    // snippet begin
    // one configuration for both solvers: 5 rows per thread, one thread for n = 5
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 5};
    using Dynamic         = tdls::CooperativeLUppSolverDynamic<double, config>;
    using Static          = tdls::CooperativeLUppSolverStatic<double, 5, config>;

    std::vector<double> work(Dynamic::workspace_size(n));
    std::vector<int> piv_dynamic(n);
    int piv_static[5];

    // same shape, same configuration: the same arithmetic, bit for bit
    const bool ok_dynamic = Dynamic::solve_inplace(n, 0, A_dynamic.data(), 1, piv_dynamic.data(), 1,
                                                   y_dynamic.data(), 1, work.data());
    const bool ok_static = Static::solve_inplace<false, false, false>(0, A_static, 1, piv_static, 1,
                                                                      y_static, 1, work.data());
    // snippet end

    if (!ok_dynamic || !ok_static) return 1;
    return snippets::check_bitwise(y_dynamic.data(), y_static, n);
}
