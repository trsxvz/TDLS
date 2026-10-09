/// \file
/// \brief Documentation snippet: the runtime solver reproduces the compile-time one.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <cstring>
#include <vector>

#include <tdls/tdls.hpp>

#include "group.hpp"

int main() {
    const int n                   = 5;
    std::vector<double> A_dynamic = {5, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 1, 7,
                                     1, 0, 0, 0, 1, 8, 1, 1, 0, 0, 1, 9};
    double A_static[5 * 5];
    for (int k = 0; k < 5 * 5; ++k)
        A_static[k] = A_dynamic[k];
    std::vector<double> y_dynamic = {12, 16, 27, 40, 50};
    double y_static[5]            = {12, 16, 27, 40, 50};
    std::vector<int> ok(3, 0);

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Dynamic         = tdls::CooperativeLUppSolverDynamic<double, config>;
    using Static          = tdls::CooperativeLUppSolverStatic<double, 5, config>;

    double work[Static::workspace_size] = {};
    std::vector<int> piv_dynamic(n);
    int piv_static[5];

    // same shape, same configuration: the same arithmetic, bit for bit
    snippets::run_group<Static::threads_per_system>([&](const int tx) {
        ok[tx] = Dynamic::solve_inplace(n, tx, A_dynamic.data(), 1, piv_dynamic.data(), 1,
                                        y_dynamic.data(), 1, work);
        ok[tx] = Static::solve_inplace<false, false, false>(tx, A_static, 1, piv_static, 1,
                                                            y_static, 1, work) &&
                 ok[tx];
    });
    // snippet end

    if (!ok[0] || !ok[1] || !ok[2]) return 1;
    return std::memcmp(y_dynamic.data(), y_static, sizeof y_static) == 0 &&
                   std::memcmp(A_dynamic.data(), A_static, sizeof A_static) == 0 &&
                   std::memcmp(piv_dynamic.data(), piv_static, sizeof piv_static) == 0
               ? 0
               : 1;
}
