/// \file
/// \brief Documentation snippet: the scalar types.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    float A_single[4 * 4]                  = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    float y_single[4]                      = {14, 14, 24, 33};
    const float expected_single[4]         = {1, 2, 3, 4};
    long double A_extended[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    long double y_extended[4]              = {14, 14, 24, 33};
    const long double expected_extended[4] = {1, 2, 3, 4};

    // snippet begin
    // the configuration carries the scalar type of the solver
    using Single =
        tdls::CooperativeLUppSolverStatic<float, 4,
                                          tdls::CooperativeLUppConfig<float>{.rows_per_thread = 4}>;
    using Extended = tdls::CooperativeLUppSolverStatic<
        long double, 4, tdls::CooperativeLUppConfig<long double>{.rows_per_thread = 4}>;

    float work_single[Single::workspace_size];
    long double work_extended[Extended::workspace_size];
    int piv[4];
    const bool ok1 =
        Single::solve_inplace<true, true, true>(0, A_single, 1, piv, 1, y_single, 1, work_single);
    const bool ok2 = Extended::solve_inplace<true, true, true>(0, A_extended, 1, piv, 1, y_extended,
                                                               1, work_extended);
    // snippet end

    if (!ok1 || !ok2) return 1;
    return snippets::check(y_single, expected_single, 4, 1e-5) |
           snippets::check(y_extended, expected_extended, 4);
}
