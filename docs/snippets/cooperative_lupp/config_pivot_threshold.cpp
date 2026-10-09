/// \file
/// \brief Documentation snippet: the relative pivot threshold.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    const double A0[4 * 4]   = {3, 1, 0, 0, 4, 6, 1, 0, 0, 1, 5, 1, 0, 0, 1, 4};
    const double b0[4]       = {5, 19, 21, 19};
    const double expected[4] = {1, 2, 3, 4};
    double A1[4 * 4], A2[4 * 4], y1[4], y2[4];
    for (int k = 0; k < 4 * 4; ++k)
        A1[k] = A2[k] = A0[k];
    for (int i = 0; i < 4; ++i)
        y1[i] = y2[i] = b0[i];
    int ok[2] = {};

    // snippet begin
    // column 0: the row in place holds 3, row 1 holds 4. The partial
    // pivoting of LAPACK, a threshold of 1, takes 4; the default threshold
    // 0.1 keeps 3, since 3 >= 0.1 * 4
    constexpr auto lapack =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2, .relative_pivot_threshold = 1.0};
    constexpr auto relaxed = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Lapack           = tdls::CooperativeLUppSolverStatic<double, 4, lapack>;
    using Relaxed          = tdls::CooperativeLUppSolverStatic<double, 4, relaxed>;

    double work[Lapack::workspace_size] = {};
    int piv1[4], piv2[4];
    snippets::run_group<2>([&](const int tx) {
        ok[tx] = Lapack::solve_inplace<false, false, false>(tx, A1, 1, piv1, 1, y1, 1, work);
    });
    snippets::run_group<2>([&](const int tx) {
        ok[tx] =
            Relaxed::solve_inplace<false, false, false>(tx, A2, 1, piv2, 1, y2, 1, work) && ok[tx];
    });

    // LAPACK exchanges rows 0 and 1; under the threshold every row stays at
    // its position, and the multipliers stay below 1 / 0.1 = 10
    const bool exchanged = piv1[0] == 1 && piv1[1] == 0;
    const bool in_place  = piv2[0] == 0 && piv2[1] == 1 && piv2[2] == 2 && piv2[3] == 3;
    // snippet end

    if (!ok[0] || !ok[1] || !exchanged || !in_place) return 1;
    return snippets::check(y1, expected, 4) | snippets::check(y2, expected, 4);
}
