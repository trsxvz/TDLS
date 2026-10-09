/// \file
/// \brief Documentation snippet: the number of rows per thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <cstring>

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    const double A0[6 * 6]   = {8, 1, 0, 0, 2, 0, 1, 8, 1, 0, 0, 2, 0, 1, 8, 1, 0, 0,
                                0, 0, 1, 8, 1, 0, 2, 0, 0, 1, 8, 1, 0, 2, 0, 0, 1, 8};
    const double b0[6]       = {20, 32, 30, 40, 52, 57};
    const double expected[6] = {1, 2, 3, 4, 5, 6};
    double A1[6 * 6], A2[6 * 6], A3[6 * 6], y1[6], y2[6], y3[6];
    for (int k = 0; k < 6 * 6; ++k)
        A1[k] = A2[k] = A3[k] = A0[k];
    for (int i = 0; i < 6; ++i)
        y1[i] = y2[i] = y3[i] = b0[i];
    int ok[6] = {};

    // snippet begin
    // the same 6 x 6 system on 6 threads of 1 row, 3 threads of 2 rows and 2 threads of 3 rows
    using One = tdls::CooperativeLUppSolverStatic<
        double, 6, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 1}>;
    using Two = tdls::CooperativeLUppSolverStatic<
        double, 6, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2}>;
    using Three = tdls::CooperativeLUppSolverStatic<
        double, 6, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3}>;
    static_assert(One::threads_per_system == 6 && Two::threads_per_system == 3 &&
                  Three::threads_per_system == 2);

    // thread tx holds rows tx, tx + threads, ...: on 3 threads, thread 0 holds rows 0 and 3
    double work[One::workspace_size] = {};
    int piv[6];
    snippets::run_group<One::threads_per_system>([&](const int tx) {
        ok[tx] = One::solve_inplace<false, false, false>(tx, A1, 1, piv, 1, y1, 1, work);
    });
    snippets::run_group<Two::threads_per_system>([&](const int tx) {
        ok[tx] = Two::solve_inplace<false, false, false>(tx, A2, 1, piv, 1, y2, 1, work) && ok[tx];
    });
    snippets::run_group<Three::threads_per_system>([&](const int tx) {
        ok[tx] =
            Three::solve_inplace<false, false, false>(tx, A3, 1, piv, 1, y3, 1, work) && ok[tx];
    });

    // the mapping changes nothing in the arithmetic: the three solutions are bitwise identical
    const bool identical =
        std::memcmp(y1, y2, sizeof y1) == 0 && std::memcmp(y1, y3, sizeof y1) == 0;
    // snippet end

    if (!ok[0] || !ok[1] || !identical) return 1;
    return snippets::check(y1, expected, 6);
}
