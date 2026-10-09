/// \file
/// \brief Documentation snippet: the row interchanges.
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
    const double A0[4 * 4]   = {0, 2, 0, 1, 4, 1, 1, 0, 0, 1, 3, 1, 2, 0, 1, 5};
    const double b0[4]       = {8, 9, 15, 25};
    const double expected[4] = {1, 2, 3, 4};
    double A1[4 * 4], A2[4 * 4], y1[4], y2[4];
    for (int k = 0; k < 4 * 4; ++k)
        A1[k] = A2[k] = A0[k];
    for (int i = 0; i < 4; ++i)
        y1[i] = y2[i] = b0[i];
    int ok[2] = {};

    // snippet begin
    // the same system under both row interchanges, on 2 threads of 2 rows
    constexpr auto logical  = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    constexpr auto physical = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = 2, .row_interchange = tdls::RowInterchange::Physical};
    using Logical  = tdls::CooperativeLUppSolverStatic<double, 4, logical>;
    using Physical = tdls::CooperativeLUppSolverStatic<double, 4, physical>;

    // 2 elements for the deduced barrier, then 3 N under logical row interchanges,
    // 2 N + 2 threads_per_system + 5 under physical ones
    static_assert(Logical::workspace_size == 14 && Physical::workspace_size == 19);
    double work[Physical::workspace_size];
    int piv1[4], piv2[4];
    snippets::run_group<2>([&](const int tx, auto&& sync) {
        ok[tx] = Logical::solve_inplace<false, false, false>(tx, A1, 1, piv1, 1, y1, 1, work, sync);
    });
    snippets::run_group<2>([&](const int tx, auto&& sync) {
        ok[tx] =
            Physical::solve_inplace<false, false, false>(tx, A2, 1, piv2, 1, y2, 1, work, sync) &&
            ok[tx];
    });

    // same pivots, same operations: the solutions are bitwise identical
    const bool identical = std::memcmp(y1, y2, sizeof y1) == 0;
    // row 1 holds the pivot of column 0. Logical: rows stay in place and
    // piv1[r] is the position of row r. Physical: rows move, row k of A2 is
    // the factored row at position k and piv2[k] its original index.
    bool placed = true;
    for (int r = 0; r < 4; ++r)
        placed = placed && piv2[piv1[r]] == r &&
                 std::memcmp(&A1[r * 4], &A2[piv1[r] * 4], 4 * sizeof(double)) == 0;
    // snippet end

    if (!ok[0] || !ok[1] || !identical || !placed || piv1[0] == 0) return 1;
    return snippets::check(y1, expected, 4);
}
