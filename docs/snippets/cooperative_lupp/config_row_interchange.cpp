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

int main() {
    const double A0[4 * 4]   = {0, 2, 0, 1, 4, 1, 1, 0, 0, 1, 3, 1, 2, 0, 1, 5};
    const double b0[4]       = {8, 9, 15, 25};
    const double expected[4] = {1, 2, 3, 4};
    double A1[4 * 4], A2[4 * 4], y1[4], y2[4];
    for (int k = 0; k < 4 * 4; ++k)
        A1[k] = A2[k] = A0[k];
    for (int i = 0; i < 4; ++i)
        y1[i] = y2[i] = b0[i];

    // snippet begin
    // the same system under both row interchanges, with one thread
    constexpr auto logical  = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    constexpr auto physical = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = 4, .row_interchange = tdls::RowInterchange::Physical};
    using Logical  = tdls::CooperativeLUppSolverStatic<double, 4, logical>;
    using Physical = tdls::CooperativeLUppSolverStatic<double, 4, physical>;

    // 2 elements for the deduced barrier, then 3 N under logical row interchanges,
    // 2 N + 2 threads_per_system + 5 under physical ones
    static_assert(Logical::workspace_size == 14 && Physical::workspace_size == 17);
    double work[Physical::workspace_size];
    int piv1[4], piv2[4];
    const bool ok1 = Logical::solve_inplace<true, true, true>(0, A1, 1, piv1, 1, y1, 1, work);
    const bool ok2 = Physical::solve_inplace<true, true, true>(0, A2, 1, piv2, 1, y2, 1, work);

    // same pivots, same operations: the solutions are bitwise identical
    const bool identical = std::memcmp(y1, y2, sizeof y1) == 0;
    // under the default row-major layout the solver factors A^T: its rows
    // are the columns of A, and column 1 holds the pivot of row 0. Logical:
    // the columns stay in place and piv1[v] is the position of column v.
    // Physical: the columns move, column k of A2 is the factored vector at
    // position k and piv2[k] its original index.
    bool placed = true;
    for (int v = 0; v < 4; ++v) {
        placed = placed && piv2[piv1[v]] == v;
        for (int i = 0; i < 4; ++i)
            placed = placed && A1[i * 4 + v] == A2[i * 4 + piv1[v]];
    }
    // snippet end

    if (!ok1 || !ok2 || !identical || !placed || piv1[0] == 0) return 1;
    return snippets::check(y1, expected, 4);
}
