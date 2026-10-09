/// \file
/// \brief Documentation snippet: the column-major layout.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    const double A0[4 * 4]   = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    double batch[4 * 4 * 3]  = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            batch[(c * 4 + r) * 3 + 1] = A0[r * 4 + c];
    int ok[2] = {};

    // snippet begin
    // element (r, c) of system 1 at batch[(c * 4 + r) * 3 + 1]: column-major, stride 3
    constexpr auto config = tdls::CooperativeLUppConfig<double>{
        .rows_per_thread = 2, .layout = tdls::MatrixLayout::ColMajor};
    using Solver = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size] = {};
    int piv[4];
    snippets::run_group<Solver::threads_per_system>([&](const int tx) {
        ok[tx] = Solver::solve_inplace<false, false, false>(tx, batch + 1, 3, piv, 1, y, 1, work);
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(y, expected, 4);
}
