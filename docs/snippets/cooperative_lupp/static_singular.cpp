/// \file
/// \brief Documentation snippet: the singular verdict, compile-time dimension.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "group.hpp"

int main() {
    double A[4 * 4] = {4, 1, 0, 2, 1, 5, 0, 0, 0, 1, 0, 1, 2, 0, 0, 7};
    double y[4]     = {14, 14, 24, 33};
    int ok[2]       = {1, 1};

    // snippet begin
    // column 2 is zero: the matrix is singular
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size] = {};
    int piv[4];

    // every thread runs the whole sequence of barriers and receives the same verdict;
    // the content of y is then unspecified
    snippets::run_group<Solver::threads_per_system>([&](const int tx) {
        ok[tx] = Solver::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work);
    });
    // snippet end

    return ok[0] || ok[1] ? 1 : 0;
}
