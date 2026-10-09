/// \file
/// \brief Documentation snippet: a group of 2 CPU threads, with
/// std::thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <thread>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    int ok[2]                = {};

    // snippet begin
    // 2 rows per thread for a 4 x 4 system: a group of 2 threads solves it
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    // shared by the group. On CPU, its first two elements must be zero before the first
    // call: they hold the barrier that the solver deduces
    double work[Solver::workspace_size] = {};
    int piv[4];

    // the group: 2 threads, of ranks tx = 0 and 1, that call the solver together; each one
    // captures its own rank, the rest by reference
    std::thread group[Solver::threads_per_system];
    for (int tx = 0; tx < Solver::threads_per_system; ++tx)
        group[tx] = std::thread([&, tx] {
            ok[tx] = Solver::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work);
        });
    // wait until both threads are done
    for (auto& thread : group)
        thread.join();
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(y, expected, 4);
}
