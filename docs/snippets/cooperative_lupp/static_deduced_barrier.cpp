/// \file
/// \brief Documentation snippet: the barrier deduced by the solver, and
/// the same barrier in the exchanges of the caller.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <thread>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]             = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double b1[4]          = {14, 14, 24, 33};
    const double b2_values[4]   = {7, 7, 8, 10};
    const double expected_x1[4] = {1, 2, 3, 4};
    const double expected_x2[4] = {1, 1, 1, 1};
    using Solver                = tdls::CooperativeLUppSolverStatic<
        double, 4, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2}>;
    double b2[4] = {}, x1[4] = {}, x2[4] = {};
    int ok[2] = {};

    // snippet begin
    // the workspace of the group, zero before the first call: its last 2 elements hold the
    // barrier the solver deduces for 2 CPU threads
    double work[Solver::workspace_size] = {};
    int piv[4];
    std::thread group[2];
    for (int tx = 0; tx < 2; ++tx)
        group[tx] = std::thread([&, tx] {
            // no barrier argument: the solver deduces it
            ok[tx] = Solver::factorize<false, false>(tx, A, 1, piv, 1, work);
            Solver::substitute<false, false, false>(tx, A, 1, piv, 1, b1, x1, 1, work);

            // the same barrier for an exchange of the caller: each thread writes the
            // right-hand side of the rows of the other one, then waits for it
            auto sync = Solver::make_sync(work);
            for (int r = 1 - tx; r < 4; r += 2)
                b2[r] = b2_values[r];
            sync();
            Solver::substitute<false, false, false>(tx, A, 1, piv, 1, b2, x2, 1, work, sync);
        });
    for (auto& thread : group)
        thread.join();
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(x1, expected_x1, 4) + snippets::check(x2, expected_x2, 4);
}
