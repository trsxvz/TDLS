/// \file
/// \brief Documentation snippet: a barrier passed by the caller, used as
/// is by the solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    int ok[2]                = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size]; // no initial value needed with an explicit barrier
    int piv[4];

    // the barrier of the caller: any callable taking no argument, called by every thread of
    // the group, that makes the writes of each thread visible to all of them. Here a barrier
    // of CPU threads; on GPU, __syncthreads, a SYCL group barrier or a Kokkos team barrier
    snippets::Barrier barrier(Solver::threads_per_system);
    snippets::run_group<Solver::threads_per_system>([&](const int tx) {
        auto sync = [&barrier] { barrier.arrive_and_wait(); };
        ok[tx]    = Solver::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work, sync);
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(y, expected, 4);
}
