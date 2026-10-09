/// \file
/// \brief Documentation snippet: the default configuration.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <type_traits>

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A2[4 * 4]         = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    double y2[4]             = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    int ok[4] = {}, ok2[4] = {};

    // snippet begin
    // every knob at its default: one row per thread, so one thread per row
    constexpr tdls::CooperativeLUppConfig<double> config{};

    using Static  = tdls::CooperativeLUppSolverStatic<double, 4, config>;
    using Dynamic = tdls::CooperativeLUppSolverDynamic<double, config>;

    // the configuration argument of both solvers defaults to that value
    static_assert(std::is_same_v<Static, tdls::CooperativeLUppSolverStatic<double, 4>>);
    static_assert(std::is_same_v<Dynamic, tdls::CooperativeLUppSolverDynamic<double>>);

    // 4 rows of 1: a group of 4 threads and the workspace they share
    double work[Static::workspace_size];
    int piv[4];

    // every thread of the group makes the same call with its rank tx and
    // the barrier of the group; y holds b on entry and x on exit
    snippets::run_group<Static::threads_per_system>([&](const int tx, auto&& sync) {
        ok[tx] = Static::solve_inplace<false, false, false>(tx, A, 1, piv, 1, y, 1, work, sync);
    });
    snippets::run_group(Dynamic::threads_per_system(4), [&](const int tx, auto&& sync) {
        ok2[tx] = Dynamic::solve_inplace(4, tx, A2, 1, piv, 1, y2, 1, work, sync);
    });
    // snippet end

    for (int tx = 0; tx < 4; ++tx)
        if (!ok[tx] || !ok2[tx]) return 1;
    return snippets::check(y, expected, 4) | snippets::check(y2, expected, 4);
}
