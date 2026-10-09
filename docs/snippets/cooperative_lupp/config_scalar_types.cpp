/// \file
/// \brief Documentation snippet: the scalar types.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    float A_single[4 * 4]                  = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    float y_single[4]                      = {14, 14, 24, 33};
    const float expected_single[4]         = {1, 2, 3, 4};
    long double A_extended[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    long double y_extended[4]              = {14, 14, 24, 33};
    const long double expected_extended[4] = {1, 2, 3, 4};
    int ok[2]                              = {};

    // snippet begin
    // the configuration carries the scalar type of the solver
    using Single =
        tdls::CooperativeLUppSolverStatic<float, 4,
                                          tdls::CooperativeLUppConfig<float>{.rows_per_thread = 2}>;
    using Extended = tdls::CooperativeLUppSolverStatic<
        long double, 4, tdls::CooperativeLUppConfig<long double>{.rows_per_thread = 2}>;

    float work_single[Single::workspace_size] = {};
    long double work_extended[Extended::workspace_size];
    int piv[4];
    // long double has no lock-free atomics on most CPUs: the compiler refuses the deduced
    // barrier of CPU threads there, so the extended solver takes a barrier of the caller
    snippets::Barrier barrier(Extended::threads_per_system);
    snippets::run_group<Single::threads_per_system>([&](const int tx) {
        ok[tx]    = Single::solve_inplace<false, false, false>(tx, A_single, 1, piv, 1, y_single, 1,
                                                               work_single);
        auto sync = [&barrier] { barrier.arrive_and_wait(); };
        ok[tx] = Extended::solve_inplace<false, false, false>(tx, A_extended, 1, piv, 1, y_extended,
                                                              1, work_extended, sync) &&
                 ok[tx];
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    return snippets::check(y_single, expected_single, 4, 1e-5) |
           snippets::check(y_extended, expected_extended, 4);
}
