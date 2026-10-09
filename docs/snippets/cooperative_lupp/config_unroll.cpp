/// \file
/// \brief Documentation snippet: the unroll policy.
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
    double A_unrolled[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A_rolled[4 * 4]   = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y_unrolled[4]     = {14, 14, 24, 33};
    double y_rolled[4]       = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    int ok[2]                = {};

    // snippet begin
    constexpr auto unrolled = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    constexpr auto rolled =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2, .unroll_loops = false};

    using Unrolled = tdls::CooperativeLUppSolverStatic<double, 4, unrolled>;
    using Rolled   = tdls::CooperativeLUppSolverStatic<double, 4, rolled>;

    double work[Unrolled::workspace_size];
    int piv[4];

    // no unroll pragma: same values, smaller code and faster builds, the choice for CPU code
    snippets::run_group<Unrolled::threads_per_system>([&](const int tx, auto&& sync) {
        ok[tx] = Unrolled::solve_inplace<false, false, false>(tx, A_unrolled, 1, piv, 1, y_unrolled,
                                                              1, work, sync);
        ok[tx] = Rolled::solve_inplace<false, false, false>(tx, A_rolled, 1, piv, 1, y_rolled, 1,
                                                            work, sync) &&
                 ok[tx];
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    if (std::memcmp(y_unrolled, y_rolled, sizeof y_rolled) != 0) return 1;
    return snippets::check(y_rolled, expected, 4);
}
