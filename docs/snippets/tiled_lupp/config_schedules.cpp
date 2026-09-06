/// \file
/// \brief Documentation snippet: the elimination schedules.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A_right[4 * 4]    = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A_left[4 * 4]     = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y_right[4]        = {14, 14, 24, 33};
    double y_left[4]         = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    constexpr auto right =
        tdls::TiledLUppConfig<double>{.tile_size = 2, .schedule = tdls::Schedule::RightLooking};
    constexpr auto left =
        tdls::TiledLUppConfig<double>{.tile_size = 2, .schedule = tdls::Schedule::LeftLooking};

    using RightLooking = tdls::TiledLUppSolverStatic<double, 4, right>;
    using LeftLooking  = tdls::TiledLUppSolverStatic<double, 4, left>;

    int piv_right[4], piv_left[4];

    // the same factors from two update orders
    const bool ok_right =
        RightLooking::solve_inplace<true, true, true>(A_right, 1, piv_right, 1, y_right, 1);
    const bool ok_left =
        LeftLooking::solve_inplace<true, true, true>(A_left, 1, piv_left, 1, y_left, 1);
    // snippet end

    if (!ok_right || !ok_left) return 1;
    return snippets::check(y_right, expected, 4) | snippets::check(y_left, expected, 4);
}
