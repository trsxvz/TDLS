/// \file
/// \brief Documentation snippet: the solver constants.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <limits>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[6 * 6]          = {8, 1, 0, 0, 2, 0, 1, 8, 1, 0, 0, 2, 0, 1, 8, 1, 0, 0,
                                0, 0, 1, 8, 1, 0, 2, 0, 0, 1, 8, 1, 0, 2, 0, 0, 1, 8};
    double y[6]              = {20, 32, 30, 40, 52, 57};
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    constexpr auto config =
        tdls::TiledLUppConfig<double>{.tile_size = 4, .schedule = tdls::Schedule::LeftLooking};
    using Solver = tdls::TiledLUppSolverStatic<double, 6, config>;

    // the tile grid: tile size, full tiles per dimension, extent of the trailing tile
    static_assert(Solver::TS == 4 && Solver::F == 1 && Solver::TAIL == 2);

    // the knobs, read back from the solver type
    static_assert(Solver::schedule == tdls::Schedule::LeftLooking);
    static_assert(Solver::oot_threshold == 1e-10);
    static_assert(Solver::singular_floor == std::numeric_limits<double>::min());

    int piv[6];

    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 6) : 1;
}
