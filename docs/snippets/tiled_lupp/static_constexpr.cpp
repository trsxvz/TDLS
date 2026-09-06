/// \file
/// \brief Documentation snippet: constant evaluation.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

// snippet begin
constexpr double solve_at_compile_time() {
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    double A[4 * 4] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]     = {14, 14, 24, 33};
    int piv[4];

    if (!Solver::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1)) return 0;
    return y[3];
}

// the whole solve runs during constant evaluation
constexpr double x3 = solve_at_compile_time();
static_assert(x3 > 4 - 1e-12 && x3 < 4 + 1e-12);
// snippet end

int main() {
    return 0;
}
