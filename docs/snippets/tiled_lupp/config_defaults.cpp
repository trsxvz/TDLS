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

int main() {
    double A[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double A2[4 * 4]         = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    double y2[4]             = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};

    // snippet begin
    // every knob at its default: tiles of 3, right-looking, default thresholds
    constexpr tdls::TiledLUppConfig<double> config{};

    using Static  = tdls::TiledLUppSolverStatic<double, 4, config>;
    using Dynamic = tdls::TiledLUppSolverDynamic<double, config>;

    // the configuration argument of both solvers defaults to that value
    static_assert(std::is_same_v<Static, tdls::TiledLUppSolverStatic<double, 4>>);
    static_assert(std::is_same_v<Dynamic, tdls::TiledLUppSolverDynamic<double>>);

    int piv[4];

    // y holds b on entry and x on exit
    const bool ok_static  = Static::solve_inplace<true, true, true>(A, 1, piv, 1, y, 1);
    const bool ok_dynamic = Dynamic::solve_inplace(4, A2, 1, piv, 1, y2, 1);
    // snippet end

    if (!ok_static || !ok_dynamic) return 1;
    return snippets::check(y, expected, 4) | snippets::check(y2, expected, 4);
}
