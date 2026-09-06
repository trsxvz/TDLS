/// \file
/// \brief Documentation snippet: the scalar types.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    float A_single[4 * 4]                  = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    float y_single[4]                      = {14, 14, 24, 33};
    const float expected_single[4]         = {1, 2, 3, 4};
    long double A_extended[4 * 4]          = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    long double y_extended[4]              = {14, 14, 24, 33};
    const long double expected_extended[4] = {1, 2, 3, 4};

    // snippet begin
    // float: the acceptable-pivot threshold defaults to 1e-4
    using Single = tdls::TiledLUppSolverStatic<float, 4>;
    static_assert(Single::oot_threshold == 1e-4f);

    // long double: the thresholds are written as double literals and stored exactly
    constexpr auto extended =
        tdls::TiledLUppConfig<long double>{.oot_threshold = 1e-12, .singular_floor = 1e-300};
    using Extended = tdls::TiledLUppSolverStatic<long double, 4, extended>;
    static_assert(Extended::oot_threshold == static_cast<long double>(1e-12));

    int piv[4];

    const bool ok_single =
        Single::solve_inplace<true, true, true>(A_single, 1, piv, 1, y_single, 1);
    const bool ok_extended =
        Extended::solve_inplace<true, true, true>(A_extended, 1, piv, 1, y_extended, 1);
    // snippet end

    if (!ok_single || !ok_extended) return 1;
    return snippets::check(y_single, expected_single, 4, 1e-4) |
           snippets::check(y_extended, expected_extended, 4);
}
