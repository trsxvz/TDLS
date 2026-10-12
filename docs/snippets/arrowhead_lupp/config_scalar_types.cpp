/// \file
/// \brief Documentation snippet: the scalar type.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    float A[6 * 6]          = {6, 1, 1, 0, 2, 0, 1, 7, 0, 1, 0, 1, 1, 0, 5, 1, 0, 0,
                               0, 2, 1, 4, 0, 0, 2, 1, 0, 0, 6, 2, 0, 1, 0, 0, 1, 5};
    float y[6]              = {21, 25, 20, 23, 46, 37};
    const float expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    // float throughout; the floating-point knobs are written as float values
    constexpr auto shape  = tdls::ArrowheadShape{.border = 2, .block = 2, .blocks = 2};
    constexpr auto config = tdls::ArrowheadLUppConfig<float>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .condensation_pivot_threshold = 0.01f};
    using Solver = tdls::ArrowheadLUppSolverStatic<float, shape, config>;

    float work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 6, 1e-5) : 1;
}
