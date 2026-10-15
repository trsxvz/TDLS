/// \file
/// \brief Documentation snippet: every knob at its default.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[6 * 6]          = {6, 1, 1, 0, 2, 0, 1, 7, 0, 1, 0, 1, 1, 0, 5, 1, 0, 0,
                                0, 2, 1, 4, 0, 0, 2, 1, 0, 0, 6, 2, 0, 1, 0, 0, 1, 5};
    double y[6]              = {21, 25, 20, 23, 46, 37};
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    // dense couplings, Full storage, row-major, pivot check at 0.01; the blocks in registers when
    // they fit, by tiles of 4 otherwise, the passes chosen for the target
    constexpr tdls::ArrowheadLUppConfig<double> config{};
    static_assert(config.coupling == tdls::ArrowheadCoupling::Dense);
    static_assert(config.storage == tdls::ArrowheadStorage::Full);
    static_assert(config.block_row_interchange == tdls::RowInterchange::Auto);
    static_assert(config.block_tile_size == 4 && config.rhs_pass_width == -1);
    static_assert(!config.gather_rhs && !config.schur_in_place);

    constexpr auto shape = tdls::ArrowheadShape{.border = 2, .block = 2, .blocks = 2};
    using Solver         = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(A, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 6) : 1;
}
