/// \file
/// \brief Documentation snippet: the TiledLUpp solvers of the blocks and of the border.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"

int main() {
    using namespace snippets::arrow;
    double y[n] = {};
    for (int i = 0; i < n; ++i)
        y[i] = b[i];

    // snippet begin
    // the blocks: TiledLUpp in tiles of 2, a whole block, the row in place kept down to 0.5 of
    // the best in-tile candidate; the border system: TiledLUpp in tiles of 2
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling        = tdls::ArrowheadCoupling::Diagonal,
                                          .storage         = tdls::ArrowheadStorage::Packed,
                                          .block_tile_size = 2,
                                          .block_relative_pivot_threshold = 0.5,
                                          .block_oot_pivot_threshold      = 1e-10,
                                          .border_tile_size               = 2};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A, M);
    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
