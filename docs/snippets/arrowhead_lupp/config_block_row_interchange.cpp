/// \file
/// \brief Documentation snippet: the blocks eliminated in registers
/// (block_row_interchange).
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
    // each block factored by TiledLUpp with physical row interchanges, every index known at
    // compile time: the block and its right-hand sides stay in registers on GPU instead of the
    // stack (what Auto, the default, chooses for blocks that fit; Physical forces it)
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling = tdls::ArrowheadCoupling::Diagonal,
                                          .storage  = tdls::ArrowheadStorage::Packed,
                                          .block_row_interchange = tdls::RowInterchange::Physical};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A, M);
    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
