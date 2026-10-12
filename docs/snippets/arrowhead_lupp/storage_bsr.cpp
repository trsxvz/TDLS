/// \file
/// \brief Documentation snippet: the Bsr storage, its canonical block index arrays.
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
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Bsr};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Map    = Solver::storage_map;

    // blocks of 2 x 2: 3 block rows; the canonical BSR arrays of the pattern
    int block_ptr[3 + 1], block_col[Solver::storage_size / 4];
    Map::bsr_pattern(shape, block_ptr, block_col);

    // each 2 x 2 block stored by rows, the blocks in the order of the arrays
    double values[Solver::storage_size];
    for (int R = 0; R < 3; ++R)
        for (int e = block_ptr[R]; e < block_ptr[R + 1]; ++e)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    values[e * 4 + i * 2 + j] = A[(R * 2 + i) * n + block_col[e] * 2 + j];

    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(values, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
