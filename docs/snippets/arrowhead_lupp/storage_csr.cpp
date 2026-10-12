/// \file
/// \brief Documentation snippet: the Csr storage, its canonical index arrays.
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
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Csr};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Map    = Solver::storage_map;

    // the canonical CSR arrays of the pattern: row pointers and column indices
    int row_ptr[n + 1], col_idx[Solver::storage_size];
    Map::csr_pattern(shape, row_ptr, col_idx);

    // the values in that order: any CSR matrix of this pattern is already in the storage
    double values[Solver::storage_size];
    for (int r = 0; r < n; ++r)
        for (int e = row_ptr[r]; e < row_ptr[r + 1]; ++e)
            values[e] = A[r * n + col_idx[e]];

    // the solver reads the values only: the positions follow from the shape
    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(values, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
