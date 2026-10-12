/// \file
/// \brief Documentation snippet: the column-major layout.
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
    // A stored by columns: element (r, c) at c * n + r
    double A_by_columns[n * n];
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            A_by_columns[c * n + r] = A[r * n + c];

    // snippet begin
    // the matrix stored column-major; the blocks are factored as stored, by TiledLUpp in
    // its column-major mode, and the pivot check reads the forward stage of the solve
    constexpr auto config =
        tdls::ArrowheadLUppConfig<double>{.coupling = tdls::ArrowheadCoupling::Diagonal,
                                          .storage  = tdls::ArrowheadStorage::Full,
                                          .layout   = tdls::MatrixLayout::ColMajor};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(A_by_columns, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
