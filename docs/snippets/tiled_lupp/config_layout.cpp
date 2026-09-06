/// \file
/// \brief Documentation snippet: the column-major layout.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double A0[4 * 4]   = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double y[4]              = {14, 14, 24, 33};
    const double expected[4] = {1, 2, 3, 4};
    double batch[4 * 4 * 3]  = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            batch[(c * 4 + r) * 3 + 1] = A0[r * 4 + c];

    // snippet begin
    constexpr auto colmajor =
        tdls::TiledLUppConfig<double>{.tile_size = 2, .layout = tdls::MatrixLayout::ColMajor};
    using Solver = tdls::TiledLUppSolverStatic<double, 4, colmajor>;

    // system 1 of an SoA batch of 3: element k of the matrix sits at batch[k * 3 + 1],
    // with k = c * 4 + r under the column-major layout
    double* A        = batch + 1;
    const int stride = 3;

    int piv[4];

    const bool ok = Solver::solve_inplace<true, true, false>(A, stride, piv, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, 4) : 1;
}
