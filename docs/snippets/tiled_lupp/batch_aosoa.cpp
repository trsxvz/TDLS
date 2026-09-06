/// \file
/// \brief Documentation snippet: an AoSoA batch.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "batch.hpp"
#include "check.hpp"

int main() {
    using namespace snippets;
    // blocks of W = 2 systems, the batch padded to 4 systems so that every block is full
    constexpr int W          = 2;
    constexpr int padded     = 4;
    double A[N * N * padded] = {}, y[N * padded] = {};
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A[(s / W) * N * N * W + k * W + s % W] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y[(s / W) * N * W + i * W + s % W] = rhs(s, i);
    }

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    // block s / 2, slot s % 2: the stride is the width 2
    for (int s = 0; s < count; ++s) {
        int piv[4];
        double* A_s = A + (s / 2) * 16 * 2 + s % 2;
        double* y_s = y + (s / 2) * 4 * 2 + s % 2;
        if (!Solver::solve_inplace(4, A_s, 2, piv, 1, y_s, 2)) return 1;
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= snippets::check(y + (s / W) * N * W + s % W, W, expected, N);
    return status;
}
