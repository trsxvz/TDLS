/// \file
/// \brief Documentation snippet: an AoS batch.
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
    double A[N * N * count], y[N * count];
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A[s * N * N + k] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y[s * N + i] = rhs(s, i);
    }

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverDynamic<double, config>;

    // the objects of system s are contiguous, at A + s * 16 and y + s * 4
    for (int s = 0; s < count; ++s) {
        int piv[4];
        if (!Solver::solve_inplace(4, A + s * 16, 1, piv, 1, y + s * 4, 1)) return 1;
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= snippets::check(y + s * N, expected, N);
    return status;
}
