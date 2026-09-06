/// \file
/// \brief Documentation snippet: the residencies.
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
    // an SoA batch of the three systems: element k of system s at base[k * count + s]
    double A_soa[N * N * count], y_soa[N * count];
    int piv_soa[N * count];
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A_soa[k * count + s] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y_soa[i * count + s] = rhs(s, i);
    }
    // system 0 in local arrays, the right-hand side of system 2 in a local array
    double A_local[N * N], y_local[N], y_mixed[N];
    for (int k = 0; k < N * N; ++k)
        A_local[k] = matrix(0, k);
    for (int i = 0; i < N; ++i) {
        y_local[i] = rhs(0, i);
        y_mixed[i] = rhs(2, i);
    }

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    int piv[4];

    // internal: every operand is a caller-local array, the strides are ignored
    const bool ok_local = Solver::solve_inplace<true, true, true>(A_local, 1, piv, 1, y_local, 1);

    // external: every operand is system 1 of the SoA batch, walked with the batch stride 3
    const bool ok_batch =
        Solver::solve_inplace<false, false, false>(A_soa + 1, 3, piv_soa + 1, 3, y_soa + 1, 3);

    // mixed: the matrix of system 2 stays in the batch, its pivot and right-hand side are local
    const bool ok_mixed =
        Solver::solve_inplace<true, true, false>(A_soa + 2, 3, piv, 1, y_mixed, 1);
    // snippet end

    if (!ok_local || !ok_batch || !ok_mixed) return 1;
    return snippets::check(y_local, expected, N) | snippets::check(y_soa + 1, 3, expected, N) |
           snippets::check(y_mixed, expected, N);
}
