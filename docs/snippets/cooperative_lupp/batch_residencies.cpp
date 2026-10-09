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
#include "group.hpp"

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
    int ok[2] = {}, ok_batch[2] = {};
    double y_local[2][N] = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];

    snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
        // internal: every operand is the slice of the thread, its rows tx and tx + 2,
        // in local arrays: the matrix rows, the right-hand side entries and the pivots
        double A_slice[2 * N], y_slice[2];
        int piv_slice[2];
        for (int K = 0; K < 2; ++K) {
            for (int c = 0; c < N; ++c)
                A_slice[K * N + c] = matrix(0, (tx + 2 * K) * N + c);
            y_slice[K] = rhs(0, tx + 2 * K);
        }
        ok[tx] = Solver::solve_inplace<true, true, true>(tx, A_slice, 1, piv_slice, 1, y_slice, 1,
                                                         work, sync);
        for (int K = 0; K < 2; ++K)
            y_local[tx][K] = y_slice[K]; // entry tx + 2 K of the solution

        // external: every operand is system 1 of the SoA batch, reachable by the whole
        // group and walked with the batch stride 3
        ok_batch[tx] = Solver::solve_inplace<false, false, false>(tx, A_soa + 1, 3, piv_soa + 1, 3,
                                                                  y_soa + 1, 3, work, sync);
    });
    // snippet end

    if (!ok[0] || !ok[1] || !ok_batch[0] || !ok_batch[1]) return 1;
    const double x0[N] = {y_local[0][0], y_local[1][0], y_local[0][1], y_local[1][1]};
    return snippets::check(x0, expected, N) | snippets::check(y_soa + 1, 3, expected, N);
}
