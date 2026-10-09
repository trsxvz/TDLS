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
#include "group.hpp"

int main() {
    using namespace snippets;
    double A[N * N * 4] = {}, y[N * 4] = {};
    int piv[N * 4] = {};
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A[(s / 4) * N * N * 4 + k * 4 + s % 4] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y[(s / 4) * N * 4 + i * 4 + s % 4] = rhs(s, i);
    }
    int ok[count][2] = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size] = {};

    // blocks of 4 interleaved systems: system s at (s / 4) * N * N * 4 + s % 4, stride 4
    for (int s = 0; s < count; ++s) {
        snippets::run_group<Solver::threads_per_system>([&](const int tx) {
            ok[s][tx] = Solver::solve_inplace<false, false, false>(
                tx, A + (s / 4) * N * N * 4 + s % 4, 4, piv + (s / 4) * N * 4 + s % 4, 4,
                y + (s / 4) * N * 4 + s % 4, 4, work);
        });
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |=
            ok[s][0] && ok[s][1] ? snippets::check(y + (s / 4) * N * 4 + s % 4, 4, expected, N) : 1;
    return status;
}
