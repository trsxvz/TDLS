/// \file
/// \brief Documentation snippet: an SoA batch.
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
    double A[N * N * count] = {}, y[N * count] = {};
    int piv[N * count] = {};
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A[k * count + s] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y[i * count + s] = rhs(s, i);
    }
    int ok[count][2] = {};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];

    // system s at A + s, every operand walked with the batch stride
    for (int s = 0; s < count; ++s) {
        snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
            ok[s][tx] = Solver::solve_inplace<false, false, false>(tx, A + s, count, piv + s, count,
                                                                   y + s, count, work, sync);
        });
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s][0] && ok[s][1] ? snippets::check(y + s, count, expected, N) : 1;
    return status;
}
