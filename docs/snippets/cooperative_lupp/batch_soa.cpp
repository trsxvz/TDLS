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
    bool ok[count] = {};

    // snippet begin
    // one thread per system: 4 rows per thread
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];

    // system s at A + s, every operand walked with the batch stride
    for (int s = 0; s < count; ++s)
        ok[s] = Solver::solve_inplace<false, false, false>(0, A + s, count, piv + s, count, y + s,
                                                           count, work);
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s] ? snippets::check(y + s, count, expected, N) : 1;
    return status;
}
