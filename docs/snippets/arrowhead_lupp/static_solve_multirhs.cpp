/// \file
/// \brief Documentation snippet: solve a block with two buffers, the
/// right-hand sides left untouched.
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
    // two right-hand sides, column m from m * n
    double y[2 * n] = {};
    for (int i = 0; i < n; ++i) {
        y[i]     = b[i];
        y[n + i] = b_second[i];
    }

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A, M);

    double work[Solver::multirhs_workspace_size<2>];
    // solve_multirhs copies the two columns of y into x, then solves in
    // place: y stays as it was
    double x[2 * n];
    const bool ok = Solver::solve_multirhs<2, true, true, true>(M, 1, work, 1, y, x, 1, n);
    // snippet end

    return ok ? snippets::check(x, expected, n) + snippets::check(x + n, expected_second, n) +
                    snippets::check(y + n, b_second, n)
              : 1;
}
