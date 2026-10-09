/// \file
/// \brief Documentation snippet: Newton iteration, then tangent columns.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <algorithm>
#include <cmath>

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

namespace {

// F(x) = A x + x * x - c, with x * x the entrywise square: row i
double residual(const double* A, const double* c, const double* x, const int i) {
    double r = x[i] * x[i] - c[i];
    for (int j = 0; j < 4; ++j)
        r += A[i * 4 + j] * x[j];
    return r;
}

} // namespace

int main() {
    const double A[4 * 4]    = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double c[4]        = {15, 18, 33, 49};
    const double expected[4] = {1, 2, 3, 4};
    const double e[2][4]     = {{1, 0, 0, 0}, {0, 1, 0, 0}};
    double x_seen[2][4]      = {};
    double dx_dc[2][4];
    int ok[2] = {1, 1};

    // snippet begin
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    // J(x) = A + 2 diag(x) and the residual, shared by the group
    double J[4 * 4], r[4], work[Solver::workspace_size];
    int piv[4];

    snippets::run_group<Solver::threads_per_system>([&](const int tx, auto&& sync) {
        // every thread iterates on its own copy of x, kept identical by the shared steps
        double x[4] = {0, 0, 0, 0};
        for (int iteration = 0; iteration < 20; ++iteration) {
            // each thread builds the rows it holds in the solver, rows tx and tx + 2: the
            // solver reads them from their owner, so no barrier is needed before the call
            for (int i = tx; i < 4; i += Solver::threads_per_system) {
                r[i] = residual(A, c, x, i);
                for (int j = 0; j < 4; ++j)
                    J[i * 4 + j] = A[i * 4 + j] + (i == j ? 2 * x[i] : 0);
            }
            // one solve in place per iteration; on return the whole step is visible to every
            // thread, so each one updates its x with the entries of the others
            if (!Solver::solve_inplace<false, false, false>(tx, J, 1, piv, 1, r, 1, work, sync))
                ok[tx] = 0;
            double step = 0;
            for (int i = 0; i < 4; ++i) {
                x[i] -= r[i];
                step = std::max(step, std::fabs(r[i]));
            }
            // the same test in every thread: the group leaves the loop together
            if (step < 1e-14) break;
            sync(); // no thread rebuilds its rows of J and r before the others have read r
        }

        // tangent columns: one factorization at the solution, two canonical columns
        for (int i = tx; i < 4; i += Solver::threads_per_system)
            for (int j = 0; j < 4; ++j)
                J[i * 4 + j] = A[i * 4 + j] + (i == j ? 2 * x[i] : 0);
        if (!Solver::factorize<false, false>(tx, J, 1, piv, 1, work, sync)) ok[tx] = 0;
        Solver::substitute_canonical<false, false, false>(tx, J, 1, piv, 1, 0, dx_dc[0], 1, work,
                                                          sync);
        Solver::substitute_canonical<false, false, false>(tx, J, 1, piv, 1, 1, dx_dc[1], 1, work,
                                                          sync);
        for (int i = 0; i < 4; ++i)
            x_seen[tx][i] = x[i];
    });
    // snippet end

    if (!ok[0] || !ok[1]) return 1;
    double J0[4 * 4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            J0[i * 4 + j] = A[i * 4 + j] + (i == j ? 2 * x_seen[0][i] : 0);
    return snippets::check(x_seen[0], expected, 4) | snippets::check(x_seen[1], expected, 4) |
           snippets::check_product(J0, dx_dc[0], e[0], 4) |
           snippets::check_product(J0, dx_dc[1], e[1], 4);
}
