/// \file
/// \brief Documentation snippet: a Newton iteration, then the tangent columns.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <algorithm>
#include <cmath>

#include <tdls/tdls.hpp>

#include "check.hpp"

namespace {

// F(x) = A x + x * x - c, with x * x the entrywise square
void residual(const double* A, const double* c, const double* x, double* r) {
    for (int i = 0; i < 4; ++i) {
        r[i] = x[i] * x[i] - c[i];
        for (int j = 0; j < 4; ++j)
            r[i] += A[i * 4 + j] * x[j];
    }
}

// J(x) = A + 2 diag(x)
void jacobian(const double* A, const double* x, double* J) {
    for (int i = 0; i < 4 * 4; ++i)
        J[i] = A[i];
    for (int i = 0; i < 4; ++i)
        J[i * 4 + i] += 2 * x[i];
}

} // namespace

int main() {
    const double A[4 * 4]    = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    const double c[4]        = {15, 18, 33, 49};
    const double expected[4] = {1, 2, 3, 4};
    const double e[2][4]     = {{1, 0, 0, 0}, {0, 1, 0, 0}};

    // snippet begin
    constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
    using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

    double x[4] = {0, 0, 0, 0};
    double J[4 * 4], r[4];
    int piv[4];

    for (int iteration = 0; iteration < 20; ++iteration) {
        residual(A, c, x, r);
        jacobian(A, x, J);

        // one solve in place per iteration, on a fresh jacobian: r becomes the Newton step
        if (!Solver::solve_inplace<true, true, true>(J, 1, piv, 1, r, 1)) return 1;

        double step = 0;
        for (int i = 0; i < 4; ++i) {
            x[i] -= r[i];
            step = std::max(step, std::fabs(r[i]));
        }
        if (step < 1e-14) break;
    }

    // tangent columns: one factorization at the solution, two canonical columns in one block
    jacobian(A, x, J);
    if (!Solver::factorize<true, true>(J, 1, piv, 1)) return 1;

    double dx_dc[2 * 4];
    Solver::substitute_canonical_multirhs<2, true, true, true>(J, 1, piv, 1, 0, dx_dc, 1, 0);
    // snippet end

    double J0[4 * 4];
    jacobian(A, x, J0);
    return snippets::check(x, expected, 4) | snippets::check_product(J0, dx_dc, e[0], 4) |
           snippets::check_product(J0, dx_dc + 4, e[1], 4);
}
