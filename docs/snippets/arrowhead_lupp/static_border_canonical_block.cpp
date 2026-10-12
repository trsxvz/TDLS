/// \file
/// \brief Documentation snippet: a block of canonical columns of the border,
/// the whole border block of the inverse.
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

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size];
    store<Solver::storage_map>(A, M);

    double work[Solver::multirhs_workspace_size<2>];
    // columns 0 and 1 of the border block of A^-1: border rows only,
    // column m from m * 2 (internal: the strides are ignored)
    double X[2 * 2];
    const bool ok =
        Solver::solve_border_canonical_multirhs<2, true, true, true>(M, 1, work, 1, 0, X, 1, 2);
    // snippet end

    return ok ? snippets::check(X, inverse_border, 4) : 1;
}
