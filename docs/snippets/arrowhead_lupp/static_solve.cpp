/// \file
/// \brief Documentation snippet: solve with two buffers, b left untouched.
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

    // b is copied into x, then solved in place; b stays as it was
    double x[n];
    double work[Solver::workspace_size];
    const bool ok = Solver::solve<true, true, true>(M, 1, work, 1, b, x, 1);
    // snippet end

    return ok ? snippets::check(x, expected, n) : 1;
}
