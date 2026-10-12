/// \file
/// \brief Documentation snippet: a solve during constant evaluation.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "arrow.hpp"

// snippet begin
constexpr double solve_at_compile_time() {
    using namespace snippets::arrow;
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    double M[Solver::storage_size] = {};
    store<Solver::storage_map>(A_singular_block, M);
    double y[n] = {21, 25, 20, 23, 15, 13};
    double work[Solver::workspace_size];

    if (!Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1)) return 0;
    return y[5];
}

// the whole solve, second pass included, runs during constant evaluation
constexpr double x5 = solve_at_compile_time();
static_assert(x5 > 6 - 1e-12 && x5 < 6 + 1e-12);
// snippet end

int main() {
    return 0;
}
