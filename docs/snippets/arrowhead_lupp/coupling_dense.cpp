/// \file
/// \brief Documentation snippet: dense couplings: every border row touches every block unknown.
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
    double y[n] = {};
    for (int i = 0; i < n; ++i)
        y[i] = b_dense[i];

    // snippet begin
    // the default coupling: border x block entries per block
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Dense, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    static_assert(Solver::storage_size == 28); // 4 + 8 couplings + 8 + 8

    double M[Solver::storage_size];
    store<Solver::storage_map>(A_dense, M);

    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
