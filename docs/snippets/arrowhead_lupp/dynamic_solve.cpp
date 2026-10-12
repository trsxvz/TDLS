/// \file
/// \brief Documentation snippet: solve with two buffers and a runtime shape.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"

int main() {
    using namespace snippets::arrow;

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;

    const tdls::ArrowheadShape runtime_shape{.border = 2, .block = 2, .blocks = 2};
    std::vector<double> M(Solver::storage_map::size(runtime_shape));
    store<Solver::storage_map>(A, M.data());
    std::vector<double> work(Solver::workspace_size(runtime_shape)), x(n);
    std::vector<int> piv(Solver::pivot_size(runtime_shape));

    // b stays as it was, x receives the solution
    const bool ok =
        Solver::solve(runtime_shape, M.data(), 1, piv.data(), 1, work.data(), 1, b, x.data(), 1);
    // snippet end

    return ok ? snippets::check(x.data(), expected, n) : 1;
}
