/// \file
/// \brief Documentation snippet: solve in place with a runtime shape.
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
    std::vector<double> y(b, b + n);

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;

    const tdls::ArrowheadShape runtime_shape{.border = 2, .block = 2, .blocks = 2};

    // the sizes follow the shape: storage, workspace, pivots
    std::vector<double> M(Solver::storage_map::size(runtime_shape));
    store<Solver::storage_map>(A, M.data());
    std::vector<double> work(Solver::workspace_size(runtime_shape));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));

    const bool ok = Solver::solve_inplace(runtime_shape, M.data(), 1, piv.data(), 1, work.data(), 1,
                                          y.data(), 1);
    // snippet end

    return ok ? snippets::check(y.data(), expected, n) : 1;
}
