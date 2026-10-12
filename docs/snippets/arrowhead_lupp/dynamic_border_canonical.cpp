/// \file
/// \brief Documentation snippet: one canonical column of the border, the
/// shape at run time.
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
    std::vector<double> work(Solver::workspace_size(runtime_shape, 1));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));

    // column 1 of the border block of A^-1: border rows only
    std::vector<double> x(runtime_shape.border);
    const bool ok = Solver::solve_border_canonical_multirhs(runtime_shape, 1, M.data(), 1,
                                                            piv.data(), 1, work.data(), 1, 1,
                                                            x.data(), 1, runtime_shape.border);
    // snippet end

    return ok ? snippets::check(x.data(), inverse_border + 2, 2) : 1;
}
