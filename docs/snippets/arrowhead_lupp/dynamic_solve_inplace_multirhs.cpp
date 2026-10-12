/// \file
/// \brief Documentation snippet: solve a block in place, the shape and the
/// number of right-hand sides at run time.
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
    // two right-hand sides, column m from m * n
    std::vector<double> y(2 * n);
    for (int i = 0; i < n; ++i) {
        y[i]     = b[i];
        y[n + i] = b_second[i];
    }

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;

    const tdls::ArrowheadShape runtime_shape{.border = 2, .block = 2, .blocks = 2};
    const int nrhs = 2;

    std::vector<double> M(Solver::storage_map::size(runtime_shape));
    store<Solver::storage_map>(A, M.data());
    // the workspace grows with the number of right-hand sides
    std::vector<double> work(Solver::workspace_size(runtime_shape, nrhs));
    std::vector<int> piv(Solver::pivot_size(runtime_shape));

    // column m of y from m * n: element stride 1, column stride n
    const bool ok = Solver::solve_inplace_multirhs(runtime_shape, nrhs, M.data(), 1, piv.data(), 1,
                                                   work.data(), 1, y.data(), 1, n);
    // snippet end

    return ok ? snippets::check(y.data(), expected, n) +
                    snippets::check(y.data() + n, expected_second, n)
              : 1;
}
