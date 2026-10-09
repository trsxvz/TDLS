/// \file
/// \brief Documentation snippet: runtime-sized TFEL objects shared by the
/// group of threads of the runtime CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <vector>

#include <TFEL/Math/matrix.hxx>
#include <TFEL/Math/vector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"
#include "group.hpp"

int main() {
    const double expected[5] = {1, 2, 3, 4, 5};
    std::vector<int> ok(3, 0);

    // snippet begin
    tfel::math::matrix<double> A = {
        {5, 1, 0, 0, 1}, {1, 6, 1, 0, 0}, {0, 1, 7, 1, 0}, {0, 0, 1, 8, 1}, {1, 0, 0, 1, 9}};
    tfel::math::vector<double> y = {12, 16, 27, 40, 50};
    tfel::math::vector<int> piv(5);
    const int n = static_cast<int>(y.size());

    // threads of 2 rows: 3 threads for n = 5, sharing the objects as external operands
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
    using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;
    std::vector<double> work(Solver::workspace_size(n));

    snippets::run_group(Solver::threads_per_system(n), [&](const int tx, auto&& sync) {
        ok[tx] = Solver::solve_inplace(n, tx, A.data(), 1, piv.data(), 1, y.data(), 1, work.data(),
                                       sync);
    });
    // snippet end

    if (!ok[0] || !ok[1] || !ok[2]) return 1;
    return snippets::check(y.data(), expected, 5);
}
