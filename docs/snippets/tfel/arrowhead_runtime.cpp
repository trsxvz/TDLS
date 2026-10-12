/// \file
/// \brief Documentation snippet: runtime-sized TFEL objects with the runtime arrowhead solver.
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

int main() {
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    tfel::math::matrix<double> J = {{6, 1, 1, 0, 2, 0}, //
                                    {1, 7, 0, 1, 0, 1}, //
                                    {1, 0, 5, 1, 0, 0}, //
                                    {0, 2, 1, 4, 0, 0}, //
                                    {2, 1, 0, 0, 6, 2}, //
                                    {0, 1, 0, 0, 1, 5}};
    tfel::math::vector<double> y = {21, 25, 20, 23, 46, 37};

    // the shape at run time, the Full storage of the row-major matrix
    const tdls::ArrowheadShape shape{.border = 2, .block = 2, .blocks = 2};
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Full};
    using Solver = tdls::ArrowheadLUppSolverDynamic<double, config>;
    std::vector<double> work(Solver::workspace_size(shape));
    std::vector<int> piv(Solver::pivot_size(shape));

    const bool ok =
        Solver::solve_inplace(shape, J.data(), 1, piv.data(), 1, work.data(), 1, y.data(), 1);
    // snippet end

    if (!ok) return 1;
    return snippets::check(y.data(), expected, 6);
}
