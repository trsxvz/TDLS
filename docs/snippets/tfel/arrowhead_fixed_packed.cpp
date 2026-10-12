/// \file
/// \brief Documentation snippet: a TFEL jacobian written in the Packed storage of an arrowhead system.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    const double expected[6] = {1, 2, 3, 4, 5, 6};

    // snippet begin
    tfel::math::tmatrix<6, 6, double> J{6, 1, 1, 0, 2, 0, //
                                        1, 7, 0, 1, 0, 1, //
                                        1, 0, 5, 1, 0, 0, //
                                        0, 2, 1, 4, 0, 0, //
                                        2, 1, 0, 0, 6, 2, //
                                        0, 1, 0, 0, 1, 5};
    tfel::math::tvector<6, double> y{21, 25, 20, 23, 46, 37};

    constexpr auto shape  = tdls::ArrowheadShape{.border = 2, .block = 2, .blocks = 2};
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Map    = Solver::storage_map;

    // the entries of the pattern in a tvector, each at the index of the map; a
    // behaviour would write its derivatives there directly
    tfel::math::tvector<Solver::storage_size, double> M;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            M[Map::abb(shape, r, c)] = J(r, c);
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < 2; ++i) {
            const int row                 = 2 + 2 * k + i;
            M[Map::abk_diag(shape, k, i)] = J(i, row);
            for (int c = 0; c < 2; ++c)
                M[Map::akb(shape, k, i, c)] = J(row, c);
            for (int j = 0; j < 2; ++j)
                M[Map::d(shape, k, i, j)] = J(row, 2 + 2 * k + j);
        }
    double work[Solver::workspace_size];

    const bool ok = Solver::solve_inplace<true, true, true>(M.data(), 1, work, 1, y.data(), 1);
    // snippet end

    if (!ok) return 1;
    return snippets::check(y.data(), expected, 6);
}
