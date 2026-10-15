/// \file
/// \brief Negative compilation test: TiledLUpp must reject
/// RowInterchange::Auto.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the row interchange contract
/// diagnostic of the TiledLUpp solvers: Auto is the default of the blocks
/// of ArrowheadLUpp only.

#include <tdls/tdls.hpp>

namespace {

constexpr auto config =
    tdls::TiledLUppConfig<double>{.row_interchange = tdls::RowInterchange::Auto};

} // namespace

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9, config>;
    return Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
