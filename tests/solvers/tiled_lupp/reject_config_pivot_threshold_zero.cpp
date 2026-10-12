/// \file
/// \brief Negative compilation test: a configuration whose relative pivot
/// threshold is zero must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the range contract diagnostic
/// of the TiledLUpp solver (a zero threshold would keep every in-tile
/// pivot in place, however small against the other candidates).

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::TiledLUppConfig<double>{.relative_pivot_threshold = 0.0};

} // namespace

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9, config>;
    return Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
