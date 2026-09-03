/// \file
/// \brief Negative compilation test: a configuration whose singularity
/// floor exceeds the out-of-tile threshold must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the threshold-ordering
/// contract diagnostic of the TiledLUpp solvers (the floor only guards
/// the out-of-tile recovery path, so it would be unreachable above the
/// threshold).

#include <tdls/tdls.hpp>

namespace {

constexpr auto config =
    tdls::TiledLUppConfig<double>{.oot_threshold = 1e-10, .singular_floor = 1e-4};

} // namespace

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9, config>;
    return Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
