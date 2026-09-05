/// \file
/// \brief Negative compilation test: a configuration whose singularity
/// floor is zero must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the positive-floor contract
/// diagnostic of the TiledLUpp solvers (no magnitude falls below zero,
/// so a zero floor would let a zero pivot through and return true with
/// infinite entries).

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::TiledLUppConfig<double>{.singular_floor = 0.0};

} // namespace

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9, config>;
    return Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
