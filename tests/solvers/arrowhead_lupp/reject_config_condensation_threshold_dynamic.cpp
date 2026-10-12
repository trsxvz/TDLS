/// \file
/// \brief Negative compilation test: a negative condensation_pivot_threshold must be rejected by the runtime solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the contract diagnostic of
/// the ArrowheadLUpp solver (the threshold is a fraction of the largest candidate, in [0, 1]).

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::ArrowheadLUppConfig<double>{.condensation_pivot_threshold = -0.5};

} // namespace

int main() {
    double M[1024] = {};
    double w[1024] = {};
    double y[64]   = {};
    int piv[16]    = {};
    using Solver   = tdls::ArrowheadLUppSolverDynamic<double, config>;
    const tdls::ArrowheadShape shape{6, 2, 3, 0};
    return Solver::solve_inplace(shape, M, 1, piv, 1, w, 1, y, 1) ? 0 : 1;
}
