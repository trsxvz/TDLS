/// \file
/// \brief Negative compilation test: solve_border_canonical_multirhs with
/// more unit columns than the border must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the contract diagnostic of
/// the ArrowheadLUpp solver (the unit columns are columns of the border).

#include <tdls/tdls.hpp>

namespace {

constexpr auto shape  = tdls::ArrowheadShape{6, 2, 3, 0};
constexpr auto config = tdls::ArrowheadLUppConfig<double>{};

} // namespace

int main() {
    double M[1024] = {};
    double w[1024] = {};
    double x[64]   = {};
    using Solver   = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    return Solver::solve_border_canonical_multirhs<7, true, true, true>(M, 1, w, 1, 0, x, 1, 6) ? 0
                                                                                                : 1;
}
