/// \file
/// \brief Negative compilation test: an rhs_pass_width above border + 1 must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the contract diagnostic of
/// the ArrowheadLUpp solver (a pass cannot hold more right-hand sides than the border + 1 columns of a block).

#include <tdls/tdls.hpp>

namespace {

constexpr auto shape  = tdls::ArrowheadShape{6, 2, 3, 0};
constexpr auto config = tdls::ArrowheadLUppConfig<double>{.rhs_pass_width = 8};

} // namespace

int main() {
    double M[1024] = {};
    double w[1024] = {};
    double y[64]   = {};
    using Solver   = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    return Solver::solve_inplace<true, true, true>(M, 1, w, 1, y, 1) ? 0 : 1;
}
