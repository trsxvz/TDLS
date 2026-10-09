/// \file
/// \brief Negative compilation test: a configuration with no row per
/// thread must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the rows_per_thread contract
/// diagnostic of the CooperativeLUpp solver.

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 0};

} // namespace

int main() {
    double M[81];
    double y[9];
    double work[29];
    int piv[9];
    using Solver = tdls::CooperativeLUppSolverStatic<double, 9, config>;
    return Solver::solve_inplace<true, true, true>(0, M, 1, piv, 1, y, 1, work) ? 0 : 1;
}
