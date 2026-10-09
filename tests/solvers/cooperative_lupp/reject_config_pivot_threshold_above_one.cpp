/// \file
/// \brief Negative compilation test: a configuration whose relative pivot
/// threshold exceeds 1 must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the range contract diagnostic
/// of the runtime CooperativeLUpp solver (no row can reach more than the
/// largest magnitude of its column: a threshold above 1 would only keep
/// the rows in place that tie with it).

#include <tdls/tdls.hpp>

namespace {

constexpr auto config =
    tdls::CooperativeLUppConfig<double>{.rows_per_thread = 9, .relative_pivot_threshold = 2.0};

} // namespace

int main() {
    double M[81];
    double y[9];
    double work[29];
    int piv[9];
    using Solver = tdls::CooperativeLUppSolverDynamic<double, config>;
    return Solver::solve_inplace(9, 0, M, 1, piv, 1, y, 1, work) ? 0 : 1;
}
