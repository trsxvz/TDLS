/// \file
/// \brief Documentation snippet: the verdict on a singular matrix, with
/// one thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

int main() {
    double A[4 * 4] = {4, 1, 0, 2, 1, 5, 0, 0, 0, 1, 0, 1, 2, 0, 0, 7};
    double y[4]     = {14, 14, 24, 33};

    // snippet begin
    // column 2 is zero: the matrix is singular
    // 4 rows per thread for a 4 x 4 system: one thread solves it
    constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

    double work[Solver::workspace_size];
    int piv[4];

    // false: singular; the content of y is then unspecified
    const bool ok = Solver::solve_inplace<true, true, true>(0, A, 1, piv, 1, y, 1, work);
    // snippet end

    return ok ? 1 : 0;
}
