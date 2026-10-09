/// \file
/// \brief Documentation snippet: the solver constants.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <limits>

#include <tdls/tdls.hpp>

int main() {
    // snippet begin
    // 13 rows on threads of 5 rows: 3 threads; slot 4 holds row 12 on thread 0 only
    using Solver = tdls::CooperativeLUppSolverStatic<
        double, 13, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 5}>;

    static_assert(Solver::rows_per_thread == 5);
    static_assert(Solver::threads_per_system == 3);
    static_assert(Solver::workspace_size == 3 * 13 + 2); // shared by the group, 2 for its barrier
    static_assert(Solver::relative_pivot_threshold == 1.0);
    static_assert(Solver::singular_floor == std::numeric_limits<double>::min());

    // slots 0 to 3 are full, slot 4 is mixed
    static_assert(Solver::slot_is_full(3) && !Solver::slot_is_full(4) && Solver::slot_has_rows(4));
    // snippet end
    return 0;
}
