/// \file
/// \brief Negative compilation test: a group of several threads called
/// with tdls::NoSync must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the barrier contract
/// diagnostic of the CooperativeLUpp solver: with three rows per thread,
/// a 9 x 9 system is solved by three threads, which cannot run without a
/// barrier. The default tdls::AutoSync deduces one; tdls::NoSync, passed
/// explicitly, is refused.

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3};

} // namespace

int main() {
    double M[27];
    double y[3];
    double work[29];
    int piv[3];
    using Solver = tdls::CooperativeLUppSolverStatic<double, 9, config>;
    return Solver::solve_inplace<true, true, true>(0, M, 1, piv, 1, y, 1, work, tdls::NoSync{}) ? 0
                                                                                                : 1;
}
