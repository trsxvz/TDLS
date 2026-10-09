/// \file
/// \brief Run-time contract test: the runtime solver called with
/// tdls::NoSync on a dimension that needs a group of several threads
/// must stop the program.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This program must NOT finish. With two rows per thread, a 5 x 5
/// system needs a group of three threads, which cannot run without a
/// barrier. The runtime solver learns it from n only, so it checks
/// tdls::NoSync on entry, and ctest passes only when the program stops
/// with its message.

#include <cstdlib>

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
using Solver          = tdls::CooperativeLUppSolverDynamic<double, config>;

} // namespace

int main() {
#if defined(_MSC_VER)
    // No dialog nor crash report: the test runs unattended.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    constexpr int n                        = 5;
    double A[n * n]                        = {};
    double b[n]                            = {};
    double work[Solver::workspace_size(n)] = {};
    int piv[n];
    return Solver::solve_inplace(n, 0, A, 1, piv, 1, b, 1, work, tdls::NoSync{}) ? 0 : 1;
}
