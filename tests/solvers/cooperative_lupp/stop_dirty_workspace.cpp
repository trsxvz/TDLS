/// \file
/// \brief Run-time contract test: a group of CPU threads whose workspace
/// was not zero before its first use must stop the program.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This program must NOT finish. With two rows per thread, a 4 x 4
/// system is solved by a group of two threads, whose deduced barrier
/// lives in the first two elements of the workspace. Their value here
/// is no state of the barrier, so the first thread to join stops the
/// program with its message, before any wait, and ctest passes only then.

#include <cstdlib>

#include <tdls/tdls.hpp>

namespace {

constexpr auto config = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 2};
using Solver          = tdls::CooperativeLUppSolverStatic<double, 4, config>;

} // namespace

int main() {
#if defined(_MSC_VER)
    // No dialog nor crash report: the test runs unattended.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    double A[16]                        = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    double b[4]                         = {14, 14, 24, 33};
    double work[Solver::workspace_size] = {};
    work[0]                             = 7; // a leftover, no state of the barrier
    int piv[4];
    return Solver::solve_inplace<false, false, false>(0, A, 1, piv, 1, b, 1, work) ? 0 : 1;
}
