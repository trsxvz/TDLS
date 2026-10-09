/// \file
/// \brief Example: stiff chemical kinetics integrated with an implicit
/// Runge-Kutta method, using the compile-time CooperativeLUpp solver on
/// one thread.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: the linear systems solved
/// here are the Newton systems of a Radau IIA time step, of size N =
/// stages x species = 9. Both factors are properties of the method and
/// of the chemical mechanism, written in the program itself, so
/// CooperativeLUppSolverStatic applies.
///
/// Why one thread: the program is sequential, so the group of the
/// solver is the thread alone. As many rows per thread as the dimension
/// make it hold the whole system, with no barrier and without forced
/// unrolling, the choice on CPU. The same scheme runs on groups of
/// threads in the GPU examples.
///
/// As in RADAU5, the Newton matrix is built from the Jacobian frozen at
/// the beginning of the step, factorized once, and reused by every
/// Newton iteration through substitute(). Every array lives on the
/// stack of main(): the internal residencies.

#include <cmath>
#include <cstdio>

#include "robertson.hpp"

int main() {
    using namespace robertson;
    using Solver = GroupSolver<N, false>; // one thread per system

    double butcher[stages][stages];
    radau_butcher(butcher);

    double y[species] = {1.0, 0.0, 0.0}; // initial concentrations
    const double h    = 1e-3;            // time step
    const int steps   = 1000;            // integrate to t = 1

    // Newton systems on the stack: the slices of the only thread are the
    // whole objects. Rank 0, no barrier, and a flag is its own reduction.
    double M[N * N], r[N], dz[N], work[Solver::workspace_size];
    int piv[N];
    tdls::NoSync sync;
    auto any = [](const bool flag) { return flag; };
    for (int step = 0; step < steps; ++step) {
        if (!radau_step_group<true, true, true, Solver>(0, sync, any, butcher, 1.0, h, y, M, 1, piv,
                                                        1, r, dz, 1, work)) {
            std::printf("Newton did not converge at step %d\n", step);
            return 1;
        }
    }

    // Self-checks: the Robertson system conserves the total mass exactly,
    // and the trajectory stays a physical concentration vector.
    const double mass = y[0] + y[1] + y[2];
    std::printf("t = 1: y = (%.6f, %.6e, %.6f), mass = %.15f\n", y[0], y[1], y[2], mass);
    const bool conserved = std::fabs(mass - 1.0) < 1e-10;
    const bool physical =
        y[0] > 0.0 && y[0] < 1.0 && y[1] > 0.0 && y[1] < 1e-3 && y[2] > 0.0 && y[2] < 1.0;
    return conserved && physical ? 0 : 1;
}
