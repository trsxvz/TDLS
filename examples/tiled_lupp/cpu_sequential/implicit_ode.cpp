/// \file
/// \brief Example: stiff chemical kinetics integrated with an implicit
/// Runge-Kutta method, using the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: the linear systems solved
/// here are the Newton systems of a Radau IIA time step. Their size is
/// N = stages x species, where the number of stages is fixed by the
/// integration method (3 for Radau IIA of order 5) and the number of
/// species is fixed by the chemical mechanism (3 for the Robertson
/// problem). Both are properties of the MODEL and the METHOD, written in
/// the program itself: no input data can change them, so N = 9 is a
/// compile-time constant and TiledLUppSolverStatic applies.
///
/// The example also shows the canonical reason for the split
/// factorize/substitute interface: following the classical RADAU5
/// practice, the Newton matrix is built from the Jacobian FROZEN at the
/// beginning of the step, factorized once, and the factorization is
/// reused by every Newton iteration through substitute().
///
/// Every array lives on the stack of main() (local residency).

#include <cmath>
#include <cstdio>

#include "robertson.hpp"

int main() {
    using namespace robertson;
    double butcher[stages][stages];
    radau_butcher(butcher);

    double y[species] = {1.0, 0.0, 0.0}; // initial concentrations
    const double h    = 1e-3;            // time step
    const int steps   = 1000;            // integrate to t = 1

    // Newton systems on the stack: every operand is caller-local.
    double M[N * N], r[N], dz[N];
    int piv[N];
    for (int step = 0; step < steps; ++step) {
        if (!radau_step<true, true, true>(butcher, 1.0, h, y, M, 1, piv, 1, r, dz, 1)) {
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
