/// \file
/// \brief Example: implicit integration of a Berveiller-Zaoui
/// polycrystal of 12 grains at one integration point, using the
/// compile-time ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why an arrowhead solver: the unknowns of the implicit scheme are the
/// macroscopic elastic and viscoplastic strains (12, the border) and the
/// viscoplastic strain of each grain (6 per grain, the blocks). Two
/// grains interact only through the macroscopic quantities, so the
/// jacobian of the 84 unknowns is an arrowhead matrix, and the cost of a
/// solve grows linearly with the number of grains (polycrystal.hpp).
///
/// Why the shape is known at compile time: the number of grains is a
/// property of the behaviour, as in the polycrystal behaviours MFront
/// generates. ArrowheadLUppSolverStatic then holds every temporary in
/// local arrays of compile-time size.
///
/// The program integrates a loading history at one point, one Newton
/// solve per iteration on a jacobian assembled straight into the Packed
/// storage. The same history is integrated again with the whole
/// jacobian in a full matrix, solved by a dense LU: both must agree. At
/// every step the viscoplastic strain must stay deviatoric and every
/// grain must dissipate.

#include <cmath>
#include <cstdio>

#include "polycrystal.hpp"

int main() {
    using namespace polycrystal;
    constexpr int G = fixed_grains;
    constexpr int n = unknowns(G);
    using Solver    = StaticSolver<G>;
    const double dt = 1.0; // time step (s)
    const int steps = 20;  // integrate to t = 20 s

    double deto[stensor_size];
    strain_increment(0, deto);

    // The state, the Newton system and the workspace on the stack: every
    // operand is local.
    double state[n] = {}, du[n], r[n], A[Solver::storage_size], work[Solver::workspace_size];
    double dense[n] = {}; // the state of the dense reference
    int iterations = 0, refused = 0;
    double worst_dense = 0, worst_trace = 0, worst_dissipation = HUGE_VAL;

    for (int step = 0; step < steps; ++step) {
        double before[n];
        for (int i = 0; i < n; ++i)
            before[i] = state[i];
        if (!integrate<G, true, true, true, true>(0, G, deto, dt, state, du, 1, A, 1, nullptr, 1,
                                                  work, 1, r, 1, iterations, refused) ||
            !integrate_dense(0, G, deto, dt, dense)) {
            std::printf("Newton did not converge at step %d\n", step);
            return 1;
        }
        const StepCheck check = check_step(G, before, state, 1);
        worst_dense           = std::fmax(worst_dense, deviation(state, 1, dense, n));
        worst_trace           = std::fmax(worst_trace, check.trace);
        worst_dissipation     = std::fmin(worst_dissipation, check.dissipation);
        if (step % 4 == 3)
            std::printf("t = %4.1f s: Sig_eq = %6.1f MPa, Evp_xx = %.3e\n", (step + 1) * dt,
                        check.seq / 1e6, state[6]);
    }

    std::printf("grains = %d, unknowns = %d, Newton iterations = %d, refused blocks = %d\n", G, n,
                iterations, refused);
    std::printf("deviation from the dense LU = %.1e, |tr Evp| / |Evp| = %.1e, "
                "smallest grain dissipation = %.1e Pa\n",
                worst_dense, worst_trace, worst_dissipation);
    return worst_dense < 1e-10 && worst_trace < 1e-12 && worst_dissipation > -1e-5 ? 0 : 1;
}
