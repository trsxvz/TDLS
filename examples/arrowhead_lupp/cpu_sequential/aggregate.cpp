/// \file
/// \brief Example: implicit integration of a Berveiller-Zaoui aggregate
/// whose number of grains is given at run time, at one integration
/// point, using the runtime ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The physics of the polycrystal examples (polycrystal.hpp), with a
/// number of grains read on the command line: 240 by default, 1452
/// unknowns. A representative aggregate has hundreds of grains, and its
/// size is a datum of the computation, not of the behaviour. So the
/// shape is known at run time only, and ArrowheadLUppSolverDynamic
/// takes it as its first argument. Its temporaries live in a workspace
/// of workspace_size(shape) entries, the pivots of its TiledLUpp solves
/// in an int array of pivot_size(shape) entries: both are vectors here,
/// as the per-point state and the jacobian.
///
/// The program integrates a loading history at one point and times it.
/// Then it integrates the first step again with the whole jacobian in a
/// full matrix, solved by a dense LU, and times that too: the dense
/// solve grows with the cube of the number of unknowns, the arrowhead
/// one linearly. Both first steps must agree, and at every step the
/// viscoplastic strain must stay deviatoric and every grain must
/// dissipate.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "polycrystal.hpp"

int main(int argc, char** argv) {
    using namespace polycrystal;
    const int G = argc > 1 ? std::atoi(argv[1]) : 240;
    if (G < 1) {
        std::printf("usage: %s [number of grains >= 1]\n", argv[0]);
        return 1;
    }
    const tdls::ArrowheadShape shape = shape_of(G);
    const int n                      = unknowns(G);
    const double dt                  = 1.0; // time step (s)
    const int steps                  = 20;  // integrate to t = 20 s

    double deto[stensor_size];
    strain_increment(0, deto);

    // Every operand is a vector sized from the shape.
    std::vector<double> state(static_cast<std::size_t>(n)), du(state.size()), r(state.size()),
        A(static_cast<std::size_t>(Map::size(shape))),
        work(static_cast<std::size_t>(DynamicSolver::workspace_size(shape)));
    std::vector<int> piv(static_cast<std::size_t>(DynamicSolver::pivot_size(shape)));
    std::vector<double> before(state.size()), first(state.size());
    int iterations = 0, refused = 0;
    double worst_trace = 0, worst_dissipation = HUGE_VAL;

    const auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        before = state;
        // Grains = 0: the runtime solver. The operands are external, with
        // a stride of 1.
        if (!integrate<0, false, false, false, false>(0, G, deto, dt, state.data(), du.data(), 1,
                                                      A.data(), 1, piv.data(), 1, work.data(), 1,
                                                      r.data(), 1, iterations, refused)) {
            std::printf("Newton did not converge at step %d\n", step);
            return 1;
        }
        if (step == 0) first = state;
        const StepCheck check = check_step(G, before.data(), state.data(), 1);
        worst_trace           = std::fmax(worst_trace, check.trace);
        worst_dissipation     = std::fmin(worst_dissipation, check.dissipation);
        if (step % 4 == 3)
            std::printf("t = %4.1f s: Sig_eq = %6.1f MPa, Evp_xx = %.3e\n", (step + 1) * dt,
                        check.seq / 1e6, state[6]);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // The first step again, by the dense LU.
    std::vector<double> dense(state.size());
    if (!integrate_dense(0, G, deto, dt, dense.data())) {
        std::printf("dense Newton did not converge\n");
        return 1;
    }
    const auto t2            = std::chrono::steady_clock::now();
    const double worst_dense = deviation(first.data(), 1, dense.data(), n);

    std::printf("grains = %d, unknowns = %d, Newton iterations = %d, refused blocks = %d\n", G, n,
                iterations, refused);
    std::printf("%d steps by ArrowheadLUpp: %.3f s; the first step by a dense LU: %.3f s\n", steps,
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
    std::printf("deviation from the dense LU = %.1e, |tr Evp| / |Evp| = %.1e, "
                "smallest grain dissipation = %.1e Pa\n",
                worst_dense, worst_trace, worst_dissipation);
    return worst_dense < 1e-10 && worst_trace < 1e-12 && worst_dissipation > -1e-5 ? 0 : 1;
}
