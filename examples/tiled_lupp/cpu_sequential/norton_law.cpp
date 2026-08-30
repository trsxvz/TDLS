/// \file
/// \brief Example: Norton viscoplasticity integrated the MFront way at
/// one integration point, using the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: the unknowns of the
/// implicit scheme are the 6 components of the elastic strain increment
/// and the viscoplastic multiplier. Their number is fixed by the
/// modelling hypothesis (3D) and by the law, never by the data. N = 7
/// is a property of the program, so TiledLUppSolverStatic applies,
/// exactly as in the behaviours MFront generates.
///
/// The program integrates a loading history at one point, one time step
/// after the other. Inside a step the solver is called twice: one
/// solve_inplace per Newton iteration on a fresh jacobian, then one
/// factorize and one substitute_canonical_multirhs for the consistent
/// tangent operator.
///
/// Self-checks at every step: the stress must reproduce the radial
/// return, the closed-form solution of the isotropic case, and the
/// tangent operator must match central differences of the integration.

#include <cmath>
#include <cstdio>

#include "norton.hpp"

int main() {
    using namespace norton;
    const double dt = 1.0; // time step (s)
    const int steps = 10;  // integrate to t = 10 s

    double eel[stensor_size] = {}; // stress-free initial state
    double p                 = 0;
    double worst_sig = 0, worst_tangent = 0;
    bool monotonic = true;

    for (int step = 0; step < steps; ++step) {
        double deto[stensor_size];
        strain_increment(0, deto);

        // State at the beginning of the step, kept for the checks.
        double eel0[stensor_size];
        for (int k = 0; k < stensor_size; ++k)
            eel0[k] = eel[k];
        const double p0 = p;

        // Newton systems on the stack: every operand is caller-local.
        double sig[stensor_size], Dt[stensor_size * stensor_size];
        double J[N * N], r[N];
        int piv[N];
        if (!integrate<true, true, true>(deto, dt, eel, p, sig, Dt, J, 1, piv, 1, r, 1)) {
            std::printf("Newton did not converge at step %d\n", step);
            return 1;
        }

        // Radial return and central differences on the same step.
        double sig_ref[stensor_size];
        radial_return(eel0, deto, dt, sig_ref);
        double smax = 0, err = 0;
        for (int k = 0; k < stensor_size; ++k) {
            smax = std::fmax(smax, std::fabs(sig_ref[k]));
            err  = std::fmax(err, std::fabs(sig[k] - sig_ref[k]));
        }
        worst_sig     = std::fmax(worst_sig, err / smax);
        worst_tangent = std::fmax(worst_tangent, tangent_deviation(eel0, p0, deto, dt, Dt));
        if (p < p0) monotonic = false;

        std::printf("t = %4.1f s: seq = %7.3f MPa, p = %.3e\n", (step + 1) * dt,
                    von_mises(sig) / 1e6, p);
    }

    std::printf("stress deviation from the radial return = %.1e, tangent deviation = %.1e\n",
                worst_sig, worst_tangent);
    return worst_sig < 1e-9 && worst_tangent < 1e-5 && monotonic ? 0 : 1;
}
