/// \file
/// \brief Example: Iwan multisurface plasticity of a soil under a
/// shear cycle at one integration point, using the compile-time
/// ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why an arrowhead solver: the unknowns of the implicit scheme are the
/// elastic strain increment (6, the border) and the plastic multiplier
/// of each of the 24 surfaces (a block of one unknown each). Two
/// surfaces interact only through the elastic strain, so the 30 x 30
/// jacobian is an arrowhead matrix with dense couplings (iwan.hpp).
///
/// Why the shape is known at compile time: the number of surfaces is a
/// constant of the behaviour, as in the Iwan behaviour of the MFront
/// gallery, and the elastic strain has 6 components in 3D.
///
/// The program drives a simple shear cycle: a monotonic loading up to
/// the last calibration point, an unloading to the opposite strain,
/// and a reloading that closes the loop. The steps stop at the corners
/// of the expected curve and halfway between them. Every step checks
/// the shear stress: the skeleton curve on the loading, Masing's rule
/// (the skeleton scaled by 2 from the reversal point) on the two
/// others. The outer surfaces have small hardening moduli: their blocks
/// fail the pivot check, and the second pass of the solver eliminates
/// them with the border rows.

#include <cmath>
#include <cstdio>

#include "iwan.hpp"

int main() {
    using namespace iwan;
    const Material mat = calibrate();
    double dir[stensor_size];
    direction(0, dir); // simple shear
    const double gmax = mat.gamma[surfaces], tmax = skeleton(mat, gmax);

    double eel[stensor_size] = {}, X[stensor_size * surfaces] = {}, sig[stensor_size];
    double A[Solver::storage_size], work[Solver::workspace_size], r[N];
    int iterations = 0, refused = 0, steps = 0;
    double gamma = 0, worst = 0;

    // The three branches, each through the corners of its expected
    // curve, and halfway between them.
    for (int branch = 0; branch < 3; ++branch) {
        const double from = branch == 0 ? 0 : branch == 1 ? gmax : -gmax;
        const double sign = branch == 1 ? -1 : 1, scale = branch == 0 ? 1 : 2;
        double previous = 0;
        for (int j = 0; j <= surfaces; ++j)
            for (int half = 0; half < 2; ++half) {
                const double g      = half == 0 ? 0.5 * (previous + mat.gamma[j]) : mat.gamma[j];
                const double target = from + sign * scale * g;
                double deto[stensor_size];
                for (int i = 0; i < stensor_size; ++i)
                    deto[i] = (target - gamma) / sqrt2 * dir[i];
                if (!integrate<true, true, true>(mat, deto, eel, X, sig, A, 1, work, 1, r, 1,
                                                 iterations, refused)) {
                    std::printf("Newton did not converge at gamma = %.3e\n", target);
                    return 1;
                }
                gamma = target;
                ++steps;
                if (half == 1) previous = mat.gamma[j];
                // Skeleton, then Masing's rule from the reversal point.
                const double expected = branch == 0   ? skeleton(mat, g)
                                        : branch == 1 ? tmax - 2 * skeleton(mat, g)
                                                      : -tmax + 2 * skeleton(mat, g);
                const double tau      = shear(sig, dir);
                worst                 = std::fmax(worst, std::fabs(tau - expected) / tmax);
                worst                 = std::fmax(worst, off_direction(sig, dir));
                if (half == 1 && j % 6 == 0)
                    std::printf("branch %d: gamma = %10.3e, tau = %8.3f kPa (expected %8.3f)\n",
                                branch, gamma, tau / 1e3, expected / 1e3);
            }
    }

    std::printf("steps = %d, Newton iterations = %d, refused blocks = %d\n", steps, iterations,
                refused);
    std::printf("deviation from the skeleton and Masing's rule = %.1e\n", worst);
    return worst < 1e-9 ? 0 : 1;
}
