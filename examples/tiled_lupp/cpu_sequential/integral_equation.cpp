/// \file
/// \brief Example: Love's integral equation solved by the Nystroem
/// method, using the runtime TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Love's equation gives the potential of a circular parallel-plate
/// capacitor with plate separation d:
///
///   u(x) + (1/pi) integral of d / (d^2 + (x-y)^2) u(y) dy = g(x)
///
/// The Nystroem method replaces the integral by a quadrature over n
/// points, which couples every point to every other: the linear system
/// is dense by nature, exactly what a dense direct solver is for.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched (here the command line). The same binary must serve
/// n = 64 and n = 200 without rebuilding, so the dimension is a runtime
/// value and TiledLUppSolverDynamic applies.
///
/// The factorization is computed once and reused by two right-hand
/// sides: a manufactured one (the discrete operator applied to a known
/// solution, which the solve must return to solver accuracy, whatever
/// the resolution) and the physical unit potential.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "love.hpp"

int main(int argc, char** argv) {
    using namespace love;
    // The quadrature resolution comes from outside the program; the
    // default is odd so that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 121;
    if (n < 5 || n > 1000) {
        std::printf("usage: %s [quadrature points in [5, 1000]]\n", argv[0]);
        return 1;
    }
    const double d = 1.0; // plate separation

    std::vector<double> A(static_cast<std::size_t>(n) * n), g(n), u(n);
    std::vector<int> piv(n);
    assemble(n, d, A.data(), 1, g.data(), 1);

    // One factorization, two right-hand sides.
    if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) {
        std::printf("singular Nystroem matrix\n");
        return 1;
    }

    // Manufactured right-hand side: the solve must return the
    // manufactured solution to solver accuracy (the equation is well
    // conditioned), whatever the quadrature resolution.
    Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
    const double err = manufactured_error(n, u.data(), 1);

    // Physical right-hand side: unit potential on the plate.
    std::fill(g.begin(), g.end(), 1.0);
    Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
    const int mid      = (n - 1) / 2;
    const double u_mid = u[mid];

    std::printf("n = %d, manufactured error = %.3e, u(%.2f) = %.6f (d = %.1f)\n", n, err,
                node(mid, n), u_mid, d);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return err < 1e-12 && u_mid > 0.4 && u_mid < 1.0 ? 0 : 1;
}
