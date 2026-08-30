/// \file
/// \brief Example: reaction substep of an operator-splitting scheme on
/// the parallel STL, one independent stiff implicit step chain per
/// cell, on the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: exactly as in the other
/// implicit_ode examples, the Newton systems of a Radau IIA step have
/// size N = stages x species, fixed by the method and by the chemical
/// mechanism. N = 9 is a property of the program.
///
/// Why the parallel STL: std::for_each with the par_unseq policy over
/// the cell indices expresses the batch without naming a backend. The
/// same source runs on the CPU cores (nvc++ -stdpar=multicore, or
/// libstdc++ on oneTBB) and on a GPU (nvc++ -stdpar=gpu), where the
/// lambda becomes a kernel and the vectors it reaches through captured
/// pointers live in managed memory. The solver needs no decoration in
/// this model: its headers compile as plain C++.
///
/// The linear systems are local arrays of the lambda, so each cell
/// solves in registers on a device; only the per-cell outputs live in
/// the vectors, structure-of-arrays.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

#include "hash01.hpp"
#include "robertson.hpp"

int main(int argc, char** argv) {
    using namespace robertson;
    // Moderate batch: one independent chemistry integration per cell.
    const int cells = argc > 1 ? std::atoi(argv[1]) : 65536;
    if (cells < 1) {
        std::printf("usage: %s [number of cells >= 1]\n", argv[0]);
        return 1;
    }
    const double h  = 1e-3; // time step of the reaction substep
    const int steps = 10;   // integrate each cell to t = 0.01

    // Per-cell outputs, species-major SoA of extent cells, reached from
    // the lambda through pointers captured by value.
    std::vector<double> state(static_cast<std::size_t>(species) * cells);
    std::vector<int> ok(cells);
    double* state_p = state.data();
    int* ok_p       = ok.data();

    // The cell indices, materialized: a random-access range that every
    // parallel STL implementation distributes.
    std::vector<int> ids(cells);
    std::iota(ids.begin(), ids.end(), 0);
    std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](const int c) {
        double butcher[stages][stages];
        radau_butcher(butcher);

        // Per-cell inputs: fresh mixture and a temperature factor in
        // [0.5, 2], log-uniform across the batch.
        double y[species]  = {1.0, 0.0, 0.0};
        const double theta = 0.5 * std::pow(4.0, hash01(static_cast<unsigned long long>(c)));

        // Newton systems local to the iteration: every operand is
        // caller-local.
        double M[N * N], r[N], dz[N];
        int piv[N];
        bool success = true;
        for (int s = 0; s < steps && success; ++s)
            success = radau_step<true, true, true>(butcher, theta, h, y, M, 1, piv, 1, r, dz, 1);

        for (int a = 0; a < species; ++a)
            state_p[static_cast<std::size_t>(a) * cells + c] = y[a];
        ok_p[c] = success ? 1 : 0;
    });

    // Serial self-checks over the gathered outputs: the Robertson
    // system conserves the total mass exactly, and the trajectory stays
    // a physical concentration vector.
    int failures = 0, unphysical = 0;
    double mass_err = 0.0;
    for (int c = 0; c < cells; ++c) {
        if (!ok[c]) ++failures;
        const double y0 = state[c];
        const double y1 = state[static_cast<std::size_t>(cells) + c];
        const double y2 = state[static_cast<std::size_t>(2) * cells + c];
        mass_err        = std::fmax(mass_err, std::fabs(y0 + y1 + y2 - 1.0));
        if (!(y0 > 0.0 && y0 < 1.0 && y1 > 0.0 && y1 < 1e-3 && y2 > 0.0 && y2 < 1.0)) ++unphysical;
    }

    std::printf("cells = %d, max mass error = %.3e\n", cells, mass_err);
    return failures == 0 && unphysical == 0 && mass_err < 1e-10 ? 0 : 1;
}
