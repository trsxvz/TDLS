/// \file
/// \brief Example: reaction substep of an operator-splitting scheme, one
/// independent stiff implicit step chain per cell, parallelized with
/// OpenMP on the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Why the dimension is known at compile time: exactly as in the
/// sequential implicit_ode example, the linear systems are the Newton
/// systems of a Radau IIA step, of size N = stages x species with both
/// factors fixed by the method and by the chemical mechanism. N = 9 is a
/// property of the program, not of the data, so TiledLUppSolverStatic
/// applies.
///
/// Why a batch appears: reactive-transport codes split every global time
/// step into a transport substep and a reaction substep. During the
/// reaction substep each cell integrates its own chemistry independently
/// of every other cell: a moderate number of small stiff systems of
/// identical size, embarrassingly parallel. One omp parallel for
/// distributes the cells; each thread solves on its own stack arrays
/// (local residency), and only the per-cell outputs (final state,
/// success flag) live in shared memory. The loop only computes and the
/// self-checks run serially on the gathered outputs afterwards, exactly
/// as in the GPU examples; the pragma thus stays plain OpenMP 2.0 (no
/// min/max reduction, which the default MSVC OpenMP runtime rejects).
///
/// Each cell carries its own temperature factor scaling the reaction
/// rates, so the Newton matrices differ from cell to cell and are
/// generated on the fly from the cell inputs.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

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

    double butcher[stages][stages];
    radau_butcher(butcher);

    const double h  = 1e-3; // time step of the reaction substep
    const int steps = 10;   // integrate each cell to t = 0.01

    // Shared per-cell outputs, written once per cell by its owning
    // iteration: the final state and a success flag, the exact outputs
    // of the GPU companion example.
    std::vector<double> state(static_cast<std::size_t>(species) * cells);
    std::vector<int> ok(cells);

#pragma omp parallel for schedule(static)
    for (int c = 0; c < cells; ++c) {
        // Per-cell inputs: fresh mixture and a temperature factor in
        // [0.5, 2], log-uniform across the batch.
        double y[species]  = {1.0, 0.0, 0.0};
        const double theta = 0.5 * std::pow(4.0, hash01(static_cast<unsigned long long>(c)));

        // Newton systems on the stack of the thread.
        double M[N * N], r[N], dz[N];
        int piv[N];
        bool success = true;
        for (int s = 0; s < steps && success; ++s)
            success = radau_step<true, true, true>(butcher, theta, h, y, M, 1, piv, 1, r, dz, 1);

        for (int a = 0; a < species; ++a)
            state[static_cast<std::size_t>(species) * c + a] = y[a];
        ok[c] = success ? 1 : 0;
    }

    // Serial self-checks over the gathered outputs: the Robertson
    // system conserves the total mass exactly, and the trajectory stays
    // a physical concentration vector.
    int failures = 0, unphysical = 0;
    double mass_err = 0.0;
    for (int c = 0; c < cells; ++c) {
        if (!ok[c]) ++failures;
        const double* y = &state[static_cast<std::size_t>(species) * c];
        mass_err        = std::fmax(mass_err, std::fabs(y[0] + y[1] + y[2] - 1.0));
        if (!(y[0] > 0.0 && y[0] < 1.0 && y[1] > 0.0 && y[1] < 1e-3 && y[2] > 0.0 && y[2] < 1.0))
            ++unphysical;
    }

    std::printf("cells = %d, threads = %d, max mass error = %.3e\n", cells, omp_get_max_threads(),
                mass_err);
    return failures == 0 && unphysical == 0 && mass_err < 1e-10 ? 0 : 1;
}
