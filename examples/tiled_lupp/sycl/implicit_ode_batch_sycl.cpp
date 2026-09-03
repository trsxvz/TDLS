/// \file
/// \brief Example: reaction substep of an operator-splitting scheme on
/// a SYCL device, one independent stiff implicit step chain per cell,
/// the linear systems living in the private memory of each work-item.
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
/// Why SYCL: the kernel is a single-source parallel_for, one work-item
/// per cell, and the solver headers compile as plain C++ inside it: no
/// decoration is needed in this model. The batch runs on whatever
/// device the default selector picks: an Intel GPU under oneAPI, any
/// other SYCL backend, or a CPU device, which is why the default batch
/// stays moderate. Every solver operand is a local array of the
/// work-item, so the systems live in private memory; device USM holds
/// the per-cell inputs and outputs, structure-of-arrays. A machine
/// without a SYCL device reports the test skipped.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>

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

    // A machine without a SYCL runtime exposes no device: the test
    // reports itself skipped, as the CUDA and HIP examples do. The
    // queue is in order, so the kernel sees the input copy and the
    // readbacks see the kernel.
    std::optional<sycl::queue> queue;
    try {
        queue.emplace(sycl::property_list{sycl::property::queue::in_order{}});
    } catch (const sycl::exception&) {
        std::printf("no device available, skipping\n");
        return 77;
    }
    sycl::queue& q = *queue;

    // Per-cell temperature factors in [0.5, 2], log-uniform across the
    // batch, synthesized on the host as the input data of the substep.
    std::vector<double> theta(cells);
    for (int c = 0; c < cells; ++c)
        theta[c] = 0.5 * std::pow(4.0, hash01(static_cast<unsigned long long>(c)));

    double* d_theta = sycl::malloc_device<double>(cells, q);
    double* d_state = sycl::malloc_device<double>(static_cast<std::size_t>(species) * cells, q);
    int* d_ok       = sycl::malloc_device<int>(cells, q);
    q.memcpy(d_theta, theta.data(), sizeof(double) * theta.size());

    // One cell per work-item: reads the cell inputs, integrates the
    // chemistry entirely in private memory, writes the final state
    // (SoA layout, coalesced) and a success flag.
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(cells)), [=](sycl::id<1> idx) {
        const int c = static_cast<int>(idx[0]);

        double butcher[stages][stages];
        radau_butcher(butcher);

        // Per-cell inputs: fresh mixture and the cell temperature factor.
        double y[species] = {1.0, 0.0, 0.0};

        // Newton systems in private memory: every operand is local.
        double M[N * N], r[N], dz[N];
        int piv[N];
        bool success = true;
        for (int s = 0; s < steps && success; ++s)
            success =
                radau_step<true, true, true>(butcher, d_theta[c], h, y, M, 1, piv, 1, r, dz, 1);

        for (int a = 0; a < species; ++a)
            d_state[static_cast<std::size_t>(a) * cells + c] = y[a];
        d_ok[c] = success ? 1 : 0;
    });

    std::vector<double> state(static_cast<std::size_t>(species) * cells);
    std::vector<int> ok(cells);
    q.memcpy(state.data(), d_state, sizeof(double) * state.size());
    q.memcpy(ok.data(), d_ok, sizeof(int) * ok.size());
    q.wait();
    sycl::free(d_theta, q);
    sycl::free(d_state, q);
    sycl::free(d_ok, q);

    // The Robertson system conserves the total mass exactly, and the
    // trajectory stays a physical concentration vector.
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
