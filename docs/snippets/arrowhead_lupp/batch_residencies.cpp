/// \file
/// \brief Documentation snippet: external and internal operands mixed in one call.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"

int main() {
    using namespace snippets::arrow;
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Map          = tdls::ArrowheadStorageMap<config.storage, config.coupling, config.layout>;
    constexpr int size = Map::size(shape);
    // the SoA batch: entry e of system s at M + e * count + s
    double M[size * count] = {}, y[n * count] = {};
    for (int s = 0; s < count; ++s) {
        double full[n * n], stored[size];
        for (int k = 0; k < n * n; ++k)
            full[k] = matrix(s, k);
        store<Map>(full, stored);
        for (int e = 0; e < size; ++e)
            M[e * count + s] = stored[e];
        for (int i = 0; i < n; ++i)
            y[i * count + s] = rhs(s, i);
    }

    // snippet begin
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    for (int s = 0; s < count; ++s) {
        // the right-hand side copied into a local array: internal, stride ignored
        double x[n];
        for (int i = 0; i < n; ++i)
            x[i] = y[i * count + s];
        double work[Solver::workspace_size];
        // <true, true, false>: x and work local, M in the batch
        if (!Solver::solve_inplace<true, true, false>(M + s, count, work, 1, x, 1)) return 1;
        for (int i = 0; i < n; ++i)
            y[i * count + s] = x[i];
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= snippets::check(y + s, count, expected, n);
    return status;
}
