/// \file
/// \brief Documentation snippet: a batch in AoSoA: blocks of 2 interleaved systems.
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
    // blocks of 2 systems, padded to 4 systems
    double M[size * 4] = {}, y[n * 4] = {};
    for (int s = 0; s < count; ++s) {
        double full[n * n], stored[size];
        for (int k = 0; k < n * n; ++k)
            full[k] = matrix(s, k);
        store<Map>(full, stored);
        for (int e = 0; e < size; ++e)
            M[(s / 2) * size * 2 + e * 2 + s % 2] = stored[e];
        for (int i = 0; i < n; ++i)
            y[(s / 2) * n * 2 + i * 2 + s % 2] = rhs(s, i);
    }

    // snippet begin
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

    // blocks of 2 interleaved systems: the stride is 2, the pointer of system s at its block
    for (int s = 0; s < count; ++s) {
        double work[Solver::workspace_size];
        double* Ms = M + (s / 2) * Solver::storage_size * 2 + s % 2;
        double* ys = y + (s / 2) * n * 2 + s % 2;
        if (!Solver::solve_inplace<false, true, false>(Ms, 2, work, 1, ys, 2)) return 1;
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= snippets::check(y + (s / 2) * n * 2 + s % 2, 2, expected, n);
    return status;
}
