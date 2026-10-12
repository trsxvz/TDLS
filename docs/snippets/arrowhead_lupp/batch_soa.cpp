/// \file
/// \brief Documentation snippet: a batch in SoA: element-major.
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
    // entry e of system s at M + e * count + s
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

    // entry e of system s at M + e * 3 + s: the stride is the batch size
    for (int s = 0; s < count; ++s) {
        double work[Solver::workspace_size];
        if (!Solver::solve_inplace<false, true, false>(M + s, 3, work, 1, y + s, 3)) return 1;
    }
    // snippet end

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= snippets::check(y + s, count, expected, n);
    return status;
}
