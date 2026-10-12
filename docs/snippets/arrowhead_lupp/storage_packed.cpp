/// \file
/// \brief Documentation snippet: the Packed storage, written through the storage map.
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
    double y[n] = {};
    for (int i = 0; i < n; ++i)
        y[i] = b[i];

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Solver = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Map    = Solver::storage_map;
    static_assert(Solver::storage_size == 24); // 4 + 4 couplings + 8 + 8

    // every entry of the pattern at the index the map gives
    double M[Solver::storage_size];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            M[Map::abb(shape, r, c)] = A[r * n + c]; // the border block
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < 2; ++i) {
            const int row                 = 2 + k * 2 + i;
            M[Map::abk_diag(shape, k, i)] = A[i * n + row]; // border row i, unknown i of block k
            for (int c = 0; c < 2; ++c)
                M[Map::akb(shape, k, i, c)] = A[row * n + c]; // row i of block k, border column c
            for (int j = 0; j < 2; ++j)
                M[Map::d(shape, k, i, j)] = A[row * n + 2 + k * 2 + j]; // the block itself
        }

    double work[Solver::workspace_size];
    const bool ok = Solver::solve_inplace<true, true, true>(M, 1, work, 1, y, 1);
    // snippet end

    return ok ? snippets::check(y, expected, n) : 1;
}
