/// \file
/// \brief Documentation snippet: the threshold of the pivot check.
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
    // the base matrix with a pivot of 1 in the second block, under a coupling of 2
    double A1[n * n], y1[n], y2[n];
    for (int k = 0; k < n * n; ++k)
        A1[k] = A[k];
    A1[4 * n + 4] = 1;
    for (int r = 0; r < n; ++r) {
        y1[r] = 0;
        for (int c = 0; c < n; ++c)
            y1[r] += A1[r * n + c] * expected[c];
        y2[r] = y1[r];
    }

    // snippet begin
    // the default, 0.01: a block passes while no multiplier of its elimination exceeds 100
    constexpr auto lenient = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    // 1: every pivot must be the largest candidate of its column, border rows included
    constexpr auto strict =
        tdls::ArrowheadLUppConfig<double>{.coupling = tdls::ArrowheadCoupling::Diagonal,
                                          .storage  = tdls::ArrowheadStorage::Packed,
                                          .condensation_pivot_threshold = 1.0};
    using Lenient = tdls::ArrowheadLUppSolverStatic<double, shape, lenient>;
    using Strict  = tdls::ArrowheadLUppSolverStatic<double, shape, strict>;

    double M1[Lenient::storage_size], M2[Strict::storage_size];
    store<Lenient::storage_map>(A1, M1);
    store<Strict::storage_map>(A1, M2);
    double w1[Lenient::workspace_size], w2[Strict::workspace_size];
    int refused_lenient = 0, refused_strict = 0;
    const bool ok1 = Lenient::solve_inplace<true, true, true>(M1, 1, w1, 1, y1, 1, refused_lenient);
    const bool ok2 = Strict::solve_inplace<true, true, true>(M2, 1, w2, 1, y2, 1, refused_strict);
    // the pivot 1 under the coupling 2, a multiplier of 2: refused_lenient == 0,
    // refused_strict == 1; both solutions are right
    // snippet end

    if (!ok1 || !ok2 || refused_lenient != 0 || refused_strict != 1) return 1;
    return snippets::check(y1, expected, n) | snippets::check(y2, expected, n);
}
