/// \file
/// \brief Documentation snippet: an OpenMP loop over a batch.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "batch.hpp"
#include "check.hpp"

// snippet begin
constexpr auto config = tdls::TiledLUppConfig<double>{.tile_size = 2};
using Solver          = tdls::TiledLUppSolverStatic<double, 4, config>;

// one iteration per system of the SoA batch
void solve_batch(const int count, double* A, double* y, int* ok) {
#pragma omp parallel for
    for (int s = 0; s < count; ++s) {
        // matrix and right-hand side in the batch, pivot local to the iteration
        int piv[4];
        ok[s] =
            Solver::solve_inplace<false, true, false>(A + s, count, piv, 1, y + s, count) ? 1 : 0;
    }
}
// snippet end

int main() {
    using namespace snippets;
    double A[N * N * count], y[N * count];
    int ok[count];
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            A[k * count + s] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            y[i * count + s] = rhs(s, i);
    }

    solve_batch(count, A, y, ok);

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s] ? snippets::check(y + s, count, expected, N) : 1;
    return status;
}
