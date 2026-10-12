/// \file
/// \brief Documentation snippet: a batch solved by OpenMP threads.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "arrow.hpp"
#include "check.hpp"

// snippet begin
constexpr auto config = tdls::ArrowheadLUppConfig<double>{
    .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
using Solver = tdls::ArrowheadLUppSolverStatic<double, snippets::arrow::shape, config>;

// one iteration per system of the SoA batch
void solve_batch(const int count, double* M, double* y, int* ok) {
#pragma omp parallel for
    for (int s = 0; s < count; ++s) {
        // matrix and right-hand side in the batch, workspace local to the iteration
        double work[Solver::workspace_size];
        ok[s] =
            Solver::solve_inplace<false, true, false>(M + s, count, work, 1, y + s, count) ? 1 : 0;
    }
}
// snippet end

int main() {
    using namespace snippets::arrow;
    constexpr int size     = Solver::storage_size;
    double M[size * count] = {}, y[n * count] = {};
    int ok[count];
    for (int s = 0; s < count; ++s) {
        double full[n * n], stored[size];
        for (int k = 0; k < n * n; ++k)
            full[k] = matrix(s, k);
        store<Solver::storage_map>(full, stored);
        for (int e = 0; e < size; ++e)
            M[e * count + s] = stored[e];
        for (int i = 0; i < n; ++i)
            y[i * count + s] = rhs(s, i);
    }

    solve_batch(count, M, y, ok);

    int status = 0;
    for (int s = 0; s < count; ++s)
        status |= ok[s] ? snippets::check(y + s, count, expected, n) : 1;
    return status;
}
