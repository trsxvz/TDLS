/// \file
/// \brief Documentation snippet: the runtime and compile-time solvers agree bit for bit.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <cstring>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrow.hpp"

int main() {
    using namespace snippets::arrow;

    // snippet begin
    constexpr auto config = tdls::ArrowheadLUppConfig<double>{
        .coupling = tdls::ArrowheadCoupling::Diagonal, .storage = tdls::ArrowheadStorage::Packed};
    using Static  = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    using Dynamic = tdls::ArrowheadLUppSolverDynamic<double, config>;

    double Ms[Static::storage_size], Md[Static::storage_size];
    store<Static::storage_map>(A_singular_block, Ms);
    store<Static::storage_map>(A_singular_block, Md);
    double ys[n], yd[n];
    for (int i = 0; i < n; ++i)
        ys[i] = yd[i] = b_singular_block[i];

    double ws[Static::workspace_size];
    std::vector<double> wd(Dynamic::workspace_size(shape));
    std::vector<int> piv(Dynamic::pivot_size(shape));
    const bool ok_static = Static::solve_inplace<true, true, true>(Ms, 1, ws, 1, ys, 1);
    const bool ok_dynamic =
        Dynamic::solve_inplace(shape, Md, 1, piv.data(), 1, wd.data(), 1, yd, 1);

    // the same arithmetic: the same bits, second pass included
    const bool same = std::memcmp(ys, yd, sizeof(ys)) == 0;
    // snippet end

    return ok_static && ok_dynamic && same ? 0 : 1;
}
