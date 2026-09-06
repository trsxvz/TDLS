/// \file
/// \brief Documentation snippet: a strided view over an SoA batch.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/Array/StridedCoalescedView.hxx>
#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "batch.hpp"
#include "check.hpp"

int main() {
    using namespace snippets;
    double soa_A[N * N * count], soa_y[N * count];
    for (int s = 0; s < count; ++s) {
        for (int k = 0; k < N * N; ++k)
            soa_A[k * count + s] = matrix(s, k);
        for (int i = 0; i < N; ++i)
            soa_y[i * count + s] = rhs(s, i);
    }

    // snippet begin
    // system 1 of the batch, stride 3: a strided view resolves the external residency
    auto A = tfel::math::map_strided<tfel::math::tmatrix<4, 4, double>>(soa_A + 1, 3);
    auto y = tfel::math::map_strided<tfel::math::tvector<4, double>>(soa_y + 1, 3);

    tfel::math::fsarray<4, int> piv;

    const bool ok = tdls::solve_inplace(A, piv, y);
    // snippet end

    return ok ? snippets::check(soa_y + 1, count, expected, N) : 1;
}
