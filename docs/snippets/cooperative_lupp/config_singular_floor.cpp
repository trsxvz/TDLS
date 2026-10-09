/// \file
/// \brief Documentation snippet: the singularity floor.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    double A[4 * 4]          = {4, 1, 0, 0, 1, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1e-8};
    double A2[4 * 4]         = {4, 1, 0, 0, 1, 5, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1e-8};
    double y[4]              = {6, 11, 18, 0};
    double y2[4]             = {6, 11, 18, 0};
    const double expected[4] = {1, 2, 3, 0};

    // snippet begin
    // the default floor, numeric_limits<double>::min(), accepts the pivot 1e-8;
    // a floor of 1e-6 declares the matrix singular
    constexpr auto standard = tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4};
    constexpr auto raised =
        tdls::CooperativeLUppConfig<double>{.rows_per_thread = 4, .singular_floor = 1e-6};
    using Default = tdls::CooperativeLUppSolverStatic<double, 4, standard>;
    using Raised  = tdls::CooperativeLUppSolverStatic<double, 4, raised>;

    double work[Default::workspace_size];
    int piv[4];
    const bool ok_default = Default::solve_inplace<true, true, true>(0, A, 1, piv, 1, y, 1, work);
    const bool ok_raised  = Raised::solve_inplace<true, true, true>(0, A2, 1, piv, 1, y2, 1, work);
    // snippet end

    if (!ok_default || ok_raised) return 1;
    return snippets::check(y, expected, 4);
}
