/// \file
/// \brief Documentation snippet, must not compile: a gather view.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <array>

#include <TFEL/Math/Array/CoalescedView.hxx>
#include <TFEL/Math/tmatrix.hxx>

#include <tdls/tdls.hpp>

int main() {
    double cells[16];
    std::array<double*, 16> pointers;
    for (int i = 0; i < 16; ++i)
        pointers[i] = cells + i;
    int piv[4];

    // snippet begin
    // one pointer per element: the view has no data() and no stride
    auto G = tfel::math::map<tfel::math::tmatrix<4, 4, double>>(pointers);

    return tdls::factorize(G, piv) ? 0 : 1;
    // snippet end
}
