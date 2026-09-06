/// \file
/// \brief Documentation snippet: the pivot forms, on fixed-size TFEL objects.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.

#include <TFEL/Math/fsarray.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

#include "check.hpp"

int main() {
    // snippet begin
    tfel::math::tmatrix<4, 4, double> A1{4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7};
    tfel::math::tmatrix<4, 4, double> A2 = A1, A3 = A1, A4 = A1;

    // the pivot is any dense int object of extent 4, or a raw int array or pointer
    tfel::math::fsarray<4, int> piv_fsarray;
    tfel::math::tvector<4, int> piv_tvector;
    int piv_array[4];
    int storage[4];
    int* piv_pointer = storage;

    const bool ok = tdls::factorize(A1, piv_fsarray) && tdls::factorize(A2, piv_tvector) &&
                    tdls::factorize(A3, piv_array) && tdls::factorize(A4, piv_pointer);
    // snippet end

    if (!ok) return 1;
    return snippets::check_bitwise(A1.data(), A2.data(), 16) |
           snippets::check_bitwise(A1.data(), A3.data(), 16) |
           snippets::check_bitwise(A1.data(), A4.data(), 16) |
           snippets::check_bitwise(piv_fsarray.data(), piv_array, 4) |
           snippets::check_bitwise(piv_tvector.data(), storage, 4);
}
