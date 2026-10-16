/// \file
/// \brief Negative compilation test: a compile-time stride that is neither
/// tdls::dynamic_stride nor positive must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the stride contract
/// diagnostic shared by the compile-time solvers (tdls::detail::stride_or):
/// a stride of 0 would alias every entry of the operand, and a bool passed
/// by mistake where a stride is expected becomes 0 or 1.

#include <tdls/tdls.hpp>

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9>;
    return Solver::solve_inplace<false, false, false, 0>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
