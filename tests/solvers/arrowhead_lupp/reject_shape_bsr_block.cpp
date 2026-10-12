/// \file
/// \brief Negative compilation test: a Bsr storage whose border is not a multiple of the block must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the contract diagnostic of
/// the ArrowheadLUpp solver (the square blocks of Bsr tile the border only when the block divides it).

#include <tdls/tdls.hpp>

namespace {

constexpr auto shape  = tdls::ArrowheadShape{8, 6, 10, 0};
constexpr auto config = tdls::ArrowheadLUppConfig<double>{.storage = tdls::ArrowheadStorage::Bsr};

} // namespace

int main() {
    double M[1024] = {};
    double w[1024] = {};
    double y[64]   = {};
    using Solver   = tdls::ArrowheadLUppSolverStatic<double, shape, config>;
    return Solver::solve_inplace<true, true, true>(M, 1, w, 1, y, 1) ? 0 : 1;
}
