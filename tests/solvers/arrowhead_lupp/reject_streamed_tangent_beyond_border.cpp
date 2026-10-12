/// \file
/// \brief Negative compilation test: solve_inplace_streamed_or_tangent with
/// more tangent columns than the border must be rejected.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the contract diagnostic of
/// the ArrowheadLUpp solver (the tangent columns are columns of the
/// border).

#include <tdls/tdls.hpp>

namespace {

constexpr auto shape  = tdls::ArrowheadShape{6, 2, 3, 0};
constexpr auto config = tdls::ArrowheadLUppConfig<double>{};
using Solver          = tdls::ArrowheadLUppSolverStatic<double, shape, config>;

/// \brief A provider that writes nothing: the call must not compile anyway.
/// \param[in] k   block
/// \param[in] blk block
void provider(const int k, Solver::StreamedBlock& blk) {
    static_cast<void>(k);
    static_cast<void>(blk);
}

/// \return whether the iteration converged
bool converged() {
    return true;
}

} // namespace

int main() {
    double M[1024] = {};
    double w[1024] = {};
    double y[64]   = {};
    double x[64]   = {};
    int refused    = 0;
    const auto o   = Solver::solve_inplace_streamed_or_tangent<7, true, true, true>(
        provider, converged, M, 1, w, 1, y, 1, 0, x, 1, 6, refused);
    return o == Solver::StreamedOutcome::tangent ? 0 : 1;
}
