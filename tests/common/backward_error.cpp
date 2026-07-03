/// \file
/// \brief Negative control of the backward-error metric.
/// \author Tristan Chenaille
///
/// Every solver suite anchors its accuracy claim on
/// tdls_tests::backward_error being small (<= 1e-9). That assertion only
/// means something if the metric can also report a *large* value on a
/// wrong solution: a metric that returned ~0 unconditionally would turn
/// every suite green while asserting nothing. This suite pins both ends
/// of the instrument - the exact solution sits at the noise floor, and
/// two deliberately wrong solutions are flagged orders of magnitude
/// above it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

TDLS_TEST_CASE("common/backward_error/discriminates-correct-from-wrong-solutions/N=8") {
    constexpr int N = 8;
    // Well-conditioned regime (bound 0.5): the reference solver returns a
    // solution to machine precision on every generated system.
    const auto batch = tdls_tests::make_batch<double>(N, 64, 20260703, 0.5);

    std::vector<double> A0(N * N), A_fac(N * N), x(N);
    int piv[N];
    int checked = 0;
    for (int s = 0; s < batch.count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A0.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_fac.begin());
        // reference_solve factors A_fac in place, so the residual is
        // always measured against the pristine copy A0.
        if (!tdls_tests::reference_solve(A_fac.data(), piv, batch.rhs(s), x.data(), N)) continue;
        ++checked;

        // Positive end: the exact solution sits below the 1e-9 the suites
        // accept.
        const double be_true = tdls_tests::backward_error(A0.data(), x.data(), batch.rhs(s), N);
        TDLS_CHECK_LE(be_true, 1e-9);

        // Negative control 1: the zero vector. Its residual is exactly -b
        // and its infinity norm is zero, so the metric collapses to
        // |b|/|b| = 1 by construction, independent of conditioning.
        std::vector<double> x_zero(N, 0.0);
        const double be_zero =
            tdls_tests::backward_error(A0.data(), x_zero.data(), batch.rhs(s), N);
        TDLS_CHECK(be_zero > 0.5);

        // Negative control 2: the exact solution with one entry bumped.
        // The bump is scaled to the solution so the flag holds whatever
        // the system's conditioning, and it must dwarf the true error -
        // real discriminating power, not a fixed offset.
        double x_max = 0.0;
        for (int i = 0; i < N; ++i)
            x_max = std::fmax(x_max, std::fabs(x[i]));
        std::vector<double> x_wrong(x);
        x_wrong[0] += std::fmax(1.0, x_max);
        const double be_wrong =
            tdls_tests::backward_error(A0.data(), x_wrong.data(), batch.rhs(s), N);
        TDLS_CHECK(be_wrong > 1e-3);
        TDLS_CHECK(be_wrong > 1e6 * be_true);
    }
    // Floor: the control must actually have run on every generated system
    // (mirrors the anti-vacuity floor of the solver suites).
    TDLS_CHECK(checked == batch.count);
}

} // namespace

TDLS_TEST_MAIN
