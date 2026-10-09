/// \file
/// \brief Anchor suite of the static CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// CooperativeLUppSolverStatic::solve_inplace, the original MAGMA kernel,
/// is compared against the independent reference LU
/// (tests/common/reference_lu.hpp) on a grid of shapes: the 1 x 1 corner,
/// small and odd dimensions, and dimensions up to 32. The sequential path
/// (one thread per system) covers the whole grid; groups of CPU threads
/// cover a full mapping, a mapping with a mixed slot (some threads holding
/// a phantom row) and one row per thread, the mapping of MAGMA. The three
/// scalar types and both input regimes (default +-0.5 and tiny entries
/// +-5e-10) are covered. The verdict of every case is the normwise
/// backward error of both solvers, plus the exact agreement of the
/// singularity verdicts (one structurally singular system is injected in
/// every batch).

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Runs one anchor comparison on a reproducible batch: solves with
/// the cooperative solver (external matrix and right-hand side, pivot
/// slices of the threads) and with the reference, requires identical
/// verdicts, uniform across the group, exactly one singular report on each
/// side, and both backward errors under the tolerance.
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam rows_per_thread rows held by each thread
/// \tparam unroll          unroll policy of the configuration (by default
///         the policy of the test configurations, see
///         tdls_tests::test_unroll)
/// \param[in] count     number of systems
/// \param[in] bound     half-width of the entry distribution
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, int N, int rows_per_thread,
         bool unroll = tdls_tests::test_unroll<N, rows_per_thread>>
void anchor_case(const int count, const double bound, const double tolerance,
                 const std::uint64_t seed) {
    constexpr auto config =
        tdls::CooperativeLUppConfig<T>{.rows_per_thread = rows_per_thread, .unroll_loops = unroll};
    using Runner = tdls_tests::GroupRunner<T, N, config, false, true, false>;
    auto batch   = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree    = true;
    bool verdicts_uniform  = true;
    int singular_coop      = 0;
    int singular_reference = 0;
    double be_coop         = 0.0;
    double be_reference    = 0.0;

    std::vector<T> A_out(N * N), Ar(N * N), x(N), xr(N);
    int piv[N], pivr[N];
    for (int s = 0; s < count; ++s) {
        bool uniform       = false;
        const bool ok_coop = Runner::template run<tdls_tests::Path::fused>(
            batch.matrix(s), batch.rhs(s), A_out.data(), piv, x.data(), uniform);
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, Ar.begin());
        const bool ok_reference =
            tdls_tests::reference_solve(Ar.data(), pivr, batch.rhs(s), xr.data(), N);
        if (!uniform) verdicts_uniform = false;
        if (ok_coop != ok_reference) verdicts_agree = false;
        if (!ok_coop) ++singular_coop;
        if (!ok_reference) ++singular_reference;
        if (ok_coop && ok_reference) {
            be_coop = std::max(
                be_coop, tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), N));
            be_reference =
                std::max(be_reference,
                         tdls_tests::backward_error(batch.matrix(s), xr.data(), batch.rhs(s), N));
        }
    }
    TDLS_CHECK(verdicts_uniform);
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(singular_coop == 1);
    TDLS_CHECK(singular_reference == 1);
    TDLS_CHECK_LE(be_coop, tolerance);
    TDLS_CHECK_LE(be_reference, tolerance);
}

} // namespace

/// Emits the default and tiny-entry anchor cases of one (type, N,
/// rows_per_thread) cell.
#define TDLS_ANCHOR_CASES(T, N, ROWS, COUNT, TOL, SEED)                                            \
    TDLS_TEST_CASE("cooperativelupp/oracle/static/" #T "/N=" #N ",rows_per_thread=" #ROWS          \
                   ",default") {                                                                   \
        anchor_case<T, N, ROWS>(COUNT, 0.5, TOL, SEED);                                            \
    }                                                                                              \
    TDLS_TEST_CASE("cooperativelupp/oracle/static/" #T "/N=" #N ",rows_per_thread=" #ROWS          \
                   ",tiny") {                                                                      \
        anchor_case<T, N, ROWS>(COUNT, 5e-10, TOL, SEED + 1);                                      \
    }

// Sequential path, double: one thread per system over the shape grid,
// under the no-pragma branch of the CPU configuration.
TDLS_ANCHOR_CASES(double, 1, 1, 1000, 1e-9, 600110)
TDLS_ANCHOR_CASES(double, 2, 2, 1000, 1e-9, 600220)
TDLS_ANCHOR_CASES(double, 7, 7, 1000, 1e-9, 600770)
TDLS_ANCHOR_CASES(double, 12, 12, 1000, 1e-9, 601212)
TDLS_ANCHOR_CASES(double, 13, 13, 1000, 1e-9, 601313)
TDLS_ANCHOR_CASES(double, 25, 25, 400, 1e-9, 602525)
TDLS_ANCHOR_CASES(double, 32, 32, 400, 1e-9, 603232)

// One thread per system under the forced unrolling, on a small system.
TDLS_TEST_CASE("cooperativelupp/oracle/static/double/N=7,rows_per_thread=7,unrolled,default") {
    anchor_case<double, 7, 7, true>(1000, 0.5, 1e-9, 600780);
}

// Groups of CPU threads, double: a full mapping (N = 12 on 4 threads of 3
// rows), a mixed slot (N = 13 on 3 threads of 5 rows, whose last slot
// holds a real row on thread 0 only), and MAGMA's one row per thread. The
// two small groups run the forced unrolling, one of them with a mixed
// slot (N = 8 on 3 threads of 3 rows).
TDLS_ANCHOR_CASES(double, 12, 3, 200, 1e-9, 611203)
TDLS_ANCHOR_CASES(double, 13, 5, 200, 1e-9, 611305)
TDLS_ANCHOR_CASES(double, 7, 1, 200, 1e-9, 610701)
TDLS_ANCHOR_CASES(double, 8, 3, 200, 1e-9, 610803)

// Float: partial pivoting on the whole column keeps the element growth
// small, so the float backward error stays near n * eps_float.
TDLS_ANCHOR_CASES(float, 12, 12, 1000, 1e-5, 701212)
TDLS_ANCHOR_CASES(float, 13, 5, 200, 1e-5, 711305)

// Long double: the backward error accumulates in double, so the double
// tolerance applies.
TDLS_ANCHOR_CASES(long double, 12, 12, 1000, 1e-9, 801212)

TDLS_TEST_MAIN
