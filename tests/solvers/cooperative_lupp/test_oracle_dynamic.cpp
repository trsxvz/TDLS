/// \file
/// \brief Anchor suite of the dynamic CooperativeLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// CooperativeLUppSolverDynamic::solve_inplace is compared against the
/// independent reference LU on the shape grid of the static anchor, plus
/// shapes the static suite does not cover: rows_per_thread exceeding the
/// dimension (one thread holding phantom slots) and dimensions beyond the
/// static grid, up to 100. Physical row interchanges and a relative pivot
/// threshold of 0.1 are anchored as well, up to 100. The verdict of every
/// case is the normwise
/// backward error of both solvers, plus the exact agreement of the
/// singularity verdicts, uniform across the group (one structurally
/// singular system is injected in every batch).

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
/// the dynamic cooperative solver (external storage, runtime dimension)
/// and with the reference, requires identical verdicts, uniform across the
/// group, exactly one singular report on each side, and both backward
/// errors under the tolerance.
/// \tparam T               scalar type
/// \tparam rows_per_thread rows held by each thread
/// \tparam interchange     row interchanges
/// \tparam relaxed         relative pivot threshold 0.1 instead of 1
/// \param[in] n         system dimension
/// \param[in] count     number of systems
/// \param[in] bound     half-width of the entry distribution
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, int rows_per_thread,
         tdls::RowInterchange interchange = tdls::RowInterchange::Logical, bool relaxed = false>
void anchor_case(const int n, const int count, const double bound, const double tolerance,
                 const std::uint64_t seed) {
    constexpr auto config =
        tdls::CooperativeLUppConfig<T>{.rows_per_thread          = rows_per_thread,
                                       .row_interchange          = interchange,
                                       .relative_pivot_threshold = relaxed ? 0.1 : 1.0};
    using Runner = tdls_tests::DynamicGroupRunner<T, config>;
    auto batch   = tdls_tests::make_batch<T>(n, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree    = true;
    bool verdicts_uniform  = true;
    int singular_coop      = 0;
    int singular_reference = 0;
    double be_coop         = 0.0;
    double be_reference    = 0.0;

    std::vector<T> A_out(static_cast<std::size_t>(n) * n), Ar(static_cast<std::size_t>(n) * n),
        x(n), xr(n);
    std::vector<int> piv(n), pivr(n);
    for (int s = 0; s < count; ++s) {
        bool uniform       = false;
        const bool ok_coop = Runner::template run<tdls_tests::Path::fused>(
            n, batch.matrix(s), batch.rhs(s), A_out.data(), piv.data(), x.data(), uniform);
        std::copy(batch.matrix(s), batch.matrix(s) + n * n, Ar.begin());
        const bool ok_reference =
            tdls_tests::reference_solve(Ar.data(), pivr.data(), batch.rhs(s), xr.data(), n);
        if (!uniform) verdicts_uniform = false;
        if (ok_coop != ok_reference) verdicts_agree = false;
        if (!ok_coop) ++singular_coop;
        if (!ok_reference) ++singular_reference;
        if (ok_coop && ok_reference) {
            be_coop = std::max(
                be_coop, tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), n));
            be_reference =
                std::max(be_reference,
                         tdls_tests::backward_error(batch.matrix(s), xr.data(), batch.rhs(s), n));
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

/// Emits the default and tiny-entry anchor cases of one (type, n,
/// rows_per_thread) cell.
#define TDLS_ANCHOR_CASES(T, N, ROWS, COUNT, TOL, SEED)                                            \
    TDLS_TEST_CASE("cooperativelupp/oracle/dynamic/" #T "/n=" #N ",rows_per_thread=" #ROWS         \
                   ",default") {                                                                   \
        anchor_case<T, ROWS>(N, COUNT, 0.5, TOL, SEED);                                            \
    }                                                                                              \
    TDLS_TEST_CASE("cooperativelupp/oracle/dynamic/" #T "/n=" #N ",rows_per_thread=" #ROWS         \
                   ",tiny") {                                                                      \
        anchor_case<T, ROWS>(N, COUNT, 5e-10, TOL, SEED + 1);                                      \
    }

// Sequential path, double: the static grid, then dimensions beyond it.
TDLS_ANCHOR_CASES(double, 1, 1, 1000, 1e-9, 680110)
TDLS_ANCHOR_CASES(double, 2, 2, 1000, 1e-9, 680220)
TDLS_ANCHOR_CASES(double, 7, 7, 1000, 1e-9, 680770)
TDLS_ANCHOR_CASES(double, 12, 12, 1000, 1e-9, 681212)
TDLS_ANCHOR_CASES(double, 13, 13, 1000, 1e-9, 681313)
TDLS_ANCHOR_CASES(double, 32, 32, 400, 1e-9, 683232)
TDLS_ANCHOR_CASES(double, 100, 100, 60, 1e-8, 690100)
// rows_per_thread exceeding the dimension: one thread, phantom slots.
TDLS_ANCHOR_CASES(double, 5, 8, 1000, 1e-9, 680508)

// Groups of CPU threads, double: a full mapping, a mixed slot and MAGMA's
// one row per thread.
TDLS_ANCHOR_CASES(double, 12, 3, 200, 1e-9, 691203)
TDLS_ANCHOR_CASES(double, 13, 5, 200, 1e-9, 691305)
TDLS_ANCHOR_CASES(double, 7, 1, 200, 1e-9, 690701)

// Float: partial pivoting on the whole column keeps the element growth
// small, so the float backward error stays near n * eps_float.
TDLS_ANCHOR_CASES(float, 12, 12, 1000, 1e-5, 781212)
TDLS_ANCHOR_CASES(float, 13, 5, 200, 1e-5, 791305)

// Long double: the backward error accumulates in double, so the double
// tolerance applies.
TDLS_ANCHOR_CASES(long double, 12, 12, 1000, 1e-9, 881212)

/// Emits the default and tiny-entry anchor cases of one (type, n,
/// rows_per_thread) cell under the given pivoting.
#define TDLS_PIVOTING_ANCHOR_CASES(T, N, ROWS, INTERCHANGE, RELAXED, NAME, COUNT, TOL, SEED)       \
    TDLS_TEST_CASE("cooperativelupp/oracle/dynamic/" #T "/n=" #N ",rows_per_thread=" #ROWS         \
                   "," NAME ",default") {                                                          \
        anchor_case<T, ROWS, tdls::RowInterchange::INTERCHANGE, RELAXED>(N, COUNT, 0.5, TOL,       \
                                                                         SEED);                    \
    }                                                                                              \
    TDLS_TEST_CASE("cooperativelupp/oracle/dynamic/" #T "/n=" #N ",rows_per_thread=" #ROWS         \
                   "," NAME ",tiny") {                                                             \
        anchor_case<T, ROWS, tdls::RowInterchange::INTERCHANGE, RELAXED>(N, COUNT, 5e-10, TOL,     \
                                                                         SEED + 1);                \
    }

// Physical row interchanges: the sequential path up to 100, a mixed slot
// and one row per thread.
TDLS_PIVOTING_ANCHOR_CASES(double, 32, 32, Physical, false, "physical", 400, 1e-9, 693232)
TDLS_PIVOTING_ANCHOR_CASES(double, 100, 100, Physical, false, "physical", 60, 1e-8, 694100)
TDLS_PIVOTING_ANCHOR_CASES(double, 13, 5, Physical, false, "physical", 200, 1e-9, 693305)
TDLS_PIVOTING_ANCHOR_CASES(double, 7, 1, Physical, false, "physical", 200, 1e-9, 692701)
// Relative pivot threshold 0.1, under both row interchanges.
TDLS_PIVOTING_ANCHOR_CASES(double, 32, 32, Physical, true, "physical,threshold", 400, 1e-9, 695232)
TDLS_PIVOTING_ANCHOR_CASES(double, 13, 5, Logical, true, "threshold", 200, 1e-9, 695305)

TDLS_TEST_MAIN
