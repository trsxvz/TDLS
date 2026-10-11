/// \file
/// \brief Bridge suite of the column-major layout.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Under the column-major layout the solvers factor the stored array
/// read row-major, that is A^T, and solve through the transposed
/// factors. Given the same bytes, the column-major factorization must
/// therefore reproduce the row-major one bitwise: factored array, pivots
/// and out-of-tile counter. The solution has no row-major counterpart;
/// its normwise backward error against A is checked instead, with the
/// tolerances of the oracle suites. The static and dynamic solvers must
/// agree bitwise, and solve_inplace with solve. The suite covers both
/// schedules, trailing tiles, a single partial tile and the scalar
/// corner, in the default and stress regimes, and the unroll_loops knob
/// off. The internal residency gets its own case, bitwise against the
/// external solve; a strided arena case exercises an element stride
/// greater than one.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Solves every system of a reproducible batch, stored
/// column-major, with the column-major static and dynamic solvers, and
/// factors the same bytes with the row-major static solver; checks the
/// bitwise equality of the factored arrays, pivots, out-of-tile counters
/// and singularity verdicts, the bitwise agreement of the column-major
/// solutions (solve and solve_inplace, static and dynamic) and their
/// backward error against A.
/// \tparam T         scalar type
/// \tparam N         system dimension
/// \tparam tile_size tile size
/// \tparam Schedule  elimination schedule
/// \tparam unroll    value of the unroll_loops knob
/// \param[in] count     number of systems
/// \param[in] bound     half-width of the entry distribution
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, int N, int tile_size, tdls::Schedule Schedule, bool unroll = true>
void colmajor_case(const int count, const double bound, const double tolerance,
                   const std::uint64_t seed) {
    constexpr auto config_row = tdls::TiledLUppConfig<T>{
        .tile_size = tile_size, .schedule = Schedule, .unroll_loops = unroll};
    constexpr auto config_col = tdls::TiledLUppConfig<T>{.tile_size    = tile_size,
                                                         .schedule     = Schedule,
                                                         .unroll_loops = unroll,
                                                         .layout = tdls::MatrixLayout::ColMajor};
    using RowStatic           = tdls::TiledLUppSolverStatic<T, N, config_row>;
    using ColStatic           = tdls::TiledLUppSolverStatic<T, N, config_col>;
    using ColDynamic          = tdls::TiledLUppSolverDynamic<T, config_col>;
    auto batch                = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree = true;
    int matched         = 0;

    std::vector<T> Ar(N * N), Ac(N * N), Ad(N * N), Ai(N * N), Aj(N * N), xc(N), xd(N), yi(N),
        yj(N);
    int pivr[N], pivc[N], pivd[N], pivi[N], pivj[N];
    for (int s = 0; s < count; ++s) {
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                Ac[c * N + r] = batch.matrix(s)[r * N + c];
        std::copy(Ac.begin(), Ac.end(), Ar.begin());
        std::copy(Ac.begin(), Ac.end(), Ad.begin());
        std::copy(Ac.begin(), Ac.end(), Ai.begin());
        std::copy(Ac.begin(), Ac.end(), Aj.begin());
        std::copy(batch.rhs(s), batch.rhs(s) + N, yi.begin());
        std::copy(batch.rhs(s), batch.rhs(s) + N, yj.begin());
        int oot_row = 0, oot_col = 0, oot_dyn = 0, oot_sip = 0, oot_dip = 0;
        const bool ok_row =
            RowStatic::template factorize<true, false>(Ar.data(), 1, pivr, 1, oot_row);
        const bool ok_col = ColStatic::template solve<false, true, false>(
            Ac.data(), 1, pivc, 1, batch.rhs(s), xc.data(), 1, oot_col);
        const bool ok_dyn =
            ColDynamic::solve(N, Ad.data(), 1, pivd, 1, batch.rhs(s), xd.data(), 1, oot_dyn);
        const bool ok_sip = ColStatic::template solve_inplace<false, true, false>(
            Ai.data(), 1, pivi, 1, yi.data(), 1, oot_sip);
        const bool ok_dip =
            ColDynamic::solve_inplace(N, Aj.data(), 1, pivj, 1, yj.data(), 1, oot_dip);
        if (ok_row != ok_col || ok_row != ok_dyn || ok_row != ok_sip || ok_row != ok_dip)
            verdicts_agree = false;
        if (!ok_row || !ok_col || !ok_dyn || !ok_sip || !ok_dip) continue;
        ++matched;
        // Same bytes, same factorization: the row-major solver factors A^T.
        TDLS_CHECK_BITWISE(Ar.data(), Ac.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(pivr, pivc, static_cast<std::size_t>(N));
        TDLS_CHECK(oot_row == oot_col);
        TDLS_CHECK_BITWISE(Ac.data(), Ad.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(xc.data(), xd.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivc, pivd, static_cast<std::size_t>(N));
        TDLS_CHECK(oot_col == oot_dyn);
        // solve_inplace, static and dynamic: the same factorization and
        // the same solution as solve.
        TDLS_CHECK_BITWISE(Ac.data(), Ai.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(Ac.data(), Aj.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(pivc, pivi, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivc, pivj, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(xc.data(), yi.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(xc.data(), yj.data(), static_cast<std::size_t>(N));
        TDLS_CHECK(oot_col == oot_sip && oot_col == oot_dip);
        TDLS_CHECK_LE(tdls_tests::backward_error(batch.matrix(s), xc.data(), batch.rhs(s), N),
                      tolerance);
    }
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(matched == count - 1); // every system but the singular one
}

/// \brief Checks the internal residency under the column-major layout:
/// solve_inplace on caller-local arrays must reproduce the external
/// solve bitwise, and its factorization the row-major one of the same
/// bytes.
/// \tparam T         scalar type
/// \tparam N         system dimension
/// \tparam tile_size tile size
/// \tparam Schedule  elimination schedule
/// \tparam unroll    value of the unroll_loops knob
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int tile_size, tdls::Schedule Schedule, bool unroll = true>
void colmajor_internal_case(const int count, const double bound, const std::uint64_t seed) {
    constexpr auto config_row = tdls::TiledLUppConfig<T>{
        .tile_size = tile_size, .schedule = Schedule, .unroll_loops = unroll};
    constexpr auto config_col = tdls::TiledLUppConfig<T>{.tile_size    = tile_size,
                                                         .schedule     = Schedule,
                                                         .unroll_loops = unroll,
                                                         .layout = tdls::MatrixLayout::ColMajor};
    using RowStatic           = tdls::TiledLUppSolverStatic<T, N, config_row>;
    using ColStatic           = tdls::TiledLUppSolverStatic<T, N, config_col>;
    const auto batch          = tdls_tests::make_batch<T>(N, count, seed, bound);

    std::vector<T> Ae(N * N), xe(N);
    int pive[N];
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        T Ar[N * N], Ai[N * N], yi[N];
        int pivr[N], pivi[N];
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                Ar[c * N + r] = batch.matrix(s)[r * N + c];
                Ai[c * N + r] = batch.matrix(s)[r * N + c];
                Ae[c * N + r] = batch.matrix(s)[r * N + c];
            }
        for (int i = 0; i < N; ++i)
            yi[i] = batch.rhs(s)[i];
        const bool ok_row = RowStatic::template factorize<true, true>(Ar, 1, pivr, 1);
        const bool ok_in =
            ColStatic::template solve_inplace<true, true, true>(Ai, 1, pivi, 1, yi, 1);
        const bool ok_ext = ColStatic::template solve<false, true, false>(
            Ae.data(), 1, pive, 1, batch.rhs(s), xe.data(), 1);
        TDLS_CHECK(ok_row == ok_in && ok_in == ok_ext);
        if (!ok_row || !ok_in || !ok_ext) continue;
        ++solved;
        TDLS_CHECK_BITWISE(Ar, Ai, static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(pivr, pivi, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(Ai, Ae.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(pivi, pive, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(yi, xe.data(), static_cast<std::size_t>(N));
    }
    // Floor: every generated system must have been solved.
    TDLS_CHECK(solved == count);
}

/// \brief Checks the column-major solver under an element stride
/// greater than one: every operand of the system under test lives in
/// the middle slot of a three-slot arena and must reproduce the
/// contiguous column-major solve bitwise.
/// \tparam T         scalar type
/// \tparam N         system dimension
/// \tparam tile_size tile size
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int tile_size>
void colmajor_strided_case(const int count, const double bound, const std::uint64_t seed) {
    constexpr int arena = 3;
    constexpr int slot  = 1;
    constexpr auto config_col =
        tdls::TiledLUppConfig<T>{.tile_size = tile_size, .layout = tdls::MatrixLayout::ColMajor};
    using ColStatic  = tdls::TiledLUppSolverStatic<T, N, config_col>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    std::vector<T> Ac(N * N), Ag(N * N), xc(N), xg(N);
    std::vector<T> Aa(static_cast<std::size_t>(N) * N * arena);
    std::vector<T> ba(static_cast<std::size_t>(N) * arena), xa(static_cast<std::size_t>(N) * arena);
    std::vector<int> piva(static_cast<std::size_t>(N) * arena), pivg(N);
    int pivc[N];
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        std::fill(Aa.begin(), Aa.end(), T(0));
        std::fill(ba.begin(), ba.end(), T(0));
        std::fill(xa.begin(), xa.end(), T(0));
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                Ac[c * N + r] = batch.matrix(s)[r * N + c];
        for (int k = 0; k < N * N; ++k)
            Aa[static_cast<std::size_t>(k) * arena + slot] = Ac[k];
        for (int i = 0; i < N; ++i)
            ba[static_cast<std::size_t>(i) * arena + slot] = batch.rhs(s)[i];
        int oot_con = 0, oot_str = 0;
        const bool ok_con = ColStatic::template solve<false, true, false>(
            Ac.data(), 1, pivc, 1, batch.rhs(s), xc.data(), 1, oot_con);
        const bool ok_str = ColStatic::template solve<false, false, false>(
            Aa.data() + slot, arena, piva.data() + slot, arena, ba.data() + slot, xa.data() + slot,
            arena, oot_str);
        TDLS_CHECK(ok_con == ok_str);
        if (!ok_con || !ok_str) continue;
        ++solved;
        for (int k = 0; k < N * N; ++k)
            Ag[k] = Aa[static_cast<std::size_t>(k) * arena + slot];
        for (int i = 0; i < N; ++i) {
            xg[i]   = xa[static_cast<std::size_t>(i) * arena + slot];
            pivg[i] = piva[static_cast<std::size_t>(i) * arena + slot];
        }
        TDLS_CHECK_BITWISE(Ac.data(), Ag.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(xc.data(), xg.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivc, pivg.data(), static_cast<std::size_t>(N));
        TDLS_CHECK(oot_con == oot_str);
    }
    // Floor: every generated system must have been solved.
    TDLS_CHECK(solved == count);
}

} // namespace

/// Emits the RL and LL column-major cases of one (type, N, tile_size, regime) cell.
#define TDLS_COLMAJOR_CASES(T, N, TILE_SIZE, REGIME, COUNT, BOUND, TOL, SEED)                      \
    TDLS_TEST_CASE("tiledlupp/bridge/colmajor/" #T "/N=" #N ",tile_size=" #TILE_SIZE               \
                   ",RL," REGIME) {                                                                \
        colmajor_case<T, N, TILE_SIZE, tdls::Schedule::RightLooking>(COUNT, BOUND, TOL, SEED);     \
    }                                                                                              \
    TDLS_TEST_CASE("tiledlupp/bridge/colmajor/" #T "/N=" #N ",tile_size=" #TILE_SIZE               \
                   ",LL," REGIME) {                                                                \
        colmajor_case<T, N, TILE_SIZE, tdls::Schedule::LeftLooking>(COUNT, BOUND, TOL, SEED + 1);  \
    }

// Nominal divisible grid and a trailing-tile grid, both regimes. The
// tolerances are those of the oracle suites (see test_oracle_static.cpp).
TDLS_COLMAJOR_CASES(double, 12, 3, "default", 200, 0.5, 1e-9, 250100)
TDLS_COLMAJOR_CASES(double, 13, 6, "default", 200, 0.5, 1e-9, 250200)
TDLS_COLMAJOR_CASES(double, 12, 3, "stress", 200, 5e-10, 1e-9, 250300)
TDLS_COLMAJOR_CASES(double, 13, 6, "stress", 200, 5e-10, 1e-9, 250400)
// Single partial tile and the scalar corner.
TDLS_COLMAJOR_CASES(double, 5, 8, "default", 200, 0.5, 1e-9, 250500)
TDLS_COLMAJOR_CASES(double, 1, 1, "default", 200, 0.5, 1e-9, 250600)
// Float cell.
TDLS_COLMAJOR_CASES(float, 12, 3, "default", 200, 0.5, 1e-2, 250700)
// The unroll_loops knob off.
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/double/N=13,tile_size=6,RL,no-unroll") {
    colmajor_case<double, 13, 6, tdls::Schedule::RightLooking, false>(200, 0.5, 1e-9, 251100);
}

TDLS_TEST_CASE("tiledlupp/bridge/colmajor/internal/double/N=12,tile_size=3,RL,default") {
    colmajor_internal_case<double, 12, 3, tdls::Schedule::RightLooking>(200, 0.5, 250800);
}
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/internal/double/N=12,tile_size=3,LL,default") {
    colmajor_internal_case<double, 12, 3, tdls::Schedule::LeftLooking>(200, 0.5, 251200);
}
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/internal/double/N=13,tile_size=6,RL,stress") {
    colmajor_internal_case<double, 13, 6, tdls::Schedule::RightLooking>(200, 5e-10, 250900);
}
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/internal/double/N=13,tile_size=6,LL,no-unroll") {
    colmajor_internal_case<double, 13, 6, tdls::Schedule::LeftLooking, false>(200, 0.5, 251300);
}
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/strided/double/N=12,tile_size=3,default") {
    colmajor_strided_case<double, 12, 3>(100, 0.5, 251000);
}

TDLS_TEST_MAIN
