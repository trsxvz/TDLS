/// \file
/// \brief Bridge suite of the column-major layout.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The layout knob of TiledLUppConfig only remaps the flat index of
/// the matrix elements. On identical inputs stored transposed, the
/// column-major solvers must therefore reproduce the row-major ones
/// bitwise: factored matrix, pivot, solution and out-of-tile counter.
/// The suite checks it on both solvers and both schedules, over shapes
/// covering trailing tiles, a single partial tile and the scalar
/// corner, in the default and stress regimes. The internal residency
/// and the fused entry point get their own cases; a strided arena case
/// exercises the interaction with an element stride greater than one.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Solves every system of a reproducible batch with the
/// row-major static solver, and with the column-major static and
/// dynamic solvers on transposed storage; checks the bitwise equality
/// of every output, singularity verdicts and out-of-tile counters
/// included.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Schedule elimination schedule
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::Schedule Schedule>
void colmajor_case(const int count, const double bound, const std::uint64_t seed) {
    constexpr auto config_row = tdls::TiledLUppConfig<T>{.tile_size = TS, .schedule = Schedule};
    constexpr auto config_col = tdls::TiledLUppConfig<T>{
        .tile_size = TS, .schedule = Schedule, .layout = tdls::MatrixLayout::ColMajor};
    using RowStatic  = tdls::TiledLUppSolverStatic<T, N, config_row>;
    using ColStatic  = tdls::TiledLUppSolverStatic<T, N, config_col>;
    using ColDynamic = tdls::TiledLUppSolverDynamic<T, config_col>;
    auto batch       = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree = true;
    int matched         = 0;

    std::vector<T> Ar(N * N), Ac(N * N), Ad(N * N), At(N * N), xr(N), xc(N), xd(N);
    int pivr[N], pivc[N], pivd[N];
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, Ar.begin());
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                Ac[c * N + r] = batch.matrix(s)[r * N + c];
        std::copy(Ac.begin(), Ac.end(), Ad.begin());
        int oot_row = 0, oot_col = 0, oot_dyn = 0;
        const bool ok_row = RowStatic::template solve<false, true, false>(
            Ar.data(), 1, pivr, 1, batch.rhs(s), xr.data(), 1, oot_row);
        const bool ok_col = ColStatic::template solve<false, true, false>(
            Ac.data(), 1, pivc, 1, batch.rhs(s), xc.data(), 1, oot_col);
        const bool ok_dyn =
            ColDynamic::solve(N, Ad.data(), 1, pivd, 1, batch.rhs(s), xd.data(), 1, oot_dyn);
        if (ok_row != ok_col || ok_row != ok_dyn) verdicts_agree = false;
        if (!ok_row || !ok_col || !ok_dyn) continue;
        ++matched;
        // The column-major factors transpose back onto the row-major ones.
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                At[r * N + c] = Ac[c * N + r];
        TDLS_CHECK_BITWISE(Ar.data(), At.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(xr.data(), xc.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivr, pivc, static_cast<std::size_t>(N));
        TDLS_CHECK(oot_row == oot_col);
        TDLS_CHECK_BITWISE(Ac.data(), Ad.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(xc.data(), xd.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivc, pivd, static_cast<std::size_t>(N));
        TDLS_CHECK(oot_col == oot_dyn);
    }
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(matched == count - 1); // every system but the singular one
}

/// \brief Checks the internal residency and the fused entry point under
/// the column-major layout: caller-local arrays, both layouts, bitwise.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS>
void colmajor_internal_case(const int count, const double bound, const std::uint64_t seed) {
    constexpr auto config_col =
        tdls::TiledLUppConfig<T>{.tile_size = TS, .layout = tdls::MatrixLayout::ColMajor};
    using RowStatic  = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T>{.tile_size = TS}>;
    using ColStatic  = tdls::TiledLUppSolverStatic<T, N, config_col>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    int solved = 0;
    for (int s = 0; s < count; ++s) {
        T Ar[N * N], Ac[N * N], At[N * N], yr[N], yc[N];
        int pivr[N], pivc[N];
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                Ar[r * N + c] = batch.matrix(s)[r * N + c];
                Ac[c * N + r] = batch.matrix(s)[r * N + c];
            }
        for (int i = 0; i < N; ++i) {
            yr[i] = batch.rhs(s)[i];
            yc[i] = batch.rhs(s)[i];
        }
        const bool ok_row =
            RowStatic::template solve_inplace<true, true, true>(Ar, 1, pivr, 1, yr, 1);
        const bool ok_col =
            ColStatic::template solve_inplace<true, true, true>(Ac, 1, pivc, 1, yc, 1);
        TDLS_CHECK(ok_row == ok_col);
        if (!ok_row || !ok_col) continue;
        ++solved;
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                At[r * N + c] = Ac[c * N + r];
        TDLS_CHECK_BITWISE(Ar, At, static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(yr, yc, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivr, pivc, static_cast<std::size_t>(N));
    }
    // Floor: every generated system must have been solved.
    TDLS_CHECK(solved == count);
}

/// \brief Checks the column-major addressing under an element stride
/// greater than one: every operand of the system under test lives in
/// the middle slot of a three-slot arena and must reproduce the
/// contiguous row-major solve bitwise.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS>
void colmajor_strided_case(const int count, const double bound, const std::uint64_t seed) {
    constexpr int arena = 3;
    constexpr int slot  = 1;
    constexpr auto config_col =
        tdls::TiledLUppConfig<T>{.tile_size = TS, .layout = tdls::MatrixLayout::ColMajor};
    using RowStatic  = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T>{.tile_size = TS}>;
    using ColStatic  = tdls::TiledLUppSolverStatic<T, N, config_col>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    std::vector<T> Ar(N * N), At(N * N), xr(N), xg(N);
    std::vector<T> Aa(static_cast<std::size_t>(N) * N * arena);
    std::vector<T> ba(static_cast<std::size_t>(N) * arena), xa(static_cast<std::size_t>(N) * arena);
    std::vector<int> piva(static_cast<std::size_t>(N) * arena), pivg(N);
    int pivr[N];
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, Ar.begin());
        std::fill(Aa.begin(), Aa.end(), T(0));
        std::fill(ba.begin(), ba.end(), T(0));
        std::fill(xa.begin(), xa.end(), T(0));
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                Aa[static_cast<std::size_t>(c * N + r) * arena + slot] = batch.matrix(s)[r * N + c];
        for (int i = 0; i < N; ++i)
            ba[static_cast<std::size_t>(i) * arena + slot] = batch.rhs(s)[i];
        int oot_row = 0, oot_col = 0;
        const bool ok_row = RowStatic::template solve<false, true, false>(
            Ar.data(), 1, pivr, 1, batch.rhs(s), xr.data(), 1, oot_row);
        const bool ok_col = ColStatic::template solve<false, false, false>(
            Aa.data() + slot, arena, piva.data() + slot, arena, ba.data() + slot, xa.data() + slot,
            arena, oot_col);
        TDLS_CHECK(ok_row == ok_col);
        if (!ok_row || !ok_col) continue;
        ++solved;
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                At[r * N + c] = Aa[static_cast<std::size_t>(c * N + r) * arena + slot];
        for (int i = 0; i < N; ++i) {
            xg[i]   = xa[static_cast<std::size_t>(i) * arena + slot];
            pivg[i] = piva[static_cast<std::size_t>(i) * arena + slot];
        }
        TDLS_CHECK_BITWISE(Ar.data(), At.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(xr.data(), xg.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivr, pivg.data(), static_cast<std::size_t>(N));
        TDLS_CHECK(oot_row == oot_col);
    }
    // Floor: every generated system must have been solved.
    TDLS_CHECK(solved == count);
}

} // namespace

/// Emits the RL and LL column-major cases of one (type, N, TS, regime) cell.
#define TDLS_COLMAJOR_CASES(T, N, TS, REGIME, COUNT, BOUND, SEED)                                  \
    TDLS_TEST_CASE("tiledlupp/bridge/colmajor/" #T "/N=" #N ",TS=" #TS ",RL," REGIME) {            \
        colmajor_case<T, N, TS, tdls::Schedule::RightLooking>(COUNT, BOUND, SEED);                 \
    }                                                                                              \
    TDLS_TEST_CASE("tiledlupp/bridge/colmajor/" #T "/N=" #N ",TS=" #TS ",LL," REGIME) {            \
        colmajor_case<T, N, TS, tdls::Schedule::LeftLooking>(COUNT, BOUND, SEED + 1);              \
    }

// Nominal divisible grid and a trailing-tile grid, both regimes.
TDLS_COLMAJOR_CASES(double, 12, 3, "default", 200, 0.5, 250100)
TDLS_COLMAJOR_CASES(double, 13, 6, "default", 200, 0.5, 250200)
TDLS_COLMAJOR_CASES(double, 12, 3, "stress", 200, 5e-10, 250300)
TDLS_COLMAJOR_CASES(double, 13, 6, "stress", 200, 5e-10, 250400)
// Single partial tile and the scalar corner.
TDLS_COLMAJOR_CASES(double, 5, 8, "default", 200, 0.5, 250500)
TDLS_COLMAJOR_CASES(double, 1, 1, "default", 200, 0.5, 250600)
// Float cell.
TDLS_COLMAJOR_CASES(float, 12, 3, "default", 200, 0.5, 250700)

TDLS_TEST_CASE("tiledlupp/bridge/colmajor/internal/double/N=12,TS=3,default") {
    colmajor_internal_case<double, 12, 3>(200, 0.5, 250800);
}
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/internal/double/N=13,TS=6,stress") {
    colmajor_internal_case<double, 13, 6>(200, 5e-10, 250900);
}
TDLS_TEST_CASE("tiledlupp/bridge/colmajor/strided/double/N=12,TS=3,default") {
    colmajor_strided_case<double, 12, 3>(100, 0.5, 251000);
}

TDLS_TEST_MAIN
