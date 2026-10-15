/// \file
/// \brief Bridge suite: the streamed entry points of the ArrowheadLUpp
/// solvers against the stored ones, bit for bit, and the contract of the
/// provider.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A streamed solve receives the blocks from a provider when it condenses
/// them; the arithmetic, the pivot check and the second pass are those of
/// the stored solve. On the standard batch (dominant and random blocks,
/// singular and pivot-poor ones, so that blocks are refused), the suite
/// compares, bit for bit, for both solvers:
///   - solve_inplace_streamed against solve_inplace;
///   - solve_border_canonical_multirhs_streamed against
///     solve_border_canonical_multirhs, every border column;
///   - solve_inplace_streamed_or_tangent: the tangent (converged) against
///     the canonical solve, refused blocks included, and the Newton pass
///     (not converged) against solve_inplace;
///   - the runtime solver against the compile-time one on every streamed
///     entry point.
/// It also checks the contract of the provider: called exactly once per
/// block, in order, and converged() once, after the last block. The
/// streamed storage starts as NaN except the border block, so that a read
/// of an entry the provider did not write would show. The paths cover the
/// four storages, both couplings and layouts, passes of several widths,
/// the gather of the right-hand sides, the Schur complement accumulated in
/// place, the unroll policy off, and the external residency (strided).
///
/// The streamed and stored paths are distinct code: a compiler contracting
/// multiplications and additions across statements could fuse them
/// differently. The suite is built with -ffp-contract=off where the
/// compiler knows the flag (tests/solvers/CMakeLists.txt).

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"

namespace {

/// \brief A configuration from its paths.
/// \param[in] storage  storage
/// \param[in] diag     diagonal coupling
/// \param[in] col      column-major layout
/// \param[in] pass     rhs_pass_width
/// \param[in] gather   gather_rhs
/// \param[in] in_place schur_in_place
/// \param[in] unroll   unroll_loops
/// \param[in] outline  streamed_out_of_line
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double>
path_config(const tdls::ArrowheadStorage storage, const bool diag, const bool col, const int pass,
            const bool gather = false, const bool in_place = false, const bool unroll = true,
            const bool outline = false) {
    tdls::ArrowheadLUppConfig<double> c;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling       = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.storage        = storage;
    c.layout         = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.rhs_pass_width = pass;
    c.gather_rhs     = gather;
    c.schur_in_place = in_place;
    c.unroll_loops   = unroll;
    c.streamed_out_of_line = outline;
    return c;
}

/// \brief The configuration c with its blocks eliminated in registers.
/// \param[in] c configuration
/// \return c with the blocks eliminated in registers
constexpr tdls::ArrowheadLUppConfig<double> registers(tdls::ArrowheadLUppConfig<double> c) {
    c.block_row_interchange = tdls::RowInterchange::Physical;
    return c;
}

/// \brief Calls of the provider and of converged() during one solve.
struct CallLog {
    int blocks    = 0;    ///< provider calls
    int next      = 0;    ///< block expected at the next call
    int converged = 0;    ///< converged() calls
    bool ordered  = true; ///< every block asked for in order, once
    bool late     = true; ///< converged() called after the last block only
};

/// \brief A provider that fills block k from the full matrix of a problem,
/// for the blocks of both solvers (same accessors), logging its calls.
/// \tparam diag diagonal coupling (the accessor of the couplings)
/// \tparam T    scalar type
/// \param[in]     p   problem
/// \param[in,out] log the calls
/// \return the provider
template<bool diag, typename T>
auto make_provider(const tdls_tests::ArrowProblem<T>& p, CallLog& log) {
    return [&p, &log](const int k, auto& blk) {
        const tdls::ArrowheadShape& s = p.shape;
        const int B = s.border, S = s.block;
        log.ordered = log.ordered && k == log.next && log.converged == 0;
        ++log.next;
        ++log.blocks;
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            for (int j = 0; j < S; ++j)
                blk.d(i, j) = p.at(row, B + k * S + j);
            for (int c = 0; c < B; ++c)
                blk.akb(i, c) = p.at(row, c);
            blk.y(i) = p.b[static_cast<std::size_t>(row)];
        }
        if constexpr (diag)
            for (int i = 0; i < S; ++i)
                blk.coupling(i) = p.at(s.coupling_row + i, B + k * S + i);
        else
            for (int r = 0; r < B; ++r)
                for (int i = 0; i < S; ++i)
                    blk.abk(r, i) = p.at(r, B + k * S + i);
    };
}

/// \brief A converged() that answers `answer` and logs its calls.
/// \param[in]     answer   whether the iteration converged
/// \param[in]     nblocks  number of blocks
/// \param[in,out] log      the calls
/// \return the callable
auto make_converged(const bool answer, const int nblocks, CallLog& log) {
    return [answer, nblocks, &log] {
        log.late = log.late && log.blocks == nblocks;
        ++log.converged;
        return answer;
    };
}

/// \brief Whether one streamed solve called its provider once per block in
/// order and converged() `converged` times after the last block; resets
/// the log.
/// \param[in,out] log       the calls
/// \param[in]     nblocks   number of blocks
/// \param[in]     converged expected calls of converged()
/// \return true when the contract holds
bool contract_held(CallLog& log, const int nblocks, const int converged) {
    const bool held = log.ordered && log.late && log.blocks == nblocks && log.next == nblocks &&
                      log.converged == converged;
    log             = CallLog{};
    return held;
}

/// \brief Strided vector of n entries, stride s, all `value`.
/// \param[in] n     entries
/// \param[in] s     stride
/// \param[in] value value of every slot
/// \return the vector
std::vector<double> strided(const int n, const int s, const double value) {
    return std::vector<double>(static_cast<std::size_t>(n) * static_cast<std::size_t>(s), value);
}

/// \brief Bitwise equality of the n strided entries of two vectors.
/// \param[in] a first vector
/// \param[in] b second vector
/// \param[in] n entries compared
/// \param[in] s stride
/// \return true when the entries have the same bits
bool same_strided(const std::vector<double>& a, const std::vector<double>& b, const int n,
                  const int s) {
    for (int i = 0; i < n; ++i) {
        const auto e = static_cast<std::size_t>(i) * static_cast<std::size_t>(s);
        if (std::memcmp(&a[e], &b[e], sizeof(double)) != 0) return false;
    }
    return true;
}

/// \brief Counters of a suite case, for its floors.
struct Floors {
    int refused        = 0; ///< refused blocks seen
    int tangent_plain  = 0; ///< tangents from a pass without refused blocks
    int tangent_second = 0; ///< tangents from a pass with refused blocks
};

/// \brief Compares the streamed and stored entry points of both solvers on
/// a standard batch.
/// \tparam Shape    shape of the matrices
/// \tparam Config   configuration
/// \tparam external strided arrays (stride 3) instead of caller-local ones
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config, bool external>
void streamed_case(const int count, const std::uint64_t seed) {
    using Static        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Dynamic       = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    using Map           = typename Static::storage_map;
    using SO            = typename Static::StreamedOutcome;
    using DO            = typename Dynamic::StreamedOutcome;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    constexpr int n = Static::dimension, B = Shape.border, NB = Shape.blocks;
    constexpr int st  = external ? 3 : 1;
    constexpr bool in = !external;
    const double nan  = std::numeric_limits<double>::quiet_NaN();
    const int msz     = Static::storage_size;
    // workspaces: the stored and streamed static solves, the runtime ones
    auto ws  = strided(Static::template multirhs_workspace_size<B>, st, 0.0);
    auto wd  = strided(Dynamic::workspace_size(Shape, B), st, 0.0);
    auto wds = strided(Dynamic::streamed_workspace_size(Shape, B), st, 0.0);
    std::vector<int> piv(static_cast<std::size_t>(Dynamic::pivot_size(Shape)) * st);
    Floors floors;
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        const auto p  = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const auto Ms = tdls_tests::store_arrow<Map>(p);
        CallLog log;
        auto provider = make_provider<diag>(p, log);
        // the full matrix and right-hand side, strided
        auto full = [&] {
            auto A = strided(msz, st, nan);
            for (int e = 0; e < msz; ++e)
                A[static_cast<std::size_t>(e * st)] = Ms[static_cast<std::size_t>(e)];
            return A;
        };
        auto rhs = [&](const int rows) {
            auto y = strided(n, st, nan);
            for (int i = 0; i < rows; ++i)
                y[static_cast<std::size_t>(i * st)] = p.b[static_cast<std::size_t>(i)];
            return y;
        };
        // the border block only: what a streamed solve is given
        auto border = [&] {
            auto A = strided(msz, st, nan);
            for (int r = 0; r < B; ++r)
                for (int c = 0; c < B; ++c) {
                    const auto e = static_cast<std::size_t>(Map::abb(Shape, r, c));
                    A[e * st]    = Ms[e];
                }
            return A;
        };

        // solve_inplace: stored, streamed; static, runtime
        auto A1 = full(), A2 = border(), A3 = full(), A4 = border();
        auto y1 = rhs(n), y2 = rhs(B), y3 = rhs(n), y4 = rhs(B);
        int r1 = 0, r2 = 0, r3 = 0, r4 = 0;
        const bool ok1 = Static::template solve_inplace<in, in, in>(A1.data(), st, ws.data(), st,
                                                                    y1.data(), st, r1);
        const bool ok2 = Static::template solve_inplace_streamed<in, in, in>(
            provider, A2.data(), st, ws.data(), st, y2.data(), st, r2);
        TDLS_CHECK(contract_held(log, NB, 0));
        const bool ok3 = Dynamic::solve_inplace(Shape, A3.data(), st, piv.data(), st, wd.data(), st,
                                                y3.data(), st, r3);
        const bool ok4 = Dynamic::solve_inplace_streamed(Shape, provider, A4.data(), st, piv.data(),
                                                         st, wds.data(), st, y4.data(), st, r4);
        TDLS_CHECK(contract_held(log, NB, 0));
        TDLS_CHECK(ok1 == ok2 && ok1 == ok3 && ok1 == ok4);
        TDLS_CHECK(r1 == r2 && r1 == r3 && r1 == r4);
        if (ok1) {
            TDLS_CHECK(same_strided(y1, y2, n, st));
            TDLS_CHECK(same_strided(y1, y3, n, st));
            TDLS_CHECK(same_strided(y1, y4, n, st));
            ++solved;
        }
        floors.refused += r1;

        // every border column of A^-1: stored, streamed; static, runtime
        auto C1 = full(), C2 = border(), C3 = full(), C4 = border();
        auto x1 = strided(B * B, st, nan), x2 = x1, x3 = x1, x4 = x1;
        int c1 = 0, c2 = 0, c3 = 0, c4 = 0;
        const bool k1 = Static::template solve_border_canonical_multirhs<B, in, in, in>(
            C1.data(), st, ws.data(), st, 0, x1.data(), st, B * st, c1);
        const bool k2 = Static::template solve_border_canonical_multirhs_streamed<B, in, in, in>(
            provider, C2.data(), st, ws.data(), st, 0, x2.data(), st, B * st, c2);
        TDLS_CHECK(contract_held(log, NB, 0));
        const bool k3 = Dynamic::solve_border_canonical_multirhs(
            Shape, B, C3.data(), st, piv.data(), st, wd.data(), st, 0, x3.data(), st, B * st, c3);
        const bool k4 = Dynamic::solve_border_canonical_multirhs_streamed(
            Shape, B, provider, C4.data(), st, piv.data(), st, wds.data(), st, 0, x4.data(), st,
            B * st, c4);
        TDLS_CHECK(contract_held(log, NB, 0));
        TDLS_CHECK(k1 == k2 && k1 == k3 && k1 == k4);
        TDLS_CHECK(c1 == c2 && c1 == c3 && c1 == c4);
        if (k1) {
            TDLS_CHECK(same_strided(x1, x2, B * B, st));
            TDLS_CHECK(same_strided(x1, x3, B * B, st));
            TDLS_CHECK(same_strided(x1, x4, B * B, st));
        }

        // the pass that converged: the tangent, refused blocks included
        auto T1 = border(), T2 = border();
        auto t1 = rhs(B), t2 = rhs(B);
        auto g1 = strided(B * B, st, nan), g2 = g1;
        int q1 = 0, q2 = 0;
        const SO o1 = Static::template solve_inplace_streamed_or_tangent<B, in, in, in>(
            provider, make_converged(true, NB, log), T1.data(), st, ws.data(), st, t1.data(), st, 0,
            g1.data(), st, B * st, q1);
        TDLS_CHECK(contract_held(log, NB, 1));
        const DO o2 = Dynamic::solve_inplace_streamed_or_tangent(
            Shape, B, provider, make_converged(true, NB, log), T2.data(), st, piv.data(), st,
            wds.data(), st, t2.data(), st, 0, g2.data(), st, B * st, q2);
        TDLS_CHECK(contract_held(log, NB, 1));
        TDLS_CHECK((o1 == SO::tangent) == k1 && (o2 == DO::tangent) == k1);
        TDLS_CHECK(q1 == c1 && q2 == c1);
        if (k1) {
            TDLS_CHECK(same_strided(x1, g1, B * B, st));
            TDLS_CHECK(same_strided(x1, g2, B * B, st));
            if (q1 > 0)
                ++floors.tangent_second;
            else
                ++floors.tangent_plain;
        }

        // the pass that did not converge: the Newton correction
        auto N1 = border(), N2 = border();
        auto z1 = rhs(B), z2 = rhs(B);
        int u1 = 0, u2 = 0;
        const SO m1 = Static::template solve_inplace_streamed_or_tangent<B, in, in, in>(
            provider, make_converged(false, NB, log), N1.data(), st, ws.data(), st, z1.data(), st,
            0, g1.data(), st, B * st, u1);
        TDLS_CHECK(contract_held(log, NB, 1));
        const DO m2 = Dynamic::solve_inplace_streamed_or_tangent(
            Shape, B, provider, make_converged(false, NB, log), N2.data(), st, piv.data(), st,
            wds.data(), st, z2.data(), st, 0, g2.data(), st, B * st, u2);
        TDLS_CHECK(contract_held(log, NB, 1));
        TDLS_CHECK((m1 == SO::solved) == ok1 && (m2 == DO::solved) == ok1);
        TDLS_CHECK(u1 == r1 && u2 == r1);
        if (ok1) {
            TDLS_CHECK(same_strided(y1, z1, n, st));
            TDLS_CHECK(same_strided(y1, z2, n, st));
        }
    }
    // Floors: the batch is invertible, its injected blocks are refused, and
    // the tangent comes out of a pass with refused blocks.
    TDLS_CHECK(solved == count);
    TDLS_CHECK(floors.refused >= 2 * (count / 3));
    TDLS_CHECK(floors.tangent_second > 0);
    TDLS_CHECK(floors.tangent_plain > 0);
}

/// \brief streamed_case for both residencies.
/// \tparam Shape  shape of the matrices
/// \tparam Config configuration
/// \param[in] seed generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void both_residencies(const std::uint64_t seed) {
    streamed_case<Shape, Config, false>(12, seed);
    streamed_case<Shape, Config, true>(12, seed + 50);
}

using tdls::ArrowheadShape;
using tdls::ArrowheadStorage;
constexpr ArrowheadShape polycrystal{12, 6, 10, 6};
constexpr ArrowheadShape uneven{9, 3, 8, 2};
constexpr ArrowheadShape uneven_dense{8, 6, 10, 0};
constexpr ArrowheadShape iwan{6, 1, 24, 0};
constexpr ArrowheadShape thin_border{3, 5, 7, 0};
constexpr ArrowheadShape big_block{6, 12, 4, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/streamed/storages") {
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0)>(600100);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Full, true, false, 0)>(600200);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Csr, true, true, 0)>(600300);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Bsr, true, false, 0)>(600400);
    both_residencies<uneven_dense, path_config(ArrowheadStorage::Csr, false, true, 0)>(600500);
    both_residencies<uneven_dense, path_config(ArrowheadStorage::Full, false, false, 0)>(600600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/streamed/couplings_and_layouts") {
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, true, 0)>(601100);
    both_residencies<uneven_dense, path_config(ArrowheadStorage::Packed, false, false, 0)>(601200);
    both_residencies<uneven_dense, path_config(ArrowheadStorage::Packed, false, true, 0)>(601300);
    both_residencies<uneven, path_config(ArrowheadStorage::Packed, true, false, 0)>(601400);
    both_residencies<uneven, path_config(ArrowheadStorage::Full, true, true, 0)>(601500);
    both_residencies<iwan, path_config(ArrowheadStorage::Full, false, false, 0)>(601600);
    both_residencies<iwan, path_config(ArrowheadStorage::Packed, false, true, 0)>(601700);
    both_residencies<thin_border, path_config(ArrowheadStorage::Packed, false, false, 0)>(601800);
    both_residencies<thin_border, path_config(ArrowheadStorage::Csr, false, true, 0)>(601900);
    both_residencies<big_block, path_config(ArrowheadStorage::Packed, false, false, 0)>(602000);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/streamed/passes") {
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 2)>(603100);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, true, 2)>(603200);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, true, 5)>(603300);
    both_residencies<uneven_dense, path_config(ArrowheadStorage::Full, false, false, 3)>(603400);
    both_residencies<uneven_dense, path_config(ArrowheadStorage::Packed, false, true, 4)>(603500);
    both_residencies<big_block, path_config(ArrowheadStorage::Packed, false, true, 2)>(603600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/streamed/gather_in_place_unroll") {
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0, true)>(
        604100);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 2, true)>(
        604200);
    both_residencies<polycrystal,
                     path_config(ArrowheadStorage::Packed, true, false, 0, false, true)>(604300);
    both_residencies<uneven_dense,
                     path_config(ArrowheadStorage::Full, false, true, 0, false, true)>(604400);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0, false,
                                              false, false)>(604500);
    both_residencies<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0, false,
                                              false, true, true)>(604600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/streamed/registers") {
    using S = ArrowheadStorage;
    both_residencies<polycrystal, registers(path_config(S::Packed, true, false, 0))>(605100);
    both_residencies<polycrystal, registers(path_config(S::Packed, true, false, 2))>(605200);
    both_residencies<polycrystal, registers(path_config(S::Csr, true, true, 0))>(605300);
    both_residencies<polycrystal, registers(path_config(S::Bsr, true, true, 2))>(605400);
    both_residencies<uneven_dense, registers(path_config(S::Full, false, false, 0))>(605500);
    both_residencies<uneven_dense, registers(path_config(S::Packed, false, false, 3))>(605600);
    both_residencies<uneven_dense, registers(path_config(S::Packed, false, true, 4))>(605700);
    both_residencies<uneven, registers(path_config(S::Packed, true, true, 0))>(605800);
    both_residencies<iwan, registers(path_config(S::Packed, false, true, 0))>(605900);
    both_residencies<thin_border, registers(path_config(S::Csr, false, true, 0))>(606000);
    both_residencies<big_block, registers(path_config(S::Packed, false, false, 0))>(606100);
    both_residencies<polycrystal, registers(path_config(S::Packed, true, false, 0, false, true))>(
        606200);
    both_residencies<polycrystal, registers(path_config(S::Packed, true, false, 0, false, false,
                                                        false))>(606300);
    both_residencies<polycrystal, registers(path_config(S::Packed, true, false, 0, false, false,
                                                        true, true))>(606400);
}

TDLS_TEST_MAIN
