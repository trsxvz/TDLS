/// \file
/// \brief Bridge suite: the recomputed entry point of the compile-time
/// ArrowheadLUpp solver against the streamed one, and the contract of the
/// provider.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// solve_inplace_recomputed asks the provider for every block twice, once
/// while it condenses it and once while it recovers its unknowns, and
/// stores only the refused blocks, at most `capacity` of them. On the
/// standard batch (dominant and random blocks, singular and pivot-poor
/// ones, so that blocks are refused) and one structurally singular matrix,
/// for both eliminations of the blocks (TiledLUpp and register_blocks):
///   - with a store of every block: solved exactly when the streamed solve
///     succeeds, the same refused blocks, the border rows of the solution
///     bit for bit, the rows of the blocks within rounding errors;
///   - with a store of none, and of one when two blocks are refused:
///     deferred, y untouched, the condensation stopped at the refused
///     block beyond the capacity;
///   - singular: y untouched;
///   - the provider: the condensation asks for the blocks in order, the
///     recovery asks for every accepted block once, the last one first,
///     with blk.recovery set, and never for a refused one.
/// The rows of the blocks of y start as NaN, so that a read of them would
/// show: the recomputed solve takes y_k from the provider. Built with
/// -ffp-contract=off where the compiler knows the flag
/// (tests/solvers/CMakeLists.txt), as the other bridge suites.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"

namespace {

/// \brief A configuration from its paths.
/// \param[in] storage   storage of the streamed reference (the recomputed
///                      solve stores no matrix)
/// \param[in] diag      diagonal coupling
/// \param[in] col       column-major layout
/// \param[in] pass      rhs_pass_width
/// \param[in] registers register_blocks
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> path_config(const tdls::ArrowheadStorage storage,
                                                        const bool diag, const bool col,
                                                        const int pass, const bool registers) {
    tdls::ArrowheadLUppConfig<double> c;
    c.coupling        = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.storage         = storage;
    c.layout          = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.rhs_pass_width  = pass;
    c.register_blocks = registers;
    return c;
}

/// \brief Calls of the provider during one solve.
struct CallLog {
    int next     = 0;           ///< block expected at the next call of the condensation
    bool ordered = true;        ///< the condensation asks in order, before any recovery
    std::vector<int> recovered; ///< the blocks asked for in the recovery, in order
};

/// \brief A provider that fills block k from the full matrix of a problem,
/// logging its calls.
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
        if (blk.recovery) {
            log.recovered.push_back(k);
        } else {
            log.ordered = log.ordered && k == log.next && log.recovered.empty();
            ++log.next;
        }
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

/// \brief Whether two arrays hold the same bits (NaN included).
/// \param[in] a first array
/// \param[in] b second array
/// \return true when every entry has the same bits
bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

/// \brief Counts of the batch, so that no branch is vacuous.
struct Floors {
    int solved_refused = 0; ///< solved with refused blocks in the store
    int deferred       = 0; ///< deferred
    int singular       = 0; ///< singular
};

/// \brief Runs the comparisons on a batch.
/// \tparam Shape  shape of the matrices
/// \tparam Config configuration
/// \tparam in     caller-local operands (stride 1) or strided ones
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config, bool in>
void recomputed_case(const int count, const std::uint64_t seed) {
    using Static        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Map           = typename Static::storage_map;
    using RO            = typename Static::RecomputedOutcome;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    constexpr bool col  = Config.layout == tdls::MatrixLayout::ColMajor;
    constexpr int B = Shape.border, NB = Shape.blocks, n = Shape.n();
    constexpr int st = in ? 1 : 3;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto strided     = [&](const int size) {
        return std::vector<double>(static_cast<std::size_t>(size * st), nan);
    };
    std::vector<double> ws   = strided(Static::workspace_size),
                        wall = strided(Static::template recomputed_workspace_size<NB> + 1),
                        wone = strided(Static::template recomputed_workspace_size<1> + 1);
    Floors floors;
    for (int s = 0; s < count; ++s) {
        auto p = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        if (s == 0) {
            // A zero column of the full matrix: structurally singular.
            tdls_tests::zero_block_unknown(p, 0, 0);
            for (int r = 0; r < B; ++r)
                p.at(r, B) = 0.0;
        }
        const auto Ms = tdls_tests::store_arrow<Map>(p);
        // the border rows of y, the rows of the blocks NaN
        auto rhs = [&] {
            auto y = strided(n);
            for (int i = 0; i < B; ++i)
                y[static_cast<std::size_t>(i * st)] = p.b[static_cast<std::size_t>(i)];
            return y;
        };
        // the reference: the streamed solve on the border block only
        auto A0 = strided(Map::size(Shape));
        for (int r = 0; r < B; ++r)
            for (int c = 0; c < B; ++c) {
                const auto e = static_cast<std::size_t>(Map::abb(Shape, r, c));
                A0[e * st]   = Ms[e];
            }
        auto y0 = rhs();
        int r0  = 0;
        CallLog log0;
        const bool ok0 = Static::template solve_inplace_streamed<in, in, in>(
            make_provider<diag>(p, log0), A0.data(), st, ws.data(), st, y0.data(), st, r0);
        // the border block in the layout of the configuration
        auto Abb = strided(B * B);
        for (int r = 0; r < B; ++r)
            for (int c = 0; c < B; ++c)
                Abb[static_cast<std::size_t>((col ? c * B + r : r * B + c) * st)] = p.at(r, c);

        // a store of every block
        auto y1         = rhs();
        const auto y1in = y1;
        int r1          = 0;
        CallLog log1;
        const RO o1 = Static::template solve_inplace_recomputed<NB, in, in, in>(
            make_provider<diag>(p, log1), Abb.data(), st, wall.data(), st, y1.data(), st, r1);
        TDLS_CHECK(r1 == r0);
        TDLS_CHECK((o1 == RO::solved) == ok0);
        TDLS_CHECK((o1 == RO::singular) == !ok0);
        TDLS_CHECK(log1.ordered && log1.next == NB);
        if (o1 == RO::solved) {
            for (int i = 0; i < B; ++i)
                TDLS_CHECK(std::memcmp(&y1[static_cast<std::size_t>(i * st)],
                                       &y0[static_cast<std::size_t>(i * st)], sizeof(double)) == 0);
            double scale = 0.0, diff = 0.0;
            for (int i = 0; i < n; ++i) {
                scale = std::max(scale, std::fabs(y0[static_cast<std::size_t>(i * st)]));
                diff  = std::max(diff, std::fabs(y1[static_cast<std::size_t>(i * st)] -
                                                 y0[static_cast<std::size_t>(i * st)]));
            }
            TDLS_CHECK(diff <= 1e-12 * scale);
            // every accepted block once, the last one first: the recovered
            // list has NB - r1 entries, strictly decreasing
            TDLS_CHECK(static_cast<int>(log1.recovered.size()) == NB - r1);
            TDLS_CHECK(std::is_sorted(log1.recovered.rbegin(), log1.recovered.rend()));
            TDLS_CHECK(std::adjacent_find(log1.recovered.begin(), log1.recovered.end()) ==
                       log1.recovered.end());
            if (r1 > 0) ++floors.solved_refused;
        } else {
            TDLS_CHECK(same_bits(y1, y1in));
            TDLS_CHECK(log1.recovered.empty());
            ++floors.singular;
        }

        // a store of none
        auto y2         = rhs();
        const auto y2in = y2;
        int r2          = 0;
        CallLog log2;
        const RO o2 = Static::template solve_inplace_recomputed<0, in, in, in>(
            make_provider<diag>(p, log2), Abb.data(), st, nullptr, 0, y2.data(), st, r2);
        if (r0 == 0) {
            TDLS_CHECK(o2 == o1);
            TDLS_CHECK(r2 == 0);
            if (o2 == RO::solved) TDLS_CHECK(same_bits(y2, y1));
        } else {
            TDLS_CHECK(o2 == RO::deferred);
            TDLS_CHECK(r2 == 1);
            TDLS_CHECK(same_bits(y2, y2in));
            TDLS_CHECK(log2.ordered && log2.next <= NB && log2.recovered.empty());
            ++floors.deferred;
        }

        // a store of one
        if (r0 >= 2) {
            auto y3         = rhs();
            const auto y3in = y3;
            int r3          = 0;
            CallLog log3;
            const RO o3 = Static::template solve_inplace_recomputed<1, in, in, in>(
                make_provider<diag>(p, log3), Abb.data(), st, wone.data(), st, y3.data(), st, r3);
            TDLS_CHECK(o3 == RO::deferred);
            TDLS_CHECK(r3 == 2);
            TDLS_CHECK(same_bits(y3, y3in));
            TDLS_CHECK(log3.recovered.empty());
        }
    }
    TDLS_CHECK(floors.solved_refused > 0);
    TDLS_CHECK(floors.deferred > 0);
    TDLS_CHECK(floors.singular > 0);
}

/// \brief Both residencies of a path.
/// \tparam Shape  shape of the matrices
/// \tparam Config configuration
/// \param[in] seed generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void both_residencies(const std::uint64_t seed) {
    recomputed_case<Shape, Config, true>(15, seed);
    recomputed_case<Shape, Config, false>(15, seed + 50);
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

TDLS_TEST_CASE("arrowheadlupp/bridge/recomputed/tiled") {
    using S = ArrowheadStorage;
    both_residencies<polycrystal, path_config(S::Packed, true, false, 0, false)>(700100);
    both_residencies<polycrystal, path_config(S::Packed, true, true, 0, false)>(700200);
    both_residencies<polycrystal, path_config(S::Packed, true, false, 2, false)>(700300);
    both_residencies<uneven_dense, path_config(S::Full, false, false, 0, false)>(700400);
    both_residencies<uneven_dense, path_config(S::Packed, false, true, 3, false)>(700500);
    both_residencies<uneven, path_config(S::Csr, true, true, 0, false)>(700600);
    both_residencies<iwan, path_config(S::Packed, false, false, 0, false)>(700700);
    both_residencies<thin_border, path_config(S::Packed, false, true, 0, false)>(700800);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/recomputed/registers") {
    using S = ArrowheadStorage;
    both_residencies<polycrystal, path_config(S::Packed, true, false, 0, true)>(701100);
    both_residencies<polycrystal, path_config(S::Packed, true, true, 0, true)>(701200);
    both_residencies<polycrystal, path_config(S::Packed, true, false, 2, true)>(701300);
    both_residencies<uneven_dense, path_config(S::Full, false, false, 0, true)>(701400);
    both_residencies<uneven_dense, path_config(S::Packed, false, true, 3, true)>(701500);
    both_residencies<uneven, path_config(S::Csr, true, true, 0, true)>(701600);
    both_residencies<iwan, path_config(S::Packed, false, false, 0, true)>(701700);
    both_residencies<thin_border, path_config(S::Packed, false, true, 0, true)>(701800);
    both_residencies<big_block, path_config(S::Packed, false, false, 0, true)>(701900);
}

TDLS_TEST_MAIN
