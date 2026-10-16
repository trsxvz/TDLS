/// \file
/// \brief Anchor suite of the static ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// ArrowheadLUppSolverStatic::solve is compared against the independent
/// reference LU (tests/common/reference_lu.hpp), which factors the full
/// matrix with partial pivoting over whole columns. The grid of shapes
/// crosses the structural boundaries of the solver: blocks of one unknown
/// (the Iwan model of TFEL), a border smaller than the blocks, a border
/// that is not a multiple of the block, blocks of 12, blocks cut into
/// tiles of 3 and 1, the shape of a polycrystal, and up to 1452
/// unknowns. Both coupling patterns, both layouts and the three scalar
/// types are covered, in the Packed storage (the storages suite bridges
/// the others bitwise). Every batch mixes dominant and random blocks,
/// singular blocks and pivot-poor ones, which the pivot check refuses and
/// the second pass eliminates, and one structurally singular matrix. The
/// condensation MinTraffic, which differs from the default under the
/// row-major layout with the dense coupling only, is anchored there. The
/// verdict of every case is the normwise backward error of both solvers,
/// the exact agreement of the singularity verdicts, and a floor of
/// refused blocks, so that the second pass is never vacuous.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Configuration of one cell: Packed storage, a coupling, a layout.
/// \tparam T scalar type
/// \param[in] diag      diagonal coupling
/// \param[in] col       column-major layout
/// \param[in] registers blocks eliminated in registers (RowInterchange::Physical)
/// \param[in] traffic   condensation MinTraffic
/// \return the configuration
template<typename T>
constexpr tdls::ArrowheadLUppConfig<T> cell_config(const bool diag, const bool col,
                                                   const bool registers = false,
                                                   const bool traffic   = false) {
    tdls::ArrowheadLUppConfig<T> c;
    // tiles of 3, so that blocks of 4 are cut into tiles of 3 and 1
    c.block_tile_size = 3;
    c.coupling        = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.storage         = tdls::ArrowheadStorage::Packed;
    c.layout          = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.block_row_interchange =
        registers ? tdls::RowInterchange::Physical : tdls::RowInterchange::Logical;
    c.condensation =
        traffic ? tdls::ArrowheadCondensation::MinTraffic : tdls::ArrowheadCondensation::MinFlops;
    return c;
}

/// \brief Runs one anchor comparison: count systems of the shape, system
/// 0 structurally singular (a zero column of the full matrix), the
/// others with injected singular or pivot-poor blocks; identical
/// singularity verdicts, both backward errors under the tolerance, and
/// at least one refused block per injected system.
/// \tparam T       scalar type
/// \tparam Shape   shape of the matrices
/// \tparam diag    diagonal coupling (and pattern)
/// \tparam col     column-major layout
/// \tparam reg     blocks eliminated in registers (RowInterchange::Physical)
/// \tparam traffic condensation MinTraffic
/// \param[in] count     number of systems
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, tdls::ArrowheadShape Shape, bool diag, bool col, bool reg = false,
         bool traffic = false>
void anchor_case(const int count, const double tolerance, const std::uint64_t seed) {
    constexpr auto config = cell_config<T>(diag, col, reg, traffic);
    using Solver          = tdls::ArrowheadLUppSolverStatic<T, Shape, config>;
    constexpr int n       = Shape.n();

    bool verdicts_agree = true;
    int singular_arrow = 0, singular_reference = 0, injected = 0, refused_total = 0;
    double be_arrow = 0.0, be_reference = 0.0;
    std::vector<T> work(Solver::workspace_size), x(n), xr(n), Ar;
    std::vector<int> pivr(n);
    for (int s = 0; s < count; ++s) {
        auto p = tdls_tests::make_arrow_problem<T>(Shape, diag,
                                                   s % 2 == 0 ? tdls_tests::ArrowBlocks::Dominant
                                                              : tdls_tests::ArrowBlocks::Random,
                                                   seed + static_cast<std::uint64_t>(s));
        if (s == 0) {
            // A zero column of the full matrix: unknown 0 of block 0 in
            // its block and in the border rows.
            tdls_tests::zero_block_unknown(p, 0, 0);
            for (int r = 0; r < Shape.border; ++r)
                p.at(r, Shape.border) = T(0);
        } else if (s % 3 == 1) {
            tdls_tests::zero_block_unknown(p, s % Shape.blocks, s % Shape.block);
            ++injected;
        } else if (s % 3 == 2) {
            tdls_tests::shrink_block(p, (s + 1) % Shape.blocks);
            ++injected;
        }
        auto M        = tdls_tests::store_arrow<typename Solver::storage_map>(p);
        int refused   = 0;
        const bool ok = Solver::template solve<true, true, false>(M.data(), 1, work.data(), 1,
                                                                  p.b.data(), x.data(), 1, refused);
        Ar            = p.A;
        const bool okr =
            tdls_tests::reference_solve(Ar.data(), pivr.data(), p.b.data(), xr.data(), n);
        if (ok != okr) verdicts_agree = false;
        if (!ok) ++singular_arrow;
        if (!okr) ++singular_reference;
        refused_total += refused;
        if (ok && okr) {
            be_arrow =
                std::max(be_arrow, tdls_tests::backward_error(p.A.data(), x.data(), p.b.data(), n));
            be_reference = std::max(
                be_reference, tdls_tests::backward_error(p.A.data(), xr.data(), p.b.data(), n));
        }
    }
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(singular_arrow == 1);
    TDLS_CHECK(singular_reference == 1);
    TDLS_CHECK(refused_total >= injected);
    TDLS_CHECK_LE(be_arrow, tolerance);
    TDLS_CHECK_LE(be_reference, tolerance);
}

} // namespace

/// Emits the two anchor cases of one (type, shape) cell under the dense
/// coupling and pattern: row-major and column-major.
#define TDLS_ARROW_ANCHOR_DENSE(T, B, S, NB, COUNT, TOL, SEED)                                     \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/" #T "/B=" #B ",S=" #S ",NB=" #NB ",dense,row") {  \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, 0}, false, false>(COUNT, TOL, SEED);         \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/" #T "/B=" #B ",S=" #S ",NB=" #NB ",dense,col") {  \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, 0}, false, true>(COUNT, TOL, SEED + 1);      \
    }

/// Emits the two anchor cases of one (type, shape) cell under the
/// diagonal coupling and pattern: row-major and column-major.
#define TDLS_ARROW_ANCHOR_DIAGONAL(T, B, S, NB, B0, COUNT, TOL, SEED)                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/" #T "/B=" #B ",S=" #S ",NB=" #NB                  \
                   ",diagonal,row") {                                                              \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, B0}, true, false>(COUNT, TOL, SEED);         \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/" #T "/B=" #B ",S=" #S ",NB=" #NB                  \
                   ",diagonal,col") {                                                              \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, B0}, true, true>(COUNT, TOL, SEED + 1);      \
    }

// Double: the whole shape grid.
TDLS_ARROW_ANCHOR_DENSE(double, 1, 1, 12, 60, 1e-12, 300100) // the Iwan shape
TDLS_ARROW_ANCHOR_DIAGONAL(double, 1, 1, 12, 0, 60, 1e-12, 300110)
TDLS_ARROW_ANCHOR_DIAGONAL(double, 2, 1, 20, 1, 60, 1e-12, 300200) // one coupled row of two
TDLS_ARROW_ANCHOR_DENSE(double, 3, 6, 8, 60, 1e-12, 300300)        // border smaller than blocks
TDLS_ARROW_ANCHOR_DENSE(double, 8, 6, 10, 60, 1e-12, 300400)       // border not a multiple
TDLS_ARROW_ANCHOR_DIAGONAL(double, 8, 6, 10, 2, 60, 1e-12, 300410)
TDLS_ARROW_ANCHOR_DENSE(double, 6, 12, 5, 60, 1e-12, 300500) // blocks of 12
TDLS_ARROW_ANCHOR_DENSE(double, 12, 4, 9, 60, 1e-12, 300600) // tiles of 3 and 1
TDLS_ARROW_ANCHOR_DIAGONAL(double, 12, 4, 9, 4, 60, 1e-12, 300610)
TDLS_ARROW_ANCHOR_DENSE(double, 6, 6, 30, 30, 1e-12, 300700) // reduced polycrystal
TDLS_ARROW_ANCHOR_DIAGONAL(double, 6, 6, 30, 0, 30, 1e-12, 300710)
TDLS_ARROW_ANCHOR_DIAGONAL(double, 12, 6, 30, 6, 30, 1e-12, 300800) // full polycrystal
TDLS_ARROW_ANCHOR_DIAGONAL(double, 12, 6, 240, 6, 4, 1e-12, 300900) // 1452 unknowns

// The tolerances sit a few orders above the errors reached, which match
// those of the reference LU (about 1e-16 in double, 1e-8 in float; the
// metric accumulates in double, hence the floor of long double).
TDLS_ARROW_ANCHOR_DIAGONAL(float, 6, 6, 30, 0, 30, 1e-5, 310100)
TDLS_ARROW_ANCHOR_DENSE(float, 1, 1, 12, 60, 1e-5, 310200)
TDLS_ARROW_ANCHOR_DIAGONAL(long double, 12, 6, 30, 6, 30, 1e-12, 320100)
TDLS_ARROW_ANCHOR_DENSE(long double, 8, 6, 10, 60, 1e-12, 320200)

/// Emits the four anchor cases of one (type, shape) cell with the blocks
/// eliminated in registers: both couplings, both layouts. The seeds are
/// those of the default cells of the same shape where there is one, whose
/// floors of refused blocks hold.
#define TDLS_ARROW_ANCHOR_REGISTERS(T, B, S, NB, B0, COUNT, TOL, SEED_DENSE, SEED_DIAGONAL)        \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/registers/" #T "/B=" #B ",S=" #S ",NB=" #NB        \
                   ",dense,row") {                                                                 \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, 0}, false, false, true>(COUNT, TOL,          \
                                                                              SEED_DENSE);         \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/registers/" #T "/B=" #B ",S=" #S ",NB=" #NB        \
                   ",dense,col") {                                                                 \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, 0}, false, true, true>(COUNT, TOL,           \
                                                                             SEED_DENSE + 1);      \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/registers/" #T "/B=" #B ",S=" #S ",NB=" #NB        \
                   ",diagonal,row") {                                                              \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, B0}, true, false, true>(COUNT, TOL,          \
                                                                              SEED_DIAGONAL);      \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/registers/" #T "/B=" #B ",S=" #S ",NB=" #NB        \
                   ",diagonal,col") {                                                              \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, B0}, true, true, true>(COUNT, TOL,           \
                                                                             SEED_DIAGONAL + 1);   \
    }

// The blocks eliminated in registers: blocks of one unknown, a border not
// a multiple of the block, the polycrystal, in double and float.
TDLS_ARROW_ANCHOR_REGISTERS(double, 1, 1, 12, 0, 60, 1e-12, 300100, 300110)
TDLS_ARROW_ANCHOR_REGISTERS(double, 8, 6, 10, 2, 60, 1e-12, 300400, 300410)
TDLS_ARROW_ANCHOR_REGISTERS(double, 12, 6, 30, 6, 30, 1e-12, 300800, 300800)
TDLS_ARROW_ANCHOR_REGISTERS(float, 6, 6, 30, 0, 30, 1e-5, 310100, 310100)

/// Emits the two anchor cases of one (type, shape) cell under the
/// condensation MinTraffic, the row-major layout and the dense coupling:
/// the blocks eliminated by tiles and in registers. The seeds are those of
/// the default cells of the same shape.
#define TDLS_ARROW_ANCHOR_MIN_TRAFFIC(T, B, S, NB, COUNT, TOL, SEED)                               \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/min_traffic/" #T "/B=" #B ",S=" #S ",NB=" #NB      \
                   ",tiles") {                                                                     \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, 0}, false, false, false, true>(COUNT, TOL,   \
                                                                                     SEED);        \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/static/min_traffic/" #T "/B=" #B ",S=" #S ",NB=" #NB      \
                   ",registers") {                                                                 \
        anchor_case<T, tdls::ArrowheadShape{B, S, NB, 0}, false, false, true, true>(COUNT, TOL,    \
                                                                                    SEED);         \
    }

// MinTraffic: the dense cells of the grid.
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(double, 1, 1, 12, 60, 1e-12, 300100) // the Iwan shape
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(double, 3, 6, 8, 60, 1e-12, 300300)
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(double, 8, 6, 10, 60, 1e-12, 300400)
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(double, 6, 12, 5, 60, 1e-12, 300500)
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(double, 12, 4, 9, 60, 1e-12, 300600)
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(double, 6, 6, 30, 30, 1e-12, 300700)
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(float, 1, 1, 12, 60, 1e-5, 310200)
TDLS_ARROW_ANCHOR_MIN_TRAFFIC(long double, 8, 6, 10, 60, 1e-12, 320200)

TDLS_TEST_MAIN
