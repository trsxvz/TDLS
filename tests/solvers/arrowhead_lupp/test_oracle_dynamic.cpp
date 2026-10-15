/// \file
/// \brief Anchor suite of the runtime ArrowheadLUpp solver.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// ArrowheadLUppSolverDynamic::solve, the shape a function argument, is
/// compared against the independent
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
/// \param[in] diag   diagonal coupling
/// \param[in] col    column-major layout
/// \return the configuration
template<typename T>
constexpr tdls::ArrowheadLUppConfig<T> cell_config(const bool diag, const bool col) {
    tdls::ArrowheadLUppConfig<T> c;
    // tiles of 3, so that blocks of 4 are cut into tiles of 3 and 1
    c.block_tile_size       = 3;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.storage  = tdls::ArrowheadStorage::Packed;
    c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    return c;
}

/// \brief Runs one anchor comparison: count systems of the shape, system
/// 0 structurally singular (a zero column of the full matrix), the
/// others with injected singular or pivot-poor blocks; identical
/// singularity verdicts, both backward errors under the tolerance, and
/// at least one refused block per injected system.
/// \tparam T    scalar type
/// \tparam diag diagonal coupling (and pattern)
/// \tparam col  column-major layout
/// \param[in] shape     shape of the matrices
/// \param[in] count     number of systems
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, bool diag, bool col>
void anchor_case(const tdls::ArrowheadShape shape, const int count, const double tolerance,
                 const std::uint64_t seed) {
    constexpr auto config = cell_config<T>(diag, col);
    using Solver          = tdls::ArrowheadLUppSolverDynamic<T, config>;
    const int n           = shape.n();
    TDLS_CHECK(Solver::storage_map::valid(shape));

    bool verdicts_agree = true;
    int singular_arrow = 0, singular_reference = 0, injected = 0, refused_total = 0;
    double be_arrow = 0.0, be_reference = 0.0;
    std::vector<T> work(static_cast<std::size_t>(Solver::workspace_size(shape))), x(n), xr(n), Ar;
    std::vector<int> piv(static_cast<std::size_t>(Solver::pivot_size(shape))), pivr(n);
    for (int s = 0; s < count; ++s) {
        auto p = tdls_tests::make_arrow_problem<T>(shape, diag,
                                                   s % 2 == 0 ? tdls_tests::ArrowBlocks::Dominant
                                                              : tdls_tests::ArrowBlocks::Random,
                                                   seed + static_cast<std::uint64_t>(s));
        if (s == 0) {
            tdls_tests::zero_block_unknown(p, 0, 0);
            for (int r = 0; r < shape.border; ++r)
                p.at(r, shape.border) = T(0);
        } else if (s % 3 == 1) {
            tdls_tests::zero_block_unknown(p, s % shape.blocks, s % shape.block);
            ++injected;
        } else if (s % 3 == 2) {
            tdls_tests::shrink_block(p, (s + 1) % shape.blocks);
            ++injected;
        }
        auto M        = tdls_tests::store_arrow<typename Solver::storage_map>(p);
        int refused   = 0;
        const bool ok = Solver::solve(shape, M.data(), 1, piv.data(), 1, work.data(), 1, p.b.data(),
                                      x.data(), 1, refused);
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
    TDLS_TEST_CASE("arrowheadlupp/oracle/dynamic/" #T "/B=" #B ",S=" #S ",NB=" #NB ",dense,row") { \
        anchor_case<T, false, false>(tdls::ArrowheadShape{B, S, NB, 0}, COUNT, TOL, SEED);         \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/dynamic/" #T "/B=" #B ",S=" #S ",NB=" #NB ",dense,col") { \
        anchor_case<T, false, true>(tdls::ArrowheadShape{B, S, NB, 0}, COUNT, TOL, SEED + 1);      \
    }

/// Emits the two anchor cases of one (type, shape) cell under the
/// diagonal coupling and pattern: row-major and column-major.
#define TDLS_ARROW_ANCHOR_DIAGONAL(T, B, S, NB, B0, COUNT, TOL, SEED)                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/dynamic/" #T "/B=" #B ",S=" #S ",NB=" #NB                 \
                   ",diagonal,row") {                                                              \
        anchor_case<T, true, false>(tdls::ArrowheadShape{B, S, NB, B0}, COUNT, TOL, SEED);         \
    }                                                                                              \
    TDLS_TEST_CASE("arrowheadlupp/oracle/dynamic/" #T "/B=" #B ",S=" #S ",NB=" #NB                 \
                   ",diagonal,col") {                                                              \
        anchor_case<T, true, true>(tdls::ArrowheadShape{B, S, NB, B0}, COUNT, TOL, SEED + 1);      \
    }

// Double: the whole shape grid.
TDLS_ARROW_ANCHOR_DENSE(double, 1, 1, 12, 60, 1e-12, 400100) // the Iwan shape
TDLS_ARROW_ANCHOR_DIAGONAL(double, 1, 1, 12, 0, 60, 1e-12, 400110)
TDLS_ARROW_ANCHOR_DIAGONAL(double, 2, 1, 20, 1, 60, 1e-12, 400200) // one coupled row of two
TDLS_ARROW_ANCHOR_DENSE(double, 3, 6, 8, 60, 1e-12, 400300)        // border smaller than blocks
TDLS_ARROW_ANCHOR_DENSE(double, 8, 6, 10, 60, 1e-12, 400400)       // border not a multiple
TDLS_ARROW_ANCHOR_DIAGONAL(double, 8, 6, 10, 2, 60, 1e-12, 400410)
TDLS_ARROW_ANCHOR_DENSE(double, 6, 12, 5, 60, 1e-12, 400500) // blocks of 12
TDLS_ARROW_ANCHOR_DENSE(double, 12, 4, 9, 60, 1e-12, 400600) // tiles of 3 and 1
TDLS_ARROW_ANCHOR_DIAGONAL(double, 12, 4, 9, 4, 60, 1e-12, 400610)
TDLS_ARROW_ANCHOR_DENSE(double, 6, 6, 30, 30, 1e-12, 400700) // reduced polycrystal
TDLS_ARROW_ANCHOR_DIAGONAL(double, 6, 6, 30, 0, 30, 1e-12, 400710)
TDLS_ARROW_ANCHOR_DIAGONAL(double, 12, 6, 30, 6, 30, 1e-12, 400800) // full polycrystal
TDLS_ARROW_ANCHOR_DIAGONAL(double, 12, 6, 240, 6, 4, 1e-12, 400900) // 1452 unknowns

// The tolerances sit a few orders above the errors reached, which match
// those of the reference LU (about 1e-16 in double, 1e-8 in float; the
// metric accumulates in double, hence the floor of long double).
TDLS_ARROW_ANCHOR_DIAGONAL(float, 6, 6, 30, 0, 30, 1e-5, 410100)
TDLS_ARROW_ANCHOR_DENSE(float, 1, 1, 12, 60, 1e-5, 410200)
TDLS_ARROW_ANCHOR_DIAGONAL(long double, 12, 6, 30, 6, 30, 1e-12, 420100)
TDLS_ARROW_ANCHOR_DENSE(long double, 8, 6, 10, 60, 1e-12, 420200)

TDLS_TEST_MAIN
