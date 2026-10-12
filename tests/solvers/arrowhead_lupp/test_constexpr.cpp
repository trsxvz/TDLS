/// \file
/// \brief Compile-time certificate suite of the ArrowheadLUpp solvers.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every static_assert below constant-evaluates a whole solve, and the
/// standard requires constant evaluation to reject undefined behaviour:
/// an out-of-bounds index, a read of an uninitialized value or a signed
/// overflow fails the compilation. Each assertion that compiles is
/// therefore a machine-checked certificate that the exercised path is
/// free of undefined behaviour. The cells cross the paths of the solver:
/// both couplings and layouts, the four storages, passes, the gather of
/// the right-hand sides, the Schur complement in place, the unroll
/// policy off, blocks of one unknown, and the second pass (a block made
/// singular, one made pivot-poor), whose solutions are certified against
/// the residual. Each cell runs both solvers and certifies their bitwise
/// agreement, itself evaluated at compile time. The certificates are also
/// re-run at run time, so the suite reports like any other.

#include <tdls/tdls.hpp>

#include "harness.hpp"

namespace {

using tdls::ArrowheadCoupling;
using tdls::ArrowheadShape;
using tdls::ArrowheadStorage;

/// \brief Deterministic congruential step mapped to [-0.5, 0.5).
/// \param[in,out] state generator state
/// \return the next value of the stream
constexpr double lcg(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<double>(state >> 16) / 65536.0 - 0.5;
}

/// \brief Outcome of one compile-time solve.
struct Certificate {
    bool ok         = false; ///< both solvers succeeded
    bool same       = false; ///< bitwise agreement of the two solvers
    int refused     = 0;     ///< refused blocks
    double residual = 0.0;   ///< max |A x - b| over the full matrix
};

/// \brief Builds a system of the shape, makes block 0 singular and block 1
/// pivot-poor when asked, solves it with both solvers, and measures the
/// residual on the full matrix.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in] refuse whether blocks 0 and 1 are made to fail the check
/// \return the certificate
template<ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
constexpr Certificate certify(const bool refuse) {
    using Static    = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Dynamic   = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    using Map       = typename Static::storage_map;
    constexpr int B = Shape.border, S = Shape.block, NB = Shape.blocks, n = Shape.n();
    constexpr bool diag = Config.coupling == ArrowheadCoupling::Diagonal;
    double F[n * n]     = {};
    double b[n]         = {};
    unsigned state      = 12345u;
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            F[r * n + c] = lcg(state) + (r == c ? 2.0 + S * NB : 0.0);
    for (int k = 0; k < NB; ++k)
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            if (diag)
                F[(Shape.coupling_row + i) * n + row] = lcg(state);
            else
                for (int r = 0; r < B; ++r)
                    F[r * n + row] = lcg(state);
            for (int c = 0; c < B; ++c)
                F[row * n + c] = lcg(state);
            for (int j = 0; j < S; ++j)
                F[row * n + B + k * S + j] = lcg(state) * (refuse && k == 1 ? 1e-4 : 1.0) +
                                             (i == j && !(refuse && k == 1) ? S + 1.0 : 0.0);
        }
    if (refuse)
        for (int t = 0; t < S; ++t) {
            F[(B + 0) * n + B + t] = 0.0;
            F[(B + t) * n + B + 0] = 0.0;
        }
    for (int i = 0; i < n; ++i)
        b[i] = lcg(state);
    // The storage, through the map.
    double As[Map::size(Shape)] = {};
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            As[Map::abb(Shape, r, c)] = F[r * n + c];
    for (int k = 0; k < NB; ++k)
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            if constexpr (diag)
                As[Map::abk_diag(Shape, k, i)] = F[(Shape.coupling_row + i) * n + row];
            else
                for (int r = 0; r < B; ++r)
                    As[Map::abk(Shape, k, r, i)] = F[r * n + row];
            for (int c = 0; c < B; ++c)
                As[Map::akb(Shape, k, i, c)] = F[row * n + c];
            for (int j = 0; j < S; ++j)
                As[Map::d(Shape, k, i, j)] = F[row * n + B + k * S + j];
        }
    double Ad[Map::size(Shape)] = {};
    for (int e = 0; e < Map::size(Shape); ++e)
        Ad[e] = As[e];
    double xs[n] = {}, xd[n] = {};
    double ws[Static::workspace_size]         = {};
    double wd[Dynamic::workspace_size(Shape)] = {};
    int piv[Dynamic::pivot_size(Shape)]       = {};
    Certificate cert;
    int rd        = 0;
    const bool os = Static::template solve<true, true, true>(As, 1, ws, 1, b, xs, 1, cert.refused);
    const bool od = Dynamic::solve(Shape, Ad, 1, piv, 1, wd, 1, b, xd, 1, rd);
    cert.ok       = os && od;
    cert.same     = rd == cert.refused;
    for (int i = 0; i < n; ++i)
        cert.same = cert.same && xs[i] == xd[i];
    for (int r = 0; r < n; ++r) {
        double acc = -b[r];
        for (int c = 0; c < n; ++c)
            acc += F[r * n + c] * xs[c];
        cert.residual = acc > cert.residual ? acc : (-acc > cert.residual ? -acc : cert.residual);
    }
    return cert;
}

/// \brief A configuration from its paths.
/// \param[in] storage  storage
/// \param[in] diag     diagonal coupling
/// \param[in] col      column-major layout
/// \param[in] pass     rhs_pass_width
/// \param[in] gather   gather_rhs
/// \param[in] in_place schur_in_place
/// \param[in] unroll   unroll_loops
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double>
cell(const ArrowheadStorage storage, const bool diag, const bool col, const int pass = 0,
     const bool gather = false, const bool in_place = false, const bool unroll = true) {
    tdls::ArrowheadLUppConfig<double> c;
    c.coupling       = diag ? ArrowheadCoupling::Diagonal : ArrowheadCoupling::Dense;
    c.storage        = storage;
    c.layout         = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.rhs_pass_width = pass;
    c.gather_rhs     = gather;
    c.schur_in_place = in_place;
    c.unroll_loops   = unroll;
    return c;
}

/// \brief Whether a certificate holds: both solvers succeed, agree bit
/// for bit, refuse the expected blocks and leave a small residual.
/// \param[in] c       certificate
/// \param[in] refused expected refused blocks
/// \return true when it holds
constexpr bool holds(const Certificate& c, const int refused) {
    return c.ok && c.same && c.refused == refused && c.residual < 1e-12;
}

constexpr ArrowheadShape small_diag{4, 2, 3, 1};
constexpr ArrowheadShape small_dense{3, 2, 3, 0};
constexpr ArrowheadShape unit_blocks{2, 1, 5, 0};
constexpr ArrowheadShape bsr_shape{4, 2, 3, 2};

// The second pass: block 0 singular and block 1 pivot-poor, refused.
static_assert(holds(certify<small_diag, cell(ArrowheadStorage::Packed, true, false)>(true), 2));
static_assert(holds(certify<small_diag, cell(ArrowheadStorage::Packed, true, true)>(true), 2));
static_assert(holds(certify<small_dense, cell(ArrowheadStorage::Packed, false, false)>(true), 2));
static_assert(holds(certify<small_dense, cell(ArrowheadStorage::Packed, false, true)>(true), 2));
static_assert(holds(certify<unit_blocks, cell(ArrowheadStorage::Full, false, false)>(true), 2));
// The first pass alone, on every storage and path.
static_assert(holds(certify<small_diag, cell(ArrowheadStorage::Full, true, true)>(false), 0));
static_assert(holds(certify<small_dense, cell(ArrowheadStorage::Csr, false, false)>(false), 0));
static_assert(holds(certify<small_dense, cell(ArrowheadStorage::Csr, false, true)>(false), 0));
static_assert(holds(certify<bsr_shape, cell(ArrowheadStorage::Bsr, true, false)>(false), 0));
static_assert(holds(certify<bsr_shape, cell(ArrowheadStorage::Bsr, true, true)>(false), 0));
static_assert(holds(certify<small_diag, cell(ArrowheadStorage::Packed, true, false, 2)>(true), 2));
static_assert(holds(certify<small_diag, cell(ArrowheadStorage::Packed, true, true, 2, true)>(true),
                    2));
static_assert(holds(
    certify<small_dense, cell(ArrowheadStorage::Packed, false, false, 0, false, true)>(true), 2));
static_assert(holds(
    certify<small_diag, cell(ArrowheadStorage::Packed, true, false, 0, false, false, false)>(true),
    2));

} // namespace

TDLS_TEST_CASE("arrowheadlupp/constexpr/second_pass") {
    TDLS_CHECK(holds(certify<small_diag, cell(ArrowheadStorage::Packed, true, false)>(true), 2));
    TDLS_CHECK(holds(certify<small_dense, cell(ArrowheadStorage::Packed, false, true)>(true), 2));
    TDLS_CHECK(holds(certify<unit_blocks, cell(ArrowheadStorage::Full, false, false)>(true), 2));
}

TDLS_TEST_CASE("arrowheadlupp/constexpr/storages_and_paths") {
    TDLS_CHECK(holds(certify<bsr_shape, cell(ArrowheadStorage::Bsr, true, true)>(false), 0));
    TDLS_CHECK(holds(certify<small_dense, cell(ArrowheadStorage::Csr, false, false)>(false), 0));
    TDLS_CHECK(
        holds(certify<small_diag, cell(ArrowheadStorage::Packed, true, true, 2, true)>(true), 2));
}

TDLS_TEST_MAIN
