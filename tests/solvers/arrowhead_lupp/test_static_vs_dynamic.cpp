/// \file
/// \brief Bridge suite: the static and runtime ArrowheadLUpp solvers are
/// bitwise identical.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// At equal shape and configuration, the two solvers run the same
/// arithmetic: the runtime one keeps in its workspace what the
/// compile-time one keeps in local arrays, and calls the runtime
/// TiledLUpp solver where the other calls the compile-time one, itself
/// bitwise identical. On the standard batch (dominant and random blocks,
/// singular and pivot-poor ones, so that the second pass runs), the
/// suite compares the verdicts, the counts of refused blocks, the
/// solutions and the matrices left in place, bit for bit, on every path
/// of the condensation: the four storages, both couplings and layouts,
/// passes of several widths (the forward-stage check of the column-major
/// layout in one pass, the separate check otherwise), the gather of the
/// right-hand sides, the Schur complement accumulated in place, and the
/// unroll policy off.

#include <cstdint>
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
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double>
path_config(const tdls::ArrowheadStorage storage, const bool diag, const bool col, const int pass,
            const bool gather = false, const bool in_place = false, const bool unroll = true) {
    tdls::ArrowheadLUppConfig<double> c;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling       = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.storage        = storage;
    c.layout         = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.rhs_pass_width = pass;
    c.gather_rhs     = gather;
    c.schur_in_place = in_place;
    c.unroll_loops   = unroll;
    return c;
}

/// \brief The configuration c with its blocks eliminated in registers.
/// \param[in] c configuration
/// \return c with the blocks eliminated in registers
constexpr tdls::ArrowheadLUppConfig<double> registers(tdls::ArrowheadLUppConfig<double> c) {
    c.block_row_interchange = tdls::RowInterchange::Physical;
    return c;
}

/// \brief The configuration c with the defaults that choose: the blocks
/// eliminated as RowInterchange::Auto decides, the passes as
/// rhs_pass_width -1 decides.
/// \param[in] c configuration
/// \return c with block_row_interchange Auto and rhs_pass_width -1
constexpr tdls::ArrowheadLUppConfig<double> automatic(tdls::ArrowheadLUppConfig<double> c) {
    c.block_row_interchange = tdls::RowInterchange::Auto;
    c.rhs_pass_width        = -1;
    return c;
}

/// \brief Compares two compile-time configurations that must run the same
/// operations on a standard batch: the same verdicts, refused blocks,
/// solutions and recovery data, bit for bit.
/// \tparam Shape shape of the matrices
/// \tparam A     first configuration
/// \tparam B     second configuration
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> A,
         tdls::ArrowheadLUppConfig<double> B>
void same_operations(const int count, const std::uint64_t seed) {
    using SA            = tdls::ArrowheadLUppSolverStatic<double, Shape, A>;
    using SB            = tdls::ArrowheadLUppSolverStatic<double, Shape, B>;
    constexpr bool diag = A.coupling == tdls::ArrowheadCoupling::Diagonal;
    std::vector<double> wa(SA::workspace_size), wb(SB::workspace_size);
    for (int s = 0; s < count; ++s) {
        const auto p = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        auto Ma      = tdls_tests::store_arrow<typename SA::storage_map>(p);
        auto Mb      = Ma;
        auto ya = p.b, yb = p.b;
        int ra = 0, rb = 0;
        const bool oka = SA::template solve_inplace<true, true, true>(Ma.data(), 1, wa.data(), 1,
                                                                      ya.data(), 1, ra);
        const bool okb = SB::template solve_inplace<true, true, true>(Mb.data(), 1, wb.data(), 1,
                                                                      yb.data(), 1, rb);
        TDLS_CHECK(oka == okb);
        TDLS_CHECK(ra == rb);
        TDLS_CHECK(tdls_tests::same_bits(ya, yb));
        TDLS_CHECK(tdls_tests::same_bits(Ma, Mb));
    }
}

/// \brief Compares the two solvers on a standard batch.
/// \tparam Shape  shape of the matrices
/// \tparam Config configuration
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void bridge_case(const int count, const std::uint64_t seed) {
    using Static        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Dynamic       = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    std::vector<double> ws(Static::workspace_size),
        wd(static_cast<std::size_t>(Dynamic::workspace_size(Shape)));
    std::vector<int> piv(static_cast<std::size_t>(Dynamic::pivot_size(Shape)));
    int refused_total = 0, solved = 0;
    for (int s = 0; s < count; ++s) {
        const auto p = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        auto Ms      = tdls_tests::store_arrow<typename Static::storage_map>(p);
        auto Md      = Ms;
        auto ys = p.b, yd = p.b;
        int rs = 0, rd = 0;
        const bool oks = Static::template solve_inplace<true, true, true>(Ms.data(), 1, ws.data(),
                                                                          1, ys.data(), 1, rs);
        const bool okd = Dynamic::solve_inplace(Shape, Md.data(), 1, piv.data(), 1, wd.data(), 1,
                                                yd.data(), 1, rd);
        TDLS_CHECK(oks == okd);
        TDLS_CHECK(rs == rd);
        if (oks && okd) {
            TDLS_CHECK(tdls_tests::same_bits(ys, yd));
            TDLS_CHECK(tdls_tests::same_bits(Ms, Md));
            ++solved;
        }
        refused_total += rs;
    }
    // Floors: the batch is invertible, and its injected blocks are
    // refused.
    TDLS_CHECK(solved == count);
    TDLS_CHECK(refused_total >= 2 * (count / 3));
}

using tdls::ArrowheadShape;
using tdls::ArrowheadStorage;
constexpr ArrowheadShape polycrystal{12, 6, 30, 6};
constexpr ArrowheadShape uneven{8, 6, 10, 2};
constexpr ArrowheadShape uneven_dense{8, 6, 10, 0};
constexpr ArrowheadShape iwan{6, 1, 24, 0};
constexpr ArrowheadShape thin_border{3, 6, 8, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/static_vs_dynamic/storages") {
    bridge_case<polycrystal, path_config(ArrowheadStorage::Full, true, false, 0)>(12, 500100);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0)>(12, 500200);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Csr, true, true, 0)>(12, 500300);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Bsr, true, true, 0)>(12, 500400);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Csr, false, false, 0)>(12, 500500);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Full, false, true, 0)>(12, 500600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_vs_dynamic/couplings_and_layouts") {
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Packed, false, false, 0)>(12, 501100);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Packed, false, true, 0)>(12, 501200);
    bridge_case<uneven, path_config(ArrowheadStorage::Packed, true, false, 0)>(12, 501300);
    bridge_case<uneven, path_config(ArrowheadStorage::Packed, true, true, 0)>(12, 501400);
    bridge_case<iwan, path_config(ArrowheadStorage::Packed, false, false, 0)>(12, 501500);
    bridge_case<iwan, path_config(ArrowheadStorage::Packed, false, true, 0)>(12, 501600);
    bridge_case<thin_border, path_config(ArrowheadStorage::Packed, false, false, 0)>(12, 501700);
    bridge_case<thin_border, path_config(ArrowheadStorage::Packed, false, true, 0)>(12, 501800);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_vs_dynamic/passes") {
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 1)>(12, 502100);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 2)>(12, 502200);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, true, 2)>(12, 502300);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, true, 5)>(12, 502400);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Packed, false, false, 3)>(12, 502500);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Packed, false, true, 4)>(12, 502600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_vs_dynamic/gather_in_place_unroll") {
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 2, true)>(12,
                                                                                          503100);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, true, 2, true)>(12,
                                                                                         503200);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0, false, true)>(
        12, 503300);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Packed, false, false, 0, false, true)>(
        12, 503400);
    bridge_case<uneven_dense, path_config(ArrowheadStorage::Packed, false, true, 0, false, true)>(
        12, 503500);
    bridge_case<polycrystal, path_config(ArrowheadStorage::Packed, true, false, 0, false, false,
                                         false)>(12, 503600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_vs_dynamic/registers") {
    using S = ArrowheadStorage;
    bridge_case<polycrystal, registers(path_config(S::Packed, true, false, 0))>(12, 504100);
    bridge_case<polycrystal, registers(path_config(S::Packed, true, false, 2))>(12, 504200);
    bridge_case<polycrystal, registers(path_config(S::Full, true, true, 0))>(12, 504300);
    bridge_case<polycrystal, registers(path_config(S::Bsr, true, true, 5))>(12, 504400);
    bridge_case<uneven_dense, registers(path_config(S::Csr, false, false, 0))>(12, 504500);
    bridge_case<uneven_dense, registers(path_config(S::Packed, false, false, 3))>(12, 504600);
    bridge_case<uneven_dense, registers(path_config(S::Packed, false, true, 0))>(12, 504700);
    bridge_case<uneven, registers(path_config(S::Packed, true, true, 2))>(12, 504800);
    bridge_case<iwan, registers(path_config(S::Packed, false, false, 0))>(12, 504900);
    bridge_case<thin_border, registers(path_config(S::Packed, false, true, 0))>(12, 505000);
    bridge_case<polycrystal, registers(path_config(S::Packed, true, false, 0, false, true))>(
        12, 505100);
    bridge_case<polycrystal,
                registers(path_config(S::Packed, true, false, 0, false, false, false))>(12, 505200);
}

// The choice of RowInterchange::Auto in this pass (a CPU pass):
// registers up to blocks of 12 in double and long double, 8 in float, tiles
// beyond.
static_assert(tdls::detail::physical_rows_fit<double>(12) &&
              !tdls::detail::physical_rows_fit<double>(13) &&
              tdls::detail::physical_rows_fit<float>(8) &&
              !tdls::detail::physical_rows_fit<float>(9) &&
              tdls::detail::physical_rows_fit<long double>(12) &&
              !tdls::detail::physical_rows_fit<long double>(13));

constexpr ArrowheadShape block12{12, 12, 4, 0};
constexpr ArrowheadShape block13{13, 13, 4, 0};

TDLS_TEST_CASE("arrowheadlupp/bridge/static_vs_dynamic/auto") {
    using S = ArrowheadStorage;
    // Auto runs the operations of the elimination it chooses, and the two
    // solvers make the same choice: in registers up to blocks of 12 in
    // double (this pass is a CPU one), by tiles beyond; one pass on CPU.
    same_operations<polycrystal, automatic(path_config(S::Packed, true, false, 0)),
                    registers(path_config(S::Packed, true, false, 0))>(12, 505300);
    same_operations<block12, automatic(path_config(S::Packed, false, false, 0)),
                    registers(path_config(S::Packed, false, false, 0))>(12, 505400);
    same_operations<block13, automatic(path_config(S::Packed, false, true, 0)),
                    path_config(S::Packed, false, true, 0)>(12, 505500);
    bridge_case<polycrystal, automatic(path_config(S::Packed, true, false, 0))>(12, 505600);
    bridge_case<block12, automatic(path_config(S::Packed, false, false, 0))>(12, 505700);
    bridge_case<block13, automatic(path_config(S::Packed, false, true, 0))>(12, 505800);
    bridge_case<iwan, automatic(path_config(S::Packed, false, false, 0))>(12, 505900);
}

TDLS_TEST_MAIN
