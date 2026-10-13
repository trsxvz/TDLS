/// \file
/// \brief Bridge suite: the four storages of ArrowheadLUpp are equivalent,
/// and their index maps match the canonical compressed formats.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The storage only places the entries: at equal shape, coupling and
/// layout, Full, Packed, Csr and Bsr run the same arithmetic, so the
/// verdicts, the counts of refused blocks and the solutions are bitwise
/// identical (the standard batch, second pass included). The index map
/// of each compressed storage is checked independently: walking the
/// canonical CSR or BSR index arrays written by csr_pattern and
/// bsr_pattern over the full matrix must produce exactly the values the
/// map stores, under both layouts and both couplings. Every map must be
/// a bijection onto [0, size). valid() accepts the shapes the solvers
/// accept and rejects the others, absurd ones included, without
/// overflowing. The runtime solver rejects, before any access, the shapes
/// whose workspace passes 2^31 entries.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"

namespace {

using tdls::ArrowheadCoupling;
using tdls::ArrowheadShape;
using tdls::ArrowheadStorage;
using tdls::MatrixLayout;

/// \brief Configuration of one storage.
/// \param[in] storage storage
/// \param[in] diag    diagonal coupling
/// \param[in] col     column-major layout
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> storage_config(const ArrowheadStorage storage,
                                                           const bool diag, const bool col) {
    tdls::ArrowheadLUppConfig<double> c;
    c.coupling = diag ? ArrowheadCoupling::Diagonal : ArrowheadCoupling::Dense;
    c.storage  = storage;
    c.layout   = col ? MatrixLayout::ColMajor : MatrixLayout::RowMajor;
    return c;
}

/// \brief Solves a problem in one storage.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in]  p       problem
/// \param[out] x       solution
/// \param[out] refused refused blocks
/// \return the verdict
template<ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
bool solve_in(const tdls_tests::ArrowProblem<double>& p, std::vector<double>& x, int& refused) {
    using Solver = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    auto M       = tdls_tests::store_arrow<typename Solver::storage_map>(p);
    std::vector<double> w(Solver::workspace_size);
    x = p.b;
    return Solver::template solve_inplace<true, true, true>(M.data(), 1, w.data(), 1, x.data(), 1,
                                                            refused);
}

/// \brief The four storages on a standard batch, bitwise against Packed.
/// \tparam Shape shape (border a multiple of the block, for Bsr)
/// \tparam diag  diagonal coupling
/// \tparam col   column-major layout
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<ArrowheadShape Shape, bool diag, bool col>
void storages_case(const int count, const std::uint64_t seed) {
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        const auto p = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        std::vector<double> x0, x1, x2, x3;
        int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        const bool ok0 =
            solve_in<Shape, storage_config(ArrowheadStorage::Packed, diag, col)>(p, x0, r0);
        const bool ok1 =
            solve_in<Shape, storage_config(ArrowheadStorage::Full, diag, col)>(p, x1, r1);
        const bool ok2 =
            solve_in<Shape, storage_config(ArrowheadStorage::Csr, diag, col)>(p, x2, r2);
        const bool ok3 =
            solve_in<Shape, storage_config(ArrowheadStorage::Bsr, diag, col)>(p, x3, r3);
        TDLS_CHECK(ok0 && ok1 && ok2 && ok3);
        TDLS_CHECK(r0 == r1 && r0 == r2 && r0 == r3);
        TDLS_CHECK(tdls_tests::same_bits(x0, x1));
        TDLS_CHECK(tdls_tests::same_bits(x0, x2));
        TDLS_CHECK(tdls_tests::same_bits(x0, x3));
        if (ok0) ++solved;
    }
    TDLS_CHECK(solved == count);
}

/// \brief The index map of a storage is a bijection onto [0, size), and
/// a compressed one matches its canonical index arrays.
/// \tparam storage storage
/// \tparam diag    diagonal coupling
/// \tparam col     column-major layout
/// \param[in] shape shape (valid)
/// \param[in] seed  generator seed
template<ArrowheadStorage storage, bool diag, bool col>
void map_case(const ArrowheadShape shape, const std::uint64_t seed) {
    using Map =
        tdls::ArrowheadStorageMap<storage,
                                  diag ? ArrowheadCoupling::Diagonal : ArrowheadCoupling::Dense,
                                  col ? MatrixLayout::ColMajor : MatrixLayout::RowMajor>;
    TDLS_CHECK(Map::valid(shape));
    const auto p =
        tdls_tests::make_arrow_problem<double>(shape, diag, tdls_tests::ArrowBlocks::Random, seed);
    const int n = shape.n(), size = Map::size(shape);
    // Bijection: every index of the pattern hit once (Full also holds the
    // zeros outside the pattern).
    std::vector<int> hits(static_cast<std::size_t>(size), 0);
    int pattern = 0;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            if (Map::in_pattern(shape, r, c)) ++pattern;
    auto stored = tdls_tests::store_arrow<Map>(p);
    if constexpr (storage != ArrowheadStorage::Full) {
        // Bsr also stores the zeros inside its blocks.
        TDLS_CHECK(pattern <= size);
        // Walk the canonical arrays over the full matrix.
        std::vector<double> walked(static_cast<std::size_t>(size), 0.0);
        if constexpr (storage == ArrowheadStorage::Csr) {
            TDLS_CHECK(pattern == size);
            std::vector<int> ptr(n + 1), idx(static_cast<std::size_t>(size));
            Map::csr_pattern(shape, ptr.data(), idx.data());
            TDLS_CHECK(ptr[n] == size);
            for (int m = 0; m < n; ++m)
                for (int e = ptr[m]; e < ptr[m + 1]; ++e) {
                    walked[static_cast<std::size_t>(e)] = col ? p.at(idx[e], m) : p.at(m, idx[e]);
                    ++hits[static_cast<std::size_t>(e)];
                }
        } else {
            const int S = shape.block, SS = S * S, nb = shape.border / S + shape.blocks;
            std::vector<int> ptr(nb + 1), idx(static_cast<std::size_t>(size / SS));
            Map::bsr_pattern(shape, ptr.data(), idx.data());
            TDLS_CHECK(ptr[nb] * SS == size);
            for (int m = 0; m < nb; ++m)
                for (int e = ptr[m]; e < ptr[m + 1]; ++e) {
                    const int R = col ? idx[e] : m, C = col ? m : idx[e];
                    for (int i = 0; i < S; ++i)
                        for (int j = 0; j < S; ++j) {
                            const int at = e * SS + (col ? j * S + i : i * S + j);
                            walked[static_cast<std::size_t>(at)] = p.at(R * S + i, C * S + j);
                            ++hits[static_cast<std::size_t>(at)];
                        }
                }
        }
        TDLS_CHECK(tdls_tests::same_bits(walked, stored));
        bool once = true;
        for (const int h : hits)
            once = once && h == 1;
        TDLS_CHECK(once);
    } else {
        TDLS_CHECK(size == n * n);
        bool placed = true;
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                placed = placed && stored[static_cast<std::size_t>(col ? c * n + r : r * n + c)] ==
                                       p.at(r, c);
        TDLS_CHECK(placed);
    }
}

constexpr ArrowheadShape polycrystal{12, 6, 30, 6};
constexpr ArrowheadShape reduced{6, 6, 30, 0};
constexpr ArrowheadShape square{6, 2, 7, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/storages/solutions") {
    storages_case<polycrystal, true, false>(12, 600100);
    storages_case<polycrystal, true, true>(12, 600200);
    storages_case<reduced, false, false>(12, 600300);
    storages_case<reduced, false, true>(12, 600400);
    storages_case<square, false, false>(12, 600500);
    storages_case<square, true, true>(12, 600600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/storages/maps") {
    map_case<ArrowheadStorage::Csr, false, false>(ArrowheadShape{8, 6, 10, 0}, 601100);
    map_case<ArrowheadStorage::Csr, false, true>(ArrowheadShape{8, 6, 10, 0}, 601200);
    map_case<ArrowheadStorage::Csr, true, false>(ArrowheadShape{8, 6, 10, 2}, 601300);
    map_case<ArrowheadStorage::Csr, true, true>(ArrowheadShape{8, 6, 10, 2}, 601400);
    map_case<ArrowheadStorage::Bsr, false, false>(ArrowheadShape{12, 6, 9, 0}, 601500);
    map_case<ArrowheadStorage::Bsr, false, true>(ArrowheadShape{12, 6, 9, 0}, 601600);
    map_case<ArrowheadStorage::Bsr, true, false>(ArrowheadShape{12, 6, 9, 6}, 601700);
    map_case<ArrowheadStorage::Bsr, true, true>(ArrowheadShape{12, 6, 9, 0}, 601800);
    map_case<ArrowheadStorage::Full, true, false>(ArrowheadShape{8, 6, 10, 2}, 601900);
    map_case<ArrowheadStorage::Full, false, true>(ArrowheadShape{8, 6, 10, 0}, 602000);
    map_case<ArrowheadStorage::Csr, true, false>(ArrowheadShape{1, 1, 12, 0}, 602100);
    map_case<ArrowheadStorage::Bsr, false, true>(ArrowheadShape{2, 1, 20, 0}, 602200);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/storages/valid") {
    using Dense  = tdls::ArrowheadStorageMap<ArrowheadStorage::Packed, ArrowheadCoupling::Dense,
                                             MatrixLayout::RowMajor>;
    using Diag   = tdls::ArrowheadStorageMap<ArrowheadStorage::Packed, ArrowheadCoupling::Diagonal,
                                             MatrixLayout::RowMajor>;
    using BsrMap = tdls::ArrowheadStorageMap<ArrowheadStorage::Bsr, ArrowheadCoupling::Diagonal,
                                             MatrixLayout::ColMajor>;
    using Full   = tdls::ArrowheadStorageMap<ArrowheadStorage::Full, ArrowheadCoupling::Dense,
                                             MatrixLayout::RowMajor>;
    TDLS_CHECK(Dense::valid(ArrowheadShape{3, 6, 8, 0}));
    TDLS_CHECK(!Dense::valid(ArrowheadShape{3, 6, 8, 1})); // coupling_row under Dense
    TDLS_CHECK(!Dense::valid(ArrowheadShape{0, 6, 8, 0})); // empty border
    TDLS_CHECK(!Dense::valid(ArrowheadShape{6, 0, 8, 0})); // empty blocks
    TDLS_CHECK(!Dense::valid(ArrowheadShape{6, 6, 0, 0})); // no block
    TDLS_CHECK(Diag::valid(ArrowheadShape{8, 6, 10, 2}));
    TDLS_CHECK(!Diag::valid(ArrowheadShape{8, 6, 10, 3})); // coupled rows past the border
    TDLS_CHECK(!Diag::valid(ArrowheadShape{3, 6, 8, 0}));  // border smaller than the block
    TDLS_CHECK(BsrMap::valid(ArrowheadShape{12, 6, 9, 6}));
    TDLS_CHECK(!BsrMap::valid(ArrowheadShape{8, 6, 10, 2}));  // border not a multiple
    TDLS_CHECK(!BsrMap::valid(ArrowheadShape{12, 6, 9, 3}));  // coupling_row not a multiple
    TDLS_CHECK(Full::valid(ArrowheadShape{12, 6, 7700, 0}));  // n = 46212
    TDLS_CHECK(!Full::valid(ArrowheadShape{12, 6, 7800, 0})); // n^2 past 2^31
    // Shapes whose checks overflowed in int, or in 64 bits.
    constexpr int big = 2147483647;
    TDLS_CHECK(!Diag::valid(ArrowheadShape{1, 1, 1, big}));           // coupling_row + block
    TDLS_CHECK(!Full::valid(ArrowheadShape{65536, 65536, 65535, 0})); // n = 2^32, n^2 = 2^64
    TDLS_CHECK(!Dense::valid(ArrowheadShape{big, big, big, 0}));
    TDLS_CHECK(!Dense::valid(ArrowheadShape{1, 1, -1, 0}));
}

TDLS_TEST_CASE("arrowheadlupp/bridge/storages/dynamic_rejects") {
    constexpr auto config = [] {
        tdls::ArrowheadLUppConfig<double> c;
        c.storage = ArrowheadStorage::Packed;
        return c;
    }();
    using Dynamic = tdls::ArrowheadLUppSolverDynamic<double, config>;
    // The largest border {B, 1, 1} whose workspace stays below 2^31, and
    // the next one, which valid() of the storage accepts: its workspace
    // overflowed an int, negative from there, then small and positive
    // again from a border of 32767.
    TDLS_CHECK(Dynamic::valid(ArrowheadShape{23168, 1, 1, 0}));
    TDLS_CHECK(Dynamic::workspace_size(ArrowheadShape{23168, 1, 1, 0}) == 2147302923);
    for (const int border : {23169, 25000, 32767}) {
        const ArrowheadShape shape{border, 1, 1, 0};
        TDLS_CHECK(Dynamic::storage_map::valid(shape));
        TDLS_CHECK(!Dynamic::valid(shape));
        TDLS_CHECK(Dynamic::workspace_size(shape) == 0);
        TDLS_CHECK(Dynamic::streamed_workspace_size(shape) == 0);
        TDLS_CHECK(Dynamic::pivot_size(shape) == 0);
    }
    // The workspace grows with the right-hand sides.
    TDLS_CHECK(Dynamic::valid(ArrowheadShape{8, 6, 10, 0}, 1000));
    TDLS_CHECK(!Dynamic::valid(ArrowheadShape{8, 6, 10, 0}, 0));
    TDLS_CHECK(!Dynamic::valid(ArrowheadShape{8, 6, 10, 0}, 100000000));
    TDLS_CHECK(!Dynamic::valid(ArrowheadShape{8, 6, 10, 0}, 2147483647));
    // A rejected shape returns false before any access: null arrays.
    for (const ArrowheadShape shape :
         {ArrowheadShape{2, 1, 3, -1}, ArrowheadShape{0, 1, 3, 0}, ArrowheadShape{2, 1, -3, 0},
          ArrowheadShape{25000, 1, 1, 0}}) {
        int refused = -1;
        TDLS_CHECK(!Dynamic::solve_inplace(shape, nullptr, 1, nullptr, 1, nullptr, 1, nullptr, 1,
                                           refused));
        TDLS_CHECK(refused == 0);
        TDLS_CHECK(!Dynamic::solve(shape, nullptr, 1, nullptr, 1, nullptr, 1, nullptr, nullptr, 1));
        TDLS_CHECK(!Dynamic::solve_multirhs(shape, 2, nullptr, 1, nullptr, 1, nullptr, 1, nullptr,
                                            nullptr, 1, 1));
    }
    int refused = -1;
    TDLS_CHECK(!Dynamic::solve_inplace_multirhs(ArrowheadShape{8, 6, 10, 0}, -4, nullptr, 1,
                                                nullptr, 1, nullptr, 1, nullptr, 1, 1, refused));
    TDLS_CHECK(refused == 0);
}

TDLS_TEST_MAIN
