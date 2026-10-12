#ifndef TDLS_TESTS_SOLVERS_ARROWHEAD_LUPP_ARROWHEAD_PROBLEMS_HPP
#define TDLS_TESTS_SOLVERS_ARROWHEAD_LUPP_ARROWHEAD_PROBLEMS_HPP



/// \file
/// \brief Reproducible arrowhead systems of the ArrowheadLUpp suites, and
/// their storages.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A Problem holds one arrowhead system as a full row-major matrix, zeros
/// outside the pattern, the form the reference LU reads. Its pattern is
/// dense (every border row touches every block unknown) or diagonal
/// (unknown i of a block touches border row coupling_row + i only); a
/// diagonal pattern also suits the solvers configured for the dense
/// coupling, which then store and read the zeros. The border diagonal is
/// reinforced so that the whole matrix stays well conditioned whatever
/// its blocks. The blocks are dominant (accepted by the pivot check) or
/// plain random (some may be refused); the injectors then make chosen
/// blocks singular or pivot-poor, the cases of the second pass. store()
/// writes a problem in any storage through ArrowheadStorageMap.

#include <cstdint>
#include <cstring>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"

namespace tdls_tests {



/// \brief Kind of the diagonal blocks of a generated problem.
enum class ArrowBlocks {
    Dominant, ///< random blocks with a dominant diagonal: every block passes the pivot check
    Random    ///< random blocks: pivoting inside the blocks, refusals possible
};

/// \brief One arrowhead system, full row-major.
/// \tparam T scalar type
template<typename T>
struct ArrowProblem {
    tdls::ArrowheadShape shape; ///< shape of the matrix
    bool diag = false;          ///< whether the coupling pattern is diagonal
    std::vector<T> A;           ///< n x n row-major, zeros outside the pattern
    std::vector<T> b;           ///< right-hand side

    /// \return the dimension
    [[nodiscard]] int n() const {
        return shape.n();
    }
    /// \brief Entry (r, c) of the full matrix.
    /// \param[in] r row
    /// \param[in] c column
    /// \return a reference to it
    T& at(const int r, const int c) {
        return A[static_cast<std::size_t>(r) * shape.n() + c];
    }
    /// \brief Entry (r, c) of the full matrix.
    /// \param[in] r row
    /// \param[in] c column
    /// \return its value
    [[nodiscard]] T at(const int r, const int c) const {
        return A[static_cast<std::size_t>(r) * shape.n() + c];
    }
};

/// \brief Builds a reproducible arrowhead system.
/// \tparam T scalar type
/// \param[in] shape  shape (coupling_row = 0 for a dense pattern)
/// \param[in] diag   whether the coupling pattern is diagonal
/// \param[in] blocks kind of the diagonal blocks
/// \param[in] seed   generator seed
/// \return the system
template<typename T>
ArrowProblem<T> make_arrow_problem(const tdls::ArrowheadShape& shape, const bool diag,
                                   const ArrowBlocks blocks, const std::uint64_t seed) {
    UniformGenerator g(seed, 0.5);
    ArrowProblem<T> p{shape, diag, {}, {}};
    const int B = shape.border, S = shape.block, NB = shape.blocks, n = shape.n();
    const int b0 = shape.coupling_row;
    p.A.assign(static_cast<std::size_t>(n) * n, T(0));
    p.b.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            p.at(r, c) = static_cast<T>(g.next());
    for (int r = 0; r < B; ++r)
        p.at(r, r) += static_cast<T>(2 + S * NB);
    for (int k = 0; k < NB; ++k) {
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            if (diag)
                p.at(b0 + i, row) = static_cast<T>(g.next());
            else
                for (int r = 0; r < B; ++r)
                    p.at(r, row) = static_cast<T>(g.next());
            for (int c = 0; c < B; ++c)
                p.at(row, c) = static_cast<T>(g.next());
            for (int j = 0; j < S; ++j)
                p.at(row, B + k * S + j) = static_cast<T>(g.next());
            if (blocks == ArrowBlocks::Dominant) p.at(row, row) += static_cast<T>(S + 1);
        }
    }
    for (int i = 0; i < n; ++i)
        p.b[static_cast<std::size_t>(i)] = static_cast<T>(g.next());
    return p;
}

/// \brief Zeroes row i and column i of block k: a singular block. The
/// matrix stays invertible through the border while fewer blocks than
/// the coupled border rows lose the same unknown.
/// \tparam T scalar type
/// \param[in,out] p problem
/// \param[in]     k block
/// \param[in]     i unknown of the block
template<typename T>
void zero_block_unknown(ArrowProblem<T>& p, const int k, const int i) {
    const int B = p.shape.border, S = p.shape.block;
    for (int t = 0; t < S; ++t) {
        p.at(B + k * S + i, B + k * S + t) = T(0);
        p.at(B + k * S + t, B + k * S + i) = T(0);
    }
}

/// \brief Scales block k down to tiny entries: its pivots fall far below
/// the couplings of the border rows, which partial pivoting over the
/// whole matrix would take. The pivot check refuses the block.
/// \tparam T scalar type
/// \param[in,out] p problem
/// \param[in]     k block
template<typename T>
void shrink_block(ArrowProblem<T>& p, const int k) {
    const int B = p.shape.border, S = p.shape.block;
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
            p.at(B + k * S + i, B + k * S + j) *= static_cast<T>(1e-4);
}

/// \brief Sets every coupling of the pattern, both ways (A_bk and A_kb),
/// to a fixed magnitude with its generated sign: strong couplings make
/// every block fail the pivot check, which reads A_bk under the
/// row-major layout and A_kb under the column-major one (it checks the
/// elimination of the transpose there). The zeros outside the pattern
/// stay zeros.
/// \tparam T scalar type
/// \param[in,out] p         problem
/// \param[in]     magnitude magnitude of the couplings
template<typename T>
void strong_couplings(ArrowProblem<T>& p, const T magnitude) {
    const int B = p.shape.border, n = p.n();
    auto set = [&](T& v) { v = v > T(0) ? magnitude : (v < T(0) ? -magnitude : T(0)); };
    for (int r = 0; r < B; ++r)
        for (int c = B; c < n; ++c) {
            set(p.at(r, c));
            set(p.at(c, r));
        }
}

/// \brief The standard batch of the bridge suites: system s of a batch,
/// dominant or random blocks in turn, one block of every third system
/// made singular and one of the next pivot-poor, so that the second pass
/// runs on two systems out of three.
/// \tparam T scalar type
/// \param[in] shape shape
/// \param[in] diag  whether the coupling pattern is diagonal
/// \param[in] s     index of the system in the batch
/// \param[in] seed  seed of the batch
/// \return the system
template<typename T>
ArrowProblem<T> batch_problem(const tdls::ArrowheadShape& shape, const bool diag, const int s,
                              const std::uint64_t seed) {
    auto p =
        make_arrow_problem<T>(shape, diag, s % 2 == 0 ? ArrowBlocks::Dominant : ArrowBlocks::Random,
                              seed + static_cast<std::uint64_t>(s));
    if (s % 3 == 1) zero_block_unknown(p, s % shape.blocks, s % shape.block);
    if (s % 3 == 2) shrink_block(p, (s + 1) % shape.blocks);
    return p;
}

/// \brief Writes a problem in a storage.
/// \tparam Map storage map (ArrowheadStorageMap)
/// \tparam T   scalar type
/// \param[in] p problem (its pattern must suit the coupling of Map)
/// \return the stored entries, zeros of the stored blocks included
template<typename Map, typename T>
std::vector<T> store_arrow(const ArrowProblem<T>& p) {
    const tdls::ArrowheadShape& s = p.shape;
    const int B = s.border, S = s.block, NB = s.blocks;
    std::vector<T> out(static_cast<std::size_t>(Map::size(s)), T(0));
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            out[static_cast<std::size_t>(Map::abb(s, r, c))] = p.at(r, c);
    for (int k = 0; k < NB; ++k)
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            if constexpr (Map::diag)
                out[static_cast<std::size_t>(Map::abk_diag(s, k, i))] =
                    p.at(s.coupling_row + i, row);
            else
                for (int r = 0; r < B; ++r)
                    out[static_cast<std::size_t>(Map::abk(s, k, r, i))] = p.at(r, row);
            for (int c = 0; c < B; ++c)
                out[static_cast<std::size_t>(Map::akb(s, k, i, c))] = p.at(row, c);
            for (int j = 0; j < S; ++j)
                out[static_cast<std::size_t>(Map::d(s, k, i, j))] = p.at(row, B + k * S + j);
        }
    return out;
}

/// \brief Bitwise equality of two vectors.
/// \tparam T scalar type
/// \param[in] a first vector
/// \param[in] b second vector
/// \return true when they have the same size and the same bits
template<typename T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}



} // namespace tdls_tests



#endif // TDLS_TESTS_SOLVERS_ARROWHEAD_LUPP_ARROWHEAD_PROBLEMS_HPP
