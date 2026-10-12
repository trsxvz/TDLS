#ifndef TDLS_SNIPPETS_ARROW_HPP
#define TDLS_SNIPPETS_ARROW_HPP



/// \file
/// \brief The systems of the ArrowheadLUpp snippets: a 6 x 6 arrowhead
/// matrix with a border of 2 and two blocks of 2, its variants, and the
/// batch of the batch snippets.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every matrix is written in full, row-major, as the pages display it;
/// store() writes it in any storage through ArrowheadStorageMap.



#include <tdls/tdls.hpp>



namespace snippets::arrow {



constexpr int n = 6; ///< dimension of the systems
/// The shape: a border of 2 unknowns, then 2 blocks of 2 unknowns.
constexpr tdls::ArrowheadShape shape{.border = 2, .block = 2, .blocks = 2};

/// The base matrix. Its couplings are diagonal: unknown i of a block
/// touches border row i only.
constexpr double A[n * n]    = {6, 1, 1, 0, 2, 0, //
                                1, 7, 0, 1, 0, 1, //
                                1, 0, 5, 1, 0, 0, //
                                0, 2, 1, 4, 0, 0, //
                                2, 1, 0, 0, 6, 2, //
                                0, 1, 0, 0, 1, 5};
constexpr double b[n]        = {21, 25, 20, 23, 46, 37}; ///< A x
constexpr double expected[n] = {1, 2, 3, 4, 5, 6};       ///< x

/// The base matrix with dense couplings: the border rows touch every
/// unknown of the blocks.
constexpr double A_dense[n * n] = {6, 1, 1, 1, 2, 0, //
                                   1, 7, 0, 1, 1, 1, //
                                   1, 0, 5, 1, 0, 0, //
                                   0, 2, 1, 4, 0, 0, //
                                   2, 1, 0, 0, 6, 2, //
                                   0, 1, 0, 0, 1, 5};
constexpr double b_dense[n]     = {25, 30, 20, 23, 46, 37}; ///< A_dense x

/// The base matrix with a singular second block: the pivot check refuses
/// it, the second pass eliminates it with the border rows.
constexpr double A_singular_block[n * n] = {6, 1, 1, 0, 2, 0, //
                                            1, 7, 0, 1, 0, 1, //
                                            1, 0, 5, 1, 0, 0, //
                                            0, 2, 1, 4, 0, 0, //
                                            2, 1, 0, 0, 1, 1, //
                                            0, 1, 0, 0, 1, 1};
constexpr double b_singular_block[n]     = {21, 25, 20, 23, 15, 13}; ///< A_singular_block x

/// A second right-hand side of the base matrix, for the snippets of
/// several right-hand sides.
constexpr double b_second[n]        = {49, 45, 29, 26, 31, 12}; ///< A x_second
constexpr double expected_second[n] = {6, 5, 4, 3, 2, 1};       ///< x_second

/// The border block of the inverse of the base matrix (its first two rows
/// and columns), the inverse of the Schur complement of the border; column
/// m from m * 2.
constexpr double inverse_border[4] = {3349.0 / 16464, -598.0 / 16464, -474.0 / 16464,
                                      2700.0 / 16464};

constexpr int count = 3; ///< systems of the batch snippets

/// \brief Element k of the full matrix of system s of the batch: A + s I.
/// \param[in] s system index
/// \param[in] k flat element index, row-major
/// \return the element
constexpr double matrix(const int s, const int k) {
    return A[k] + (k % (n + 1) == 0 ? s : 0);
}

/// \brief Entry i of the right-hand side of system s of the batch: b + s x.
/// \param[in] s system index
/// \param[in] i entry index
/// \return the entry
constexpr double rhs(const int s, const int i) {
    return b[i] + s * expected[i];
}

/// \brief Writes a full row-major matrix of the snippet shape in a
/// storage: the entries of the pattern, each at the index of the map.
/// \tparam Map storage map (ArrowheadStorageMap)
/// \param[in]  full   matrix, n x n row-major
/// \param[out] stored Map::size(shape) entries
template<typename Map>
constexpr void store(const double* full, double* stored) {
    const int B = shape.border, S = shape.block;
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            stored[Map::abb(shape, r, c)] = full[r * n + c];
    for (int k = 0; k < shape.blocks; ++k)
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            if constexpr (Map::diag)
                stored[Map::abk_diag(shape, k, i)] = full[(shape.coupling_row + i) * n + row];
            else
                for (int r = 0; r < B; ++r)
                    stored[Map::abk(shape, k, r, i)] = full[r * n + row];
            for (int c = 0; c < B; ++c)
                stored[Map::akb(shape, k, i, c)] = full[row * n + c];
            for (int j = 0; j < S; ++j)
                stored[Map::d(shape, k, i, j)] = full[row * n + B + k * S + j];
        }
}



} // namespace snippets::arrow



#endif // TDLS_SNIPPETS_ARROW_HPP
