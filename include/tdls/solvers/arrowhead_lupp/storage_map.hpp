#ifndef TDLS_SOLVERS_ARROWHEAD_LUPP_STORAGE_MAP_HPP
#define TDLS_SOLVERS_ARROWHEAD_LUPP_STORAGE_MAP_HPP



/// \file
/// \brief Position of every entry of an arrowhead matrix in its storage.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The pattern of an arrowhead matrix follows from its shape, so the index
/// of every entry in a storage is a function of the shape, the storage,
/// the coupling and the layout: ArrowheadStorageMap computes it. The
/// caller fills its matrix through it, and the solvers read through the
/// same functions. With a compile-time shape every index folds to a
/// constant.
///
/// Row-major storages compress or order by rows, column-major ones by
/// columns:
///   - Full: (r, c) at r * n + c, or c * n + r;
///   - Packed: Abb, then the couplings (border x block per block, or one
///     entry per block row under the diagonal coupling), then the A_kb
///     (block x border each), then the D_k (block x block each), every
///     stored block by rows or by columns;
///   - Csr: border row r holds Abb(r, :) then its coupling entries; block
///     row (k, i) holds A_kb(i, :) then D_k(i, :). CSC: border column c
///     holds Abb(:, c) then A_kb(:, c) of every block; block column (k, j)
///     holds its coupling entries (one under the diagonal coupling, border
///     otherwise) then D_k(:, j);
///   - Bsr / BSC: square blocks of `block` entries, stored by rows or by
///     columns, block rows (block columns) in increasing order.
///
/// Indices are ints: the number of stored entries must stay below 2^31,
/// which valid() checks.



#include <tdls/core/macros.hpp>
#include <tdls/solvers/arrowhead_lupp/config.hpp>
#include <tdls/solvers/options.hpp>



namespace tdls {



/// \brief Index of every entry of an arrowhead matrix in its storage.
/// \tparam Storage  storage of the matrix
/// \tparam Coupling pattern of the couplings A_bk
/// \tparam Layout   order of the stored matrix, by rows or by columns
template<ArrowheadStorage Storage, ArrowheadCoupling Coupling, MatrixLayout Layout>
struct ArrowheadStorageMap {
    /// Whether the couplings are diagonal.
    static constexpr bool diag = Coupling == ArrowheadCoupling::Diagonal;
    /// Whether the matrix is stored by columns.
    static constexpr bool col = Layout == MatrixLayout::ColMajor;

    /// \brief Whether a shape suits this storage and coupling: positive
    /// sizes, coupled rows inside the border (diagonal coupling) or
    /// coupling_row = 0 (dense coupling), a border and a coupling_row
    /// multiple of the block (Bsr), fewer than 2^31 stored entries.
    /// \param[in] s shape
    /// \return true when the solvers accept the shape
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool valid(const ArrowheadShape& s) noexcept {
        if (s.border < 1 || s.block < 1 || s.blocks < 1 || s.coupling_row < 0) return false;
        if (diag ? s.coupling_row + s.block > s.border : s.coupling_row != 0) return false;
        if (Storage == ArrowheadStorage::Bsr &&
            (s.border % s.block != 0 || (diag && s.coupling_row % s.block != 0)))
            return false;
        return wide_size(s) < (1LL << 31);
    }

    /// \brief Number of stored entries, in 64-bit arithmetic (the
    /// overflow check of valid()).
    /// \param[in] s shape
    /// \return the number of entries of the storage
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr long long
    wide_size(const ArrowheadShape& s) noexcept {
        const long long B = s.border, S = s.block, NB = s.blocks, n = B + S * NB;
        if constexpr (Storage == ArrowheadStorage::Full) {
            return n * n;
        } else if constexpr (Storage == ArrowheadStorage::Packed) {
            return B * B + (diag ? NB * S : NB * B * S) + NB * S * B + NB * S * S;
        } else if constexpr (Storage == ArrowheadStorage::Csr) {
            if constexpr (col)
                return B * n + NB * S * (diag ? 1 + S : B + S);
            else
                return (diag ? B * B + NB * S : B * n) + NB * S * (B + S);
        } else {
            const long long BB = S > 0 ? B / S : 0;
            if constexpr (col)
                return (BB * (BB + NB) + NB * (diag ? 2 : BB + 1)) * S * S;
            else
                return ((diag ? BB * BB + NB : BB * (BB + NB)) + NB * (BB + 1)) * S * S;
        }
    }

    /// \brief Number of stored entries.
    /// \param[in] s shape (valid)
    /// \return the number of entries of the storage
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    size(const ArrowheadShape& s) noexcept {
        return static_cast<int>(wide_size(s));
    }

    /// \brief Border-border entry (r, c), r and c < border.
    /// \param[in] s shape (valid)
    /// \param[in] r border row
    /// \param[in] c border column
    /// \return its index in the storage
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    abb(const ArrowheadShape& s, const int r, const int c) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks, n = B + S * NB;
        if constexpr (Storage == ArrowheadStorage::Full) {
            return col ? c * n + r : r * n + c;
        } else if constexpr (Storage == ArrowheadStorage::Packed) {
            return col ? c * B + r : r * B + c;
        } else if constexpr (Storage == ArrowheadStorage::Csr) {
            return col ? c * n + r : csr_row(s, r) + c;
        } else if constexpr (col) {
            const int BB = B / S;
            return ((c / S) * (BB + NB) + r / S) * S * S + in_block(s, r % S, c % S);
        } else {
            return (bsr_row(s, r / S) + c / S) * S * S + in_block(s, r % S, c % S);
        }
    }

    /// \brief Coupling entry of border row r and unknown i of block k
    /// (dense coupling).
    /// \param[in] s shape (valid)
    /// \param[in] k block
    /// \param[in] r border row
    /// \param[in] i unknown of the block
    /// \return its index in the storage
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    abk(const ArrowheadShape& s, const int k, const int r, const int i) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks, n = B + S * NB;
        if constexpr (Storage == ArrowheadStorage::Full) {
            return col ? (B + k * S + i) * n + r : r * n + B + k * S + i;
        } else if constexpr (Storage == ArrowheadStorage::Packed) {
            return col ? B * B + (k * S + i) * B + r : B * B + (k * B + r) * S + i;
        } else if constexpr (Storage == ArrowheadStorage::Csr) {
            return col ? csc_block_col(s, k, i) + r : csr_row(s, r) + B + k * S + i;
        } else if constexpr (col) {
            return (bsc_block_col(s, k) + r / S) * S * S + in_block(s, r % S, i);
        } else {
            return (bsr_row(s, r / S) + B / S + k) * S * S + in_block(s, r % S, i);
        }
    }

    /// \brief Coupling entry of border row coupling_row + i and unknown i
    /// of block k (diagonal coupling).
    /// \param[in] s shape (valid)
    /// \param[in] k block
    /// \param[in] i unknown of the block
    /// \return its index in the storage
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    abk_diag(const ArrowheadShape& s, const int k, const int i) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks, n = B + S * NB, b0 = s.coupling_row;
        if constexpr (Storage == ArrowheadStorage::Full) {
            return col ? (B + k * S + i) * n + b0 + i : (b0 + i) * n + B + k * S + i;
        } else if constexpr (Storage == ArrowheadStorage::Packed) {
            return B * B + k * S + i;
        } else if constexpr (Storage == ArrowheadStorage::Csr) {
            return col ? csc_block_col(s, k, i) : csr_row(s, b0 + i) + B + k;
        } else if constexpr (col) {
            return bsc_block_col(s, k) * S * S + in_block(s, i, i);
        } else {
            return (bsr_row(s, b0 / S) + B / S + k) * S * S + in_block(s, i, i);
        }
    }

    /// \brief Entry of row i of block k in border column c.
    /// \param[in] s shape (valid)
    /// \param[in] k block
    /// \param[in] i row of the block
    /// \param[in] c border column
    /// \return its index in the storage
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    akb(const ArrowheadShape& s, const int k, const int i, const int c) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks, n = B + S * NB;
        if constexpr (Storage == ArrowheadStorage::Full) {
            return col ? c * n + B + k * S + i : (B + k * S + i) * n + c;
        } else if constexpr (Storage == ArrowheadStorage::Packed) {
            const int o = B * B + (diag ? NB * S : NB * B * S);
            return col ? o + (k * B + c) * S + i : o + (k * S + i) * B + c;
        } else if constexpr (Storage == ArrowheadStorage::Csr) {
            return col ? c * n + B + k * S + i : csr_block_row(s, k, i) + c;
        } else if constexpr (col) {
            const int BB = B / S;
            return ((c / S) * (BB + NB) + BB + k) * S * S + in_block(s, i, c % S);
        } else {
            return (bsr_block_row(s, k) + c / S) * S * S + in_block(s, i, c % S);
        }
    }

    /// \brief Entry (i, j) of the diagonal block k.
    /// \param[in] s shape (valid)
    /// \param[in] k block
    /// \param[in] i row of the block
    /// \param[in] j column of the block
    /// \return its index in the storage
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    d(const ArrowheadShape& s, const int k, const int i, const int j) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks, n = B + S * NB;
        if constexpr (Storage == ArrowheadStorage::Full) {
            return col ? (B + k * S + j) * n + B + k * S + i : (B + k * S + i) * n + B + k * S + j;
        } else if constexpr (Storage == ArrowheadStorage::Packed) {
            const int o = B * B + (diag ? NB * S : NB * B * S) + NB * S * B;
            return o + k * S * S + in_block(s, i, j);
        } else if constexpr (Storage == ArrowheadStorage::Csr) {
            return col ? csc_block_col(s, k, j) + (diag ? 1 : B) + i
                       : csr_block_row(s, k, i) + B + j;
        } else if constexpr (col) {
            return (bsc_block_col(s, k) + (diag ? 1 : B / S)) * S * S + in_block(s, i, j);
        } else {
            return (bsr_block_row(s, k) + B / S) * S * S + in_block(s, i, j);
        }
    }

    /// \brief Whether entry (r, c) of the full matrix belongs to the
    /// pattern.
    /// \param[in] s shape (valid)
    /// \param[in] r row of the full matrix
    /// \param[in] c column of the full matrix
    /// \return true for a stored entry
    [[nodiscard]] TDLS_HOST_DEVICE static constexpr bool
    in_pattern(const ArrowheadShape& s, const int r, const int c) noexcept {
        const int B = s.border, S = s.block, b0 = s.coupling_row;
        if (r < B && c < B) return true;
        if (r < B) return !diag || (r >= b0 && r < b0 + S && (c - B) % S == r - b0);
        if (c < B) return true;
        return (r - B) / S == (c - B) / S;
    }

    /// \brief Writes the canonical compressed index arrays of the pattern:
    /// CSR (row pointers, column indices) under the row-major layout, CSC
    /// (column pointers, row indices) under the column-major one, minor
    /// indices in increasing order. Meaningful for the Csr storage.
    /// \param[in]  s   shape (valid)
    /// \param[out] ptr n + 1 pointers
    /// \param[out] idx size(s) minor indices
    TDLS_HOST_DEVICE static constexpr void csr_pattern(const ArrowheadShape& s, int* ptr,
                                                       int* idx) noexcept {
        const int n = s.n();
        int e       = 0;
        for (int a = 0; a < n; ++a) {
            ptr[a] = e;
            for (int b = 0; b < n; ++b)
                if (col ? in_pattern(s, b, a) : in_pattern(s, a, b)) idx[e++] = b;
        }
        ptr[n] = e;
    }

    /// \brief Writes the canonical block compressed index arrays, in
    /// square blocks of `block` entries: BSR by block rows under the
    /// row-major layout, BSC by block columns under the column-major one,
    /// minor block indices in increasing order. Meaningful for the Bsr
    /// storage.
    /// \param[in]  s   shape (valid, border a multiple of block)
    /// \param[out] ptr border / block + blocks + 1 pointers
    /// \param[out] idx size(s) / (block * block) minor block indices
    TDLS_HOST_DEVICE static constexpr void bsr_pattern(const ArrowheadShape& s, int* ptr,
                                                       int* idx) noexcept {
        const int BB = s.border / s.block, NB = s.blocks, Rc = s.coupling_row / s.block;
        auto in = [&](const int R, const int Cb) {
            if (R < BB) return Cb < BB || !diag || R == Rc;
            return Cb < BB || Cb == R;
        };
        int e = 0;
        for (int a = 0; a < BB + NB; ++a) {
            ptr[a] = e;
            for (int b = 0; b < BB + NB; ++b)
                if (col ? in(b, a) : in(a, b)) idx[e++] = b;
        }
        ptr[BB + NB] = e;
    }

  private:
    /// \brief Entry (a, b) of a stored square block, by rows or by
    /// columns.
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    in_block(const ArrowheadShape& s, const int a, const int b) noexcept {
        return col ? b * s.block + a : a * s.block + b;
    }
    /// \brief First entry of border row r (CSR).
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    csr_row(const ArrowheadShape& s, const int r) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks, b0 = s.coupling_row;
        if constexpr (diag) {
            const int coupled = r < b0 ? 0 : (r < b0 + S ? r - b0 : S);
            return r * B + NB * coupled;
        } else {
            return r * (B + S * NB);
        }
    }
    /// \brief First entry of row i of block k (CSR).
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    csr_block_row(const ArrowheadShape& s, const int k, const int i) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks;
        return (diag ? B * B + NB * S : B * (B + S * NB)) + (k * S + i) * (B + S);
    }
    /// \brief First entry of column j of block k (CSC).
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    csc_block_col(const ArrowheadShape& s, const int k, const int j) noexcept {
        const int B = s.border, S = s.block, NB = s.blocks;
        return B * (B + S * NB) + (k * S + j) * (diag ? 1 + S : B + S);
    }
    /// \brief First block of border block row R (BSR).
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    bsr_row(const ArrowheadShape& s, const int R) noexcept {
        const int BB = s.border / s.block, NB = s.blocks, Rc = s.coupling_row / s.block;
        if constexpr (diag)
            return R * BB + (R > Rc ? NB : 0);
        else
            return R * (BB + NB);
    }
    /// \brief First block of the block row of block k (BSR).
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    bsr_block_row(const ArrowheadShape& s, const int k) noexcept {
        const int BB = s.border / s.block, NB = s.blocks;
        return (diag ? BB * BB + NB : BB * (BB + NB)) + k * (BB + 1);
    }
    /// \brief First block of the block column of block k (BSC).
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU static constexpr int
    bsc_block_col(const ArrowheadShape& s, const int k) noexcept {
        const int BB = s.border / s.block, NB = s.blocks;
        return BB * (BB + NB) + k * (diag ? 2 : BB + 1);
    }
};



} // namespace tdls



#endif // TDLS_SOLVERS_ARROWHEAD_LUPP_STORAGE_MAP_HPP
