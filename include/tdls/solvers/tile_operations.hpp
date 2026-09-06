#ifndef TDLS_SOLVERS_TILE_OPERATIONS_HPP
#define TDLS_SOLVERS_TILE_OPERATIONS_HPP



/// \file
/// \brief Register-tile micro-kernels shared by the solver families.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A tile is a tile_size x tile_size register array with row stride
/// tile_size; partial (trailing) tiles use the same storage with extent
/// template parameters bounding every loop, so phantom slots are never
/// read or written and cost nothing.
///
/// These operate on register arrays only. Remote-memory movement lives
/// in the solvers, next to the addressing macros. Family-specific
/// kernels extend this struct in their own header.
///
/// Every loop is subject to the `unroll_inner` knob of the family
/// configuration: with it, the loops must effectively unroll or the
/// tiles are demoted to local memory on GPU backends; without it, no
/// pragma is emitted at all.



#include <tdls/core/macros.hpp>



// clang reports a forced unrolling that the optimizer could not perform
// through -Wpass-failed. The unrolling requested by TDLS_UNROLL_FORCE is
// a performance hint: a failed hint does not affect correctness. The
// suppression is scoped to this header and to that warning only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

namespace tdls {



/// \brief tile_size x tile_size register-tile micro-kernels shared by the
/// solver families.
/// \tparam T            scalar type
/// \tparam tile_size    tile size (int, row stride of the register tiles)
/// \tparam unroll_inner unroll knob, forwarded from the solver configuration
template<typename T, int tile_size, bool unroll_inner>
struct TileOperations {

    /// \brief Row swap k <-> r inside the KExKE active part of the tile,
    /// compile-time indexed on both sides.
    ///
    /// The equality test against every unrolled row index keeps the tile
    /// addressing static; a dynamic t[r*tile_size+j] would spill the tile.
    /// k_extent bounds both loops, so the phantom slots of trailing tiles
    /// are never touched, as in every other micro-kernel.
    /// \tparam k_extent active extent of the tile
    /// \param[in,out] t register tile
    /// \param[in]     k destination row
    /// \param[in]     r source row to swap in
    template<int k_extent>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void swap_rows(T* TDLS_RESTRICT t, int k,
                                                                      int r) noexcept {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int row = 0; row < k_extent; ++row) {
                if (row == r) {
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < k_extent; ++j) {
                        const T tmp            = t[k * tile_size + j];
                        t[k * tile_size + j]   = t[row * tile_size + j];
                        t[row * tile_size + j] = tmp;
                    }
                }
            }
        } else {
            for (int row = 0; row < k_extent; ++row) {
                if (row == r) {
                    for (int j = 0; j < k_extent; ++j) {
                        const T tmp            = t[k * tile_size + j];
                        t[k * tile_size + j]   = t[row * tile_size + j];
                        t[row * tile_size + j] = tmp;
                    }
                }
            }
        }
    }

    /// \brief B := L^-1 B, with L the unit lower part of the factored
    /// diagonal tile. L is KDxKD, B is KDxC.
    /// \tparam diag_extent extent of the factored diagonal tile
    /// \tparam col_extent  column extent of B
    /// \param[in]     lu factored diagonal tile (L\\U)
    /// \param[in,out] B  updated register tile
    template<int diag_extent, int col_extent>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    trsm_left_unit(const T* TDLS_RESTRICT lu, T* TDLS_RESTRICT B) noexcept {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int k = 0; k < diag_extent; ++k) {
                TDLS_UNROLL_FORCE
                for (int i = k + 1; i < diag_extent; ++i) {
                    const T L_ik = lu[i * tile_size + k];
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < col_extent; ++j)
                        B[i * tile_size + j] -= L_ik * B[k * tile_size + j];
                }
            }
        } else {
            for (int k = 0; k < diag_extent; ++k) {
                for (int i = k + 1; i < diag_extent; ++i) {
                    const T L_ik = lu[i * tile_size + k];
                    for (int j = 0; j < col_extent; ++j)
                        B[i * tile_size + j] -= L_ik * B[k * tile_size + j];
                }
            }
        }
    }

    /// \brief B := B U^-1, with U the upper part of the factored diagonal
    /// tile. U is KDxKD, B is RxKD.
    ///
    /// The diagonal of `lu` must already hold the pivot RECIPROCALS: no
    /// divisions here.
    /// \tparam diag_extent extent of the factored diagonal tile
    /// \tparam row_extent  row extent of B
    /// \param[in]     lu factored diagonal tile (L\\U, reciprocal diagonal)
    /// \param[in,out] B  updated register tile
    template<int diag_extent, int row_extent>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    trsm_right(const T* TDLS_RESTRICT lu, T* TDLS_RESTRICT B) noexcept {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int k = 0; k < diag_extent; ++k) {
                const T U_kk_inv = lu[k * tile_size + k];
                TDLS_UNROLL_FORCE
                for (int i = 0; i < row_extent; ++i)
                    B[i * tile_size + k] *= U_kk_inv;
                TDLS_UNROLL_FORCE
                for (int j = k + 1; j < diag_extent; ++j) {
                    const T U_kj = lu[k * tile_size + j];
                    TDLS_UNROLL_FORCE
                    for (int i = 0; i < row_extent; ++i)
                        B[i * tile_size + j] -= B[i * tile_size + k] * U_kj;
                }
            }
        } else {
            for (int k = 0; k < diag_extent; ++k) {
                const T U_kk_inv = lu[k * tile_size + k];
                for (int i = 0; i < row_extent; ++i)
                    B[i * tile_size + k] *= U_kk_inv;
                for (int j = k + 1; j < diag_extent; ++j) {
                    const T U_kj = lu[k * tile_size + j];
                    for (int i = 0; i < row_extent; ++i)
                        B[i * tile_size + j] -= B[i * tile_size + k] * U_kj;
                }
            }
        }
    }

    /// \brief Ct -= At*Bt with per-element dot-product accumulation.
    /// At is RxK, Bt is KxC, Ct is RxC.
    /// \tparam row_extent row extent of Ct and At
    /// \tparam col_extent column extent of Ct and Bt
    /// \tparam k_extent   inner extent (columns of At, rows of Bt)
    /// \param[in,out] Ct accumulator tile
    /// \param[in]     At left factor tile
    /// \param[in]     Bt right factor tile
    template<int row_extent, int col_extent, int k_extent>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void
    gemm_sub(T* TDLS_RESTRICT Ct, const T* TDLS_RESTRICT At, const T* TDLS_RESTRICT Bt) noexcept {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < row_extent; ++i) {
                TDLS_UNROLL_FORCE
                for (int j = 0; j < col_extent; ++j) {
                    T sum = T(0);
                    TDLS_UNROLL_FORCE
                    for (int k = 0; k < k_extent; ++k)
                        sum += At[i * tile_size + k] * Bt[k * tile_size + j];
                    Ct[i * tile_size + j] -= sum;
                }
            }
        } else {
            for (int i = 0; i < row_extent; ++i) {
                for (int j = 0; j < col_extent; ++j) {
                    T sum = T(0);
                    for (int k = 0; k < k_extent; ++k)
                        sum += At[i * tile_size + k] * Bt[k * tile_size + j];
                    Ct[i * tile_size + j] -= sum;
                }
            }
        }
    }
};



} // namespace tdls

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_TILE_OPERATIONS_HPP
