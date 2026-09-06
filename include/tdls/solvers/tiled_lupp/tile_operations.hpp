#ifndef TDLS_SOLVERS_TILED_LUPP_TILE_OPERATIONS_HPP
#define TDLS_SOLVERS_TILED_LUPP_TILE_OPERATIONS_HPP



/// \file
/// \brief LU register-tile micro-kernel of the TiledLUpp solver family.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// TiledLUppTileOperations extends the shared TileOperations of solvers/tile_operations.hpp
/// with the elimination kernel of the family, whose factored format
/// stores the RECIPROCAL of each pivot on the diagonal. Tile storage,
/// extent bounding and the `unroll_inner` knob follow the shared
/// header.



#include <tdls/core/macros.hpp>
#include <tdls/solvers/tile_operations.hpp>



// clang reports a forced unrolling that the optimizer could not perform
// through -Wpass-failed. The unrolling requested by TDLS_UNROLL_FORCE is
// a performance hint: a failed hint does not affect correctness. The
// suppression is scoped to this header and to that warning only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

namespace tdls {



/// \brief tile_size x tile_size register-tile micro-kernels of the TiledLUpp
/// solvers: the shared kernels plus the LU column elimination.
/// \tparam T            scalar type
/// \tparam tile_size    tile size (int, row stride of the register tiles)
/// \tparam unroll_inner unroll knob, forwarded from the TiledLUpp solver configuration
template<typename T, int tile_size, bool unroll_inner>
struct TiledLUppTileOperations : TileOperations<T, tile_size, unroll_inner> {

    /// \brief Gaussian elimination of column k inside the diagonal tile.
    ///
    /// Scales the sub-column by 1/pivot and updates the trailing block.
    /// Extents <row_extent, col_extent> bound the active part of the tile.
    /// The RECIPROCAL of the pivot is stored at the diagonal slot: every
    /// downstream consumer (trsm_right, backward substitution, out-of-tile
    /// replays) multiplies instead of dividing. The division is paid once
    /// here, where it already had to happen.
    /// \tparam row_extent active row extent of the tile
    /// \tparam col_extent active column extent of the tile
    /// \param[in,out] t register tile
    /// \param[in]     k column to eliminate
    template<int row_extent, int col_extent>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void eliminate_column(T* TDLS_RESTRICT t,
                                                                             int k) noexcept {
        const T inv_pivot    = T(1) / t[k * tile_size + k];
        t[k * tile_size + k] = inv_pivot;
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = k + 1; i < row_extent; ++i) {
                t[i * tile_size + k] *= inv_pivot;
                TDLS_UNROLL_FORCE
                for (int j = k + 1; j < col_extent; ++j)
                    t[i * tile_size + j] -= t[i * tile_size + k] * t[k * tile_size + j];
            }
        } else {
            for (int i = k + 1; i < row_extent; ++i) {
                t[i * tile_size + k] *= inv_pivot;
                for (int j = k + 1; j < col_extent; ++j)
                    t[i * tile_size + j] -= t[i * tile_size + k] * t[k * tile_size + j];
            }
        }
    }
};



} // namespace tdls

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_TILED_LUPP_TILE_OPERATIONS_HPP
