#ifndef TDLS_SOLVERS_TILED_LUPP_TILE_OPS_HPP
#define TDLS_SOLVERS_TILED_LUPP_TILE_OPS_HPP



/// \file
/// \brief LU register-tile micro-kernel of the TiledLUpp solver family.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// TiledLUppTileOps extends the shared TileOps of solvers/tile_ops.hpp
/// with the elimination kernel of the family, whose factored format
/// stores the RECIPROCAL of each pivot on the diagonal. Tile storage,
/// extent bounding and the `unroll_inner` knob follow the shared
/// header.



#include <tdls/core/macros.hpp>
#include <tdls/solvers/tile_ops.hpp>



// clang reports a forced unrolling that the optimizer could not perform
// through -Wpass-failed. The unrolling requested by TDLS_UNROLL_FORCE is
// a performance hint: a failed hint does not affect correctness. The
// suppression is scoped to this header and to that warning only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

namespace tdls {



/// \brief TSxTS register-tile micro-kernels of the TiledLUpp solvers:
/// the shared kernels plus the LU column elimination.
/// \tparam T            scalar type
/// \tparam TS           tile size (int, row stride of the register tiles)
/// \tparam unroll_inner unroll knob, forwarded from the TiledLUpp solver configuration
template<typename T, int TS, bool unroll_inner>
struct TiledLUppTileOps : TileOps<T, TS, unroll_inner> {

    /// \brief Gaussian elimination of column k inside the diagonal tile.
    ///
    /// Scales the sub-column by 1/pivot and updates the trailing block.
    /// Extents <R, C> bound the active part of the tile.
    /// The RECIPROCAL of the pivot is stored at the diagonal slot: every
    /// downstream consumer (trsm_right, backward substitution, out-of-tile
    /// replays) multiplies instead of dividing. The division is paid once
    /// here, where it already had to happen.
    /// \tparam R active row extent of the tile
    /// \tparam C active column extent of the tile
    /// \param[in,out] t register tile
    /// \param[in]     k column to eliminate
    template<int R, int C>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr void eliminate_column(T* TDLS_RESTRICT t,
                                                                             int k) noexcept {
        const T inv_pivot = T(1) / t[k * TS + k];
        t[k * TS + k]     = inv_pivot;
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = k + 1; i < R; ++i) {
                t[i * TS + k] *= inv_pivot;
                TDLS_UNROLL_FORCE
                for (int j = k + 1; j < C; ++j)
                    t[i * TS + j] -= t[i * TS + k] * t[k * TS + j];
            }
        } else {
            for (int i = k + 1; i < R; ++i) {
                t[i * TS + k] *= inv_pivot;
                for (int j = k + 1; j < C; ++j)
                    t[i * TS + j] -= t[i * TS + k] * t[k * TS + j];
            }
        }
    }
};



} // namespace tdls

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_TILED_LUPP_TILE_OPS_HPP
