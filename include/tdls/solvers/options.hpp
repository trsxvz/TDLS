#ifndef TDLS_SOLVERS_OPTIONS_HPP
#define TDLS_SOLVERS_OPTIONS_HPP



/// \file
/// \brief Shared vocabulary and conventions of the solver families.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Knob types shared by the configuration of every solver family: the
/// elimination schedule of a factorization and the memory layout of a
/// dense matrix. Each family configuration carries its own members of
/// these types.
///
/// The families also share one calling convention. Every operand is a
/// raw pointer pre-offset by the caller plus one runtime element
/// stride, which covers the AoS, SoA and AoSoA batch layouts alike.
/// Offsets are computed in unsigned 32-bit arithmetic and must stay
/// below 2^32. Residency template booleans, where a solver offers
/// them, describe what the passed buffers are and carry no default.
/// Factorizing entry points come in counting and diagnostics-free
/// overloads, the latter compiling the diagnostics out entirely.



namespace tdls {



/// \brief Elimination schedule of a factorization.
enum class Schedule {
    RightLooking, ///< Factor the diagonal block, push updates into the trailing matrix.
    LeftLooking   ///< Pull updates from prior blocks when a block is visited.
};



/// \brief Memory layout of a dense matrix.
enum class MatrixLayout {
    RowMajor, ///< Element (r, c) at flat index r * N + c, the TFEL convention.
    ColMajor  ///< Element (r, c) at flat index c * N + r.
};



} // namespace tdls



/// \def TDLS_LAYOUT_INDEX
/// \brief Flat index of matrix element (r, c) under the configured
/// layout: r * dim + c row-major, c * dim + r column-major.
///
/// Expands inside solver member functions, where a constexpr `Config`
/// value carrying a `layout` member is in scope.
#define TDLS_LAYOUT_INDEX(r, c, dim)                                                               \
    (Config.layout == tdls::MatrixLayout::RowMajor ? unsigned((r) * (dim) + (c))                   \
                                                   : unsigned((c) * (dim) + (r)))



#endif // TDLS_SOLVERS_OPTIONS_HPP
