#ifndef TDLS_SOLVERS_OPTIONS_HPP
#define TDLS_SOLVERS_OPTIONS_HPP



/// \file
/// \brief Shared vocabulary of the solver families.
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



#endif // TDLS_SOLVERS_OPTIONS_HPP
