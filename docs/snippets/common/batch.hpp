#ifndef TDLS_SNIPPETS_BATCH_HPP
#define TDLS_SNIPPETS_BATCH_HPP



/// \file
/// \brief The batch of the batch snippets: three 4 x 4 systems A + s I,
/// s = 0, 1, 2, all with the solution (1, 2, 3, 4).
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.



namespace snippets {



constexpr int N     = 4; ///< dimension of the systems
constexpr int count = 3; ///< number of systems

constexpr double A[N * N] = {4, 1, 0, 2, 1, 5, 1, 0, 0, 1, 6, 1, 2, 0, 1, 7}; ///< the base matrix
constexpr double b[N]     = {14, 14, 24, 33};                                 ///< A x
constexpr double expected[N] = {1, 2, 3, 4};                                  ///< x

/// \brief Element k of the matrix of system s: A + s I.
/// \param[in] s system index
/// \param[in] k flat element index, row-major
/// \return the element
constexpr double matrix(const int s, const int k) {
    return A[k] + (k % (N + 1) == 0 ? s : 0);
}

/// \brief Entry i of the right-hand side of system s: b + s x.
/// \param[in] s system index
/// \param[in] i entry index
/// \return the entry
constexpr double rhs(const int s, const int i) {
    return b[i] + s * expected[i];
}



} // namespace snippets



#endif // TDLS_SNIPPETS_BATCH_HPP
