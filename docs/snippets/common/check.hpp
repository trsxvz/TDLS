#ifndef TDLS_SNIPPETS_CHECK_HPP
#define TDLS_SNIPPETS_CHECK_HPP



/// \file
/// \brief Result check shared by the documentation snippets.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every snippet is a program: the code shown in the documentation
/// sits between two markers, and the program ends by comparing its
/// result to the expected solution. The return value of the check is
/// the exit code of the snippet, which ctest consumes.



#include <cmath>
#include <cstdio>
#include <cstring>



namespace snippets {



/// \brief Compares n entries of x, `stride` apart, to the expected ones.
/// \tparam T scalar type
/// \param[in] x        computed values
/// \param[in] stride   element stride of x
/// \param[in] expected expected values, contiguous
/// \param[in] n        number of entries
/// \param[in] tol      absolute tolerance
/// \return 0 when every entry matches within tol, 1 otherwise
template<typename T>
int check(const T* x, const int stride, const T* expected, const int n, const double tol = 1e-9) {
    for (int i = 0; i < n; ++i) {
        const double got  = static_cast<double>(x[i * stride]);
        const double want = static_cast<double>(expected[i]);
        if (!(std::fabs(got - want) <= tol)) {
            std::printf("entry %d: %.17g, expected %.17g\n", i, got, want);
            return 1;
        }
    }
    return 0;
}

/// \brief Compares n contiguous entries of x to the expected ones.
/// \tparam T scalar type
/// \param[in] x        computed values
/// \param[in] expected expected values
/// \param[in] n        number of entries
/// \param[in] tol      absolute tolerance
/// \return 0 when every entry matches within tol, 1 otherwise
template<typename T>
int check(const T* x, const T* expected, const int n, const double tol = 1e-9) {
    return check(x, 1, expected, n, tol);
}

/// \brief Checks A x = b for a row-major n x n matrix: the residual test of
/// the canonical columns, whose exact solution is not an integer vector.
/// \tparam T scalar type
/// \param[in] A   matrix, row-major
/// \param[in] x   computed solution
/// \param[in] b   right-hand side the product must reproduce
/// \param[in] n   dimension
/// \param[in] tol absolute tolerance on every entry of the product
/// \return 0 when every entry of A x matches b within tol, 1 otherwise
template<typename T>
int check_product(const T* A, const T* x, const T* b, const int n, const double tol = 1e-9) {
    for (int i = 0; i < n; ++i) {
        double acc = 0;
        for (int j = 0; j < n; ++j)
            acc += static_cast<double>(A[i * n + j]) * static_cast<double>(x[j]);
        if (!(std::fabs(acc - static_cast<double>(b[i])) <= tol)) {
            std::printf("row %d: %.17g, expected %.17g\n", i, acc, static_cast<double>(b[i]));
            return 1;
        }
    }
    return 0;
}

/// \brief Compares two contiguous ranges bitwise.
/// \tparam T scalar type (float or double: long double carries padding)
/// \param[in] a first range
/// \param[in] b second range
/// \param[in] n number of entries
/// \return 0 when the ranges are identical, 1 otherwise
template<typename T>
int check_bitwise(const T* a, const T* b, const int n) {
    if (std::memcmp(a, b, static_cast<std::size_t>(n) * sizeof(T)) == 0) return 0;
    std::printf("the two ranges differ\n");
    return 1;
}



} // namespace snippets



#endif // TDLS_SNIPPETS_CHECK_HPP
