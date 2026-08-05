#ifndef TDLS_CORE_MATH_HPP
#define TDLS_CORE_MATH_HPP



/// \file
/// \brief Scalar math helpers usable in constant expressions.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. Released under the
/// BSD 3-Clause License (see the LICENSE file).



#include <cmath>

#include <tdls/core/macros.hpp>



namespace tdls {



namespace detail {

/// \brief Absolute value usable in constant expressions.
///
/// std::fabs is not constexpr before C++23, so constant evaluation
/// falls back to a ternary, indistinguishable from std::fabs in the
/// magnitude comparisons of the solvers (signed zeros compare equal,
/// NaNs propagate either way). At run time the function IS std::fabs:
/// the naive ternary alone would forbid the single sign-bit
/// instruction (it must preserve -0.0) and was measured 3% heavier in
/// device code on the pivot search.
/// \tparam T scalar type
/// \param[in] x value
/// \return the absolute value of x
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr T abs(const T x) noexcept {
#if defined(__has_builtin)
#if __has_builtin(__builtin_is_constant_evaluated)
    if (__builtin_is_constant_evaluated()) return x < T(0) ? -x : x;
#endif
#elif defined(_MSC_VER) && _MSC_VER >= 1925
    if (__builtin_is_constant_evaluated()) return x < T(0) ? -x : x;
#endif
    return std::fabs(x);
}

} // namespace detail



} // namespace tdls



#endif // TDLS_CORE_MATH_HPP
