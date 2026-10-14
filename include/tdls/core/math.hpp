#ifndef TDLS_CORE_MATH_HPP
#define TDLS_CORE_MATH_HPP



/// \file
/// \brief Scalar math helpers usable in constant expressions.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.



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
[[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU constexpr T abs(const T x) noexcept {
#if defined(__has_builtin)
#if __has_builtin(__builtin_is_constant_evaluated)
    if (__builtin_is_constant_evaluated()) return x < T(0) ? -x : x;
#endif
#elif defined(_MSC_VER) && _MSC_VER >= 1925
    if (__builtin_is_constant_evaluated()) return x < T(0) ? -x : x;
#endif
    return std::fabs(x);
}

/// \brief A dimension hidden from the optimizer of nvcc when it compiles
/// device code for Blackwell (sm_100 and later), the identity elsewhere.
///
/// nvcc miscompiles TiledLUppSolverDynamic for these targets when a
/// dimension passed by the caller is a compile-time constant and
/// propagates into its loops (TDLS issue #16). Every entry point of that
/// solver passes its dimensions through this function first, so that the
/// solver is compiled for a runtime dimension, as it is meant to be. The
/// runtime solvers of CooperativeLUpp and ArrowheadLUpp were found
/// correct; ArrowheadLUpp reaches TiledLUpp through these entry points.
/// nvc++ compiles the solver correctly for these targets (checked with
/// -stdpar=gpu for cc120), so the statement is limited to nvcc. The empty
/// assembly statement costs no instruction.
/// \param[in] v dimension
/// \return v
TDLS_HOST_DEVICE TDLS_FORCE_INLINE_GPU constexpr int runtime_dimension(int v) noexcept {
#if defined(__NVCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (!__builtin_is_constant_evaluated()) asm volatile("" : "+r"(v));
#endif
    return v;
}

} // namespace detail



} // namespace tdls



#endif // TDLS_CORE_MATH_HPP
