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
/// elimination schedule of a factorization, the row interchanges of its
/// partial pivoting and the memory layout of a dense matrix. Each family
/// configuration carries its own members of these types.
///
/// The families also share one calling convention. Every operand is a
/// raw pointer pre-offset by the caller plus one runtime element
/// stride, which covers the AoS, SoA and AoSoA batch layouts alike. The
/// entry points of the compile-time solvers also take each stride as a
/// template argument, dynamic_stride by default: a stride known at
/// compile time is given there (see dynamic_stride).
/// The flat element index of an object (N*N for a matrix) must stay
/// below 2^31; element offsets, index times stride, are computed in
/// 64-bit arithmetic (TDLS_STRIDED). Residency template
/// booleans, where a solver offers
/// them, describe what the passed buffers are and carry no default.
/// Factorizing entry points come in counting and diagnostics-free
/// overloads, the latter compiling the diagnostics out entirely.



#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/core/target.hpp>

#if defined(TDLS_DETAIL_SIDE_DISPATCH)
#include <nv/target>
#endif



namespace tdls {



/// \brief Elimination schedule of a factorization.
enum class Schedule {
    RightLooking, ///< Factor the diagonal block, push updates into the trailing matrix.
    LeftLooking   ///< Pull updates from prior blocks when a block is visited.
};



/// \brief Row interchanges of a factorization with partial pivoting.
enum class RowInterchange {
    Logical,  ///< Rows never move: the position of each row in the pivoted order is tracked.
    Physical, ///< Rows are exchanged, as in LAPACK: the row at each position is stored there.
    /// Physical when the matrix is small enough to stay in registers, Logical
    /// otherwise, for the target of the code (detail::physical_rows_fit):
    /// the default of the blocks of ArrowheadLUpp
    /// (ArrowheadLUppConfig::block_row_interchange), factored and used within
    /// one solve. TiledLUpp and CooperativeLUpp reject it.
    Auto
};

namespace detail {

/// \brief Whether a matrix of n unknowns is small enough for the physical
/// row interchanges on the given hardware (RowInterchange::Auto). On GPU,
/// the matrix must fit in the registers of a thread, n^2 values of T within
/// about 200 registers of 32 bits: n <= 14 in float, 10 in double and long
/// double (which GPUs compute as double). On CPU, measured on 6 cores (the
/// blocks of ArrowheadLUpp): n <= 8 in float (the tiled solver vectorizes
/// better beyond), 12 in double and long double.
/// \tparam T   scalar type
/// \tparam gpu the hardware: GPU (true) or CPU (false)
/// \param[in] n unknowns
/// \return true when Auto takes the physical row interchanges
template<typename T, bool gpu>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool physical_rows_fit_on(const int n) noexcept {
    if constexpr (gpu)
        return sizeof(T) <= 4 ? n <= 14 : n <= 10;
    else
        return sizeof(T) <= 4 ? n <= 8 : n <= 12;
}

/// \brief physical_rows_fit_on for the hardware of this compilation pass
/// (TDLS_TARGET_GPU).
/// \tparam T scalar type
/// \param[in] n unknowns
/// \return true when Auto takes the physical row interchanges
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool physical_rows_fit(const int n) noexcept {
    return physical_rows_fit_on<T, TDLS_TARGET_GPU != 0>(n);
}

/// \brief Whether a factorization of n unknowns interchanges its rows
/// physically on the given hardware, RowInterchange::Auto resolved.
/// \tparam T   scalar type
/// \tparam gpu the hardware: GPU (true) or CPU (false)
/// \param[in] r the row interchanges of the configuration
/// \param[in] n unknowns
/// \return true for physical row interchanges
template<typename T, bool gpu>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool physical_rows_on(const RowInterchange r,
                                                               const int n) noexcept {
    return r == RowInterchange::Physical ||
           (r == RowInterchange::Auto && physical_rows_fit_on<T, gpu>(n));
}

/// \brief physical_rows_on for the hardware of this compilation pass
/// (TDLS_TARGET_GPU).
/// \tparam T scalar type
/// \param[in] r the row interchanges of the configuration
/// \param[in] n unknowns
/// \return true for physical row interchanges
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool physical_rows(const RowInterchange r,
                                                            const int n) noexcept {
    return physical_rows_on<T, TDLS_TARGET_GPU != 0>(r, n);
}

#if defined(TDLS_DETAIL_SIDE_DISPATCH)
/// \brief Whether this code runs on the GPU side of a single-pass nvc++
/// compilation, for the defaults that depend on the hardware: the two
/// sides of NV_IF_TARGET, so that host code takes the CPU defaults and
/// device code the GPU ones.
/// \return true on the device side
TDLS_HOST_DEVICE inline bool on_device_side() noexcept {
    NV_IF_TARGET(NV_IS_DEVICE, (return true;), (return false;))
}
#endif

/// \brief Whether a factorization of n unknowns interchanges its rows
/// physically, decided at run time on the side of the code: as
/// physical_rows, except in a single-pass nvc++ compilation, where host
/// code takes the CPU rule and device code the GPU rule.
/// \tparam T scalar type
/// \param[in] r the row interchanges of the configuration
/// \param[in] n unknowns
/// \return true for physical row interchanges
template<typename T>
[[nodiscard]] TDLS_HOST_DEVICE constexpr bool physical_rows_here(const RowInterchange r,
                                                                 const int n) noexcept {
#if defined(TDLS_DETAIL_SIDE_DISPATCH)
    if (!std::is_constant_evaluated())
        return on_device_side() ? physical_rows_on<T, true>(r, n)
                                : physical_rows_on<T, false>(r, n);
#endif
    return physical_rows<T>(r, n);
}

} // namespace detail



/// \brief The compile-time stride of an operand whose stride is known at
/// run time only: the default of the stride template parameters of the
/// compile-time solvers.
///
/// Every entry point of TiledLUppSolverStatic, CooperativeLUppSolverStatic
/// and ArrowheadLUppSolverStatic takes the element stride of each strided
/// operand twice: an int function argument, X_stride, and an int template
/// parameter, X_static_stride, dynamic_stride by default. These template
/// parameters come after the others, but for the deduced types of the
/// callables, in the order of the function arguments. With
/// dynamic_stride, the solver reads the function argument. With a
/// positive value, 1 in AoS or 32 in AoSoA by blocks of 32 interleaved
/// systems for instance, the solver uses that value and ignores the
/// function argument: it runs as if the caller had passed the constant,
/// which the compilers fold wherever they inline the solver, everywhere on
/// GPU, and the out-of-line parts of ArrowheadLUpp (the second pass,
/// streamed_out_of_line) receive it as a template argument. A stride
/// known only at run time, the number of systems in SoA, stays a
/// function argument. Under internal residency the stride of the operand
/// is ignored either way.
inline constexpr int dynamic_stride = -1;

namespace detail {

/// \brief The stride an entry point uses: the compile-time stride when it
/// is known, the runtime one otherwise (see dynamic_stride).
/// \tparam s compile-time stride: dynamic_stride or positive
/// \param[in] runtime the stride given as a function argument
/// \return s, or runtime when s is dynamic_stride
template<int s>
[[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCE_INLINE_ALL constexpr int
stride_or(const int runtime) noexcept {
    static_assert(s == dynamic_stride || s >= 1,
                  "tdls: a compile-time stride must be tdls::dynamic_stride or positive");
    if constexpr (s == dynamic_stride)
        return runtime;
    else
        return s;
}

/// \brief Whether one of the strides of an entry point is known at
/// compile time: the entry point then runs its runtime-stride
/// instantiation with the constants.
/// \tparam s the compile-time strides
template<int... s>
inline constexpr bool any_static_stride = ((s != dynamic_stride) || ...);

} // namespace detail



/// \brief Memory layout of a dense matrix.
enum class MatrixLayout {
    RowMajor, ///< Element (r, c) at flat index r * N + c, the TFEL convention.
    ColMajor  ///< Element (r, c) at flat index c * N + r.
};



} // namespace tdls



#endif // TDLS_SOLVERS_OPTIONS_HPP
