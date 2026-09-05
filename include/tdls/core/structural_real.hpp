#ifndef TDLS_CORE_STRUCTURAL_REAL_HPP
#define TDLS_CORE_STRUCTURAL_REAL_HPP



/// \file
/// \brief Exact floating-point value admissible as a template argument.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The solver configurations travel as class-type non-type template
/// parameters, which C++20 restricts to structural types: every member
/// must itself be structural. Floating-point members only became
/// structural with P1907R1, a late C++20 addition whose support lags
/// behind the rest of the standard (GCC 11, MSVC 19.29, and the EDG
/// frontend of nvcc only from CUDA 13.0). Integer members, on the other
/// hand, are accepted by every compiler implementing class-type
/// template parameters at all (GCC 10, Clang 12, MSVC 19.28, nvcc 12.0).
///
/// StructuralReal therefore stores a floating-point value as an odd
/// integer mantissa and a power-of-two exponent, m * 2^e. Every finite
/// float or double value is represented exactly (53 significant bits fit
/// the 63 of a signed 64-bit mantissa), and so is a long double value
/// within those 63 bits, which every double literal is: a configuration
/// threshold round-trips bit for bit and the solvers see exactly the
/// value written by the caller.
/// The representation is canonical (odd mantissa, zero as (0, 0)), so
/// two configurations built from equal values name the same solver
/// instantiation, as they would with plain floating-point members.
/// Both directions are constant-evaluated: the solvers read the
/// thresholds into `static constexpr` scalars at instantiation and no
/// code of this header reaches a kernel.



#include <type_traits>

#include <tdls/core/macros.hpp>



namespace tdls {



/// \brief Floating-point value stored as an odd integer mantissa and a
/// power-of-two exponent, admissible as a template argument on every
/// compiler supporting class-type non-type template parameters.
///
/// Built implicitly from a value of the scalar type and converted back
/// implicitly, so a designated initializer such as
/// `.oot_threshold = 1e-10` needs no wrapper. Only finite values whose
/// odd mantissa fits 63 bits are representable: every float and double,
/// and the long double values built from double literals. A NaN, an
/// infinity or a wider long double mantissa is stored as a sentinel
/// that is_finite() reports, and the solvers reject at compile time.
///
/// The member-wise constructor exists for the CUDA toolchain: when nvcc
/// re-emits a translation unit for the host compiler, it spells every
/// class-type template argument as a brace list of its members, nested
/// classes included, so `{mantissa, exponent}` must be an accepted
/// initialization of this type.
/// \tparam T scalar type (float, double or long double)
template<typename T>
struct StructuralReal {

    static_assert(std::is_floating_point_v<T>, "tdls::StructuralReal: T must be floating-point");

    long long mantissa = 0; ///< odd integer mantissa, 0 for the value zero
    int exponent       = 0; ///< power-of-two exponent: value = mantissa * 2^exponent

    /// \brief The value zero.
    constexpr StructuralReal() noexcept = default;

    /// \brief Member-wise form, m * 2^e taken as is (no normalization).
    ///
    /// Required by the host-side re-emission of nvcc (see the class
    /// documentation); also the natural way to write an exact power of
    /// two. Two configurations only name the same solver instantiation
    /// when their members are equal, so a non-canonical mantissa (even,
    /// or with a different exponent) names a distinct instantiation of
    /// the same numerical value.
    /// \param[in] m integer mantissa
    /// \param[in] e power-of-two exponent
    TDLS_HOST_DEVICE constexpr StructuralReal(const long long m, const int e) noexcept
        : mantissa(m), exponent(e) {
    }

    /// \brief Exact decomposition of a finite value into m * 2^e.
    ///
    /// The magnitude is first halved below 2^62 (exact on a normal
    /// number), then doubled until integral (exact, and stopped by the
    /// sentinel below 2^63), and the integer mantissa is finally reduced
    /// to its odd part. Signed zeros both map to (0, 0): thresholds are
    /// magnitudes, the sign of zero carries nothing.
    /// \param[in] v the value to store
    TDLS_HOST_DEVICE constexpr StructuralReal(const T v) noexcept {
        if (v == T(0)) return;
        // A NaN differs from itself; an infinity is the only non-zero
        // value unchanged by halving (zero was excluded above). Neither
        // test produces a NaN, which constant evaluation would reject,
        // and neither needs a library call, which nvcc would reject in
        // this host-and-device constructor.
        if (v != v || v * T(0.5) == v) {
            // Non-finite sentinel: no finite value yields a zero mantissa
            // with a non-zero exponent.
            mantissa = 0;
            exponent = 1;
            return;
        }
        constexpr T big = T(1LL << 62);
        T a             = v;
        int e           = 0;
        while (a >= big || a <= -big) {
            a /= T(2);
            ++e;
        }
        while (T(static_cast<long long>(a)) != a) {
            // A mantissa wider than 63 bits (long double): the sentinel.
            if (a >= big || a <= -big) {
                mantissa = 0;
                exponent = 1;
                return;
            }
            a *= T(2);
            --e;
        }
        long long m = static_cast<long long>(a);
        while ((m & 1LL) == 0) {
            m /= 2;
            ++e;
        }
        mantissa = m;
        exponent = e;
    }

    /// \brief Whether the stored value is representable (a NaN, an
    /// infinity or a mantissa wider than 63 bits given to the constructor
    /// is stored as a sentinel instead).
    /// \return true for every representable value, zero included
    [[nodiscard]] TDLS_HOST_DEVICE constexpr bool is_finite() const noexcept {
        return !(mantissa == 0 && exponent != 0);
    }

    /// \brief The stored value, reconstructed exactly.
    ///
    /// Scaling by a power of two one step at a time is exact at every
    /// step: intermediate values carry at least as much precision as
    /// the representable end value. The non-finite sentinel converts
    /// to zero; the solvers reject it before any conversion.
    /// \return mantissa * 2^exponent
    [[nodiscard]] TDLS_HOST_DEVICE constexpr operator T() const noexcept {
        T r = T(mantissa);
        for (int e = exponent; e > 0; --e)
            r *= T(2);
        for (int e = exponent; e < 0; ++e)
            r /= T(2);
        return r;
    }
};



} // namespace tdls



#endif // TDLS_CORE_STRUCTURAL_REAL_HPP
