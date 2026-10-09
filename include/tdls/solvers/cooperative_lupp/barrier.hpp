#ifndef TDLS_SOLVERS_COOPERATIVE_LUPP_BARRIER_HPP
#define TDLS_SOLVERS_COOPERATIVE_LUPP_BARRIER_HPP



/// \file
/// \brief Default barrier of the CooperativeLUpp solver family.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The threads of a group synchronize through a barrier provided by the
/// caller: any callable taking no argument. This header defines the one
/// the solvers default to, valid only when one thread solves a system,
/// and the trait that makes the entry points noexcept exactly when the
/// barrier is.



#include <utility>

#include <tdls/core/macros.hpp>



namespace tdls {



/// \brief Barrier that does nothing: the default barrier of the
/// CooperativeLUpp entry points, valid only when one thread solves a
/// system.
struct NoSync {
    /// \brief Does nothing.
    TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void operator()() const noexcept {
    }
};

namespace detail {

/// \brief Whether a call of the barrier is noexcept, the condition of the
/// noexcept specification of every entry point. Spelled as a
/// noexcept-expression rather than through std::is_nothrow_invocable_v,
/// whose libstdc++ implementation evaluates the call inside a host
/// function, which nvcc 12.0 rejects for a device lambda.
/// \tparam Sync callable type of the barrier
template<typename Sync>
inline constexpr bool nothrow_sync = noexcept(std::declval<Sync&>()());

} // namespace detail



} // namespace tdls



#endif // TDLS_SOLVERS_COOPERATIVE_LUPP_BARRIER_HPP
