#ifndef TDLS_CORE_GROUP_HPP
#define TDLS_CORE_GROUP_HPP



/// \file
/// \brief Barriers of the groups of threads that share a system: the
/// barrier deduced from the compilation target, the barrier of a single
/// thread, and the resolution of the barrier argument of the solvers.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A cooperative solver takes the barrier of its group as its last
/// argument, `sync`. Three kinds of value are accepted:
///   - tdls::AutoSync, the default: the solver deduces the barrier from
///     the target the code is compiled for, as described below;
///   - tdls::NoSync: no barrier at all, valid only for a group of one
///     thread, a contract checked at compile time;
///   - any other callable taking no argument: the barrier of the caller,
///     used as is, without any check and at no extra cost.
///
/// **The deduced barrier.** With one thread per system it does nothing,
/// and the call stays a constant expression. With several threads, the
/// group is identified by its workspace, which every thread of the group
/// passes and no other group shares:
///   - on a GPU, the threads of a group are lanes of one warp (NVIDIA) or
///     one wavefront (AMD). On entry, the lanes that run together match
///     the address of their workspace, and the group must find all its
///     threads among them. An incomplete group stops the program with a
///     message, before any data is exchanged: a wrong placement of the
///     threads never yields a wrong result nor a deadlock. The barrier
///     then involves exactly the lanes of the group.
///   - on a CPU, the threads of a group are any threads running
///     concurrently: OpenMP, std::thread, Kokkos team threads and so on.
///     The barrier is a counter and a flag held in the last two elements
///     of the workspace, which must be zero before the first call and
///     stay zero between calls. A group waiting for more than a second
///     prints a diagnostic once and keeps waiting: a thread may be late
///     for a legitimate reason, so the wait is never cut short.
///
/// **Targets.** The target is a property of the compilation pass, decided
/// by the preprocessor in one place below:
///   - NVIDIA through the CUDA API: the device pass of nvcc and of clang
///     in CUDA mode, and the device side of nvc++ with -cuda or
///     -stdpar=gpu (nvc++ compiles host and device in one pass and picks
///     the side with NV_IF_TARGET);
///   - NVIDIA through the clang builtins: the other clang device passes
///     for NVPTX, OpenMP offloading and SYCL;
///   - AMD through the HIP API (ROCm 7.0 or newer, where the warp
///     synchronous functions are enabled by default);
///   - AMD through the clang builtins: the other clang device passes for
///     AMDGCN, OpenMP offloading and SYCL (clang 19 or newer);
///   - the host, for every other pass.
/// A few passes offer no warp primitive callable without a handle passed
/// by the kernel: nvc++ OpenMP or OpenACC offloading without -cuda, its
/// OpenACC backend of the parallel algorithms, the generic mode of
/// AdaptiveCpp, GCC offloading and SYCL on SPIR-V devices. There the
/// deduced barrier accepts only a group of one thread known at compile
/// time, and the compiler asks for an explicit barrier otherwise.
///
/// The GPU checks follow the warp synchronous primitives of each vendor:
/// __match_any_sync on the active lanes, __syncwarp on the lanes of the
/// group. AMD executes the lanes of a wavefront in lockstep, so its
/// barrier is a wavefront fence around a scheduling barrier, the
/// __syncwarp of HIP itself. On NVIDIA, sm_70 or newer is required.



#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <utility>

#include <tdls/core/macros.hpp>

#if defined(__NVCOMPILER)
#include <nv/target>
#endif

// The warp functions of HIP come with its runtime header, which a HIP
// translation unit may include after this one.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif



/* =========================================================================
   Target of the compilation pass.
   The order matters. clang defines __CUDA_ARCH__ in the OpenMP device
   pass for AMDGCN up to clang 18, and DPC++ does not define it for NVPTX:
   the AMDGCN and NVPTX macros are tested first.
   ========================================================================= */

#if defined(__NVCOMPILER)
#if defined(__NVCOMPILER_STDPAR_OPENACC_GPU)
/// No deduced barrier in this pass; the reason, appended to the diagnostic.
#define TDLS_DETAIL_GROUP_NONE "the OpenACC backend of nvc++ for the parallel algorithms has none"
#elif defined(_NVHPC_CUDA)
/// nvc++ with CUDA: NV_IF_TARGET selects the device or the host side.
#define TDLS_DETAIL_GROUP_NVCXX
#elif defined(__NVCOMPILER_OPENMP_GPU) || defined(__NVCOMPILER_OPENACC_GPU)
/// No deduced barrier in this pass; the reason, appended to the diagnostic.
#define TDLS_DETAIL_GROUP_NONE "nvc++ offers it to OpenMP and OpenACC offloading with -cuda"
#else
/// The host.
#define TDLS_DETAIL_GROUP_HOST
#endif
#elif defined(ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP) || defined(HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP)
/// No deduced barrier in this pass; the reason, appended to the diagnostic.
#define TDLS_DETAIL_GROUP_NONE "the generic mode of AdaptiveCpp has none; its cuda and hip modes do"
#elif defined(__AMDGCN__) && defined(__clang__)
#if defined(__HIP_DEVICE_COMPILE__)
/// AMD through the HIP API.
#define TDLS_DETAIL_GROUP_HIP
#else
/// AMD through the clang builtins.
#define TDLS_DETAIL_GROUP_AMDGCN
#endif
#elif defined(__CUDA_ARCH__) && (defined(__CUDACC__) || defined(__CUDA__))
/// NVIDIA through the CUDA API.
#define TDLS_DETAIL_GROUP_CUDA
#elif defined(__NVPTX__) && defined(__clang__)
/// NVIDIA through the clang builtins.
#define TDLS_DETAIL_GROUP_NVPTX
#elif defined(__nvptx__) || defined(__AMDGCN__) || defined(__amdgcn__)
/// No deduced barrier in this pass; the reason, appended to the diagnostic.
#define TDLS_DETAIL_GROUP_NONE "GCC offloading has none"
#elif defined(__SYCL_DEVICE_ONLY__)
/// No deduced barrier in this pass; the reason, appended to the diagnostic.
#define TDLS_DETAIL_GROUP_NONE "SYCL devices other than NVIDIA and AMD have none yet"
#else
/// The host.
#define TDLS_DETAIL_GROUP_HOST
#endif

/// Statement list expanded as is.
#define TDLS_DETAIL_GROUP_EXPAND(...) __VA_ARGS__

// The arguments of the two selection macros are parenthesized statement
// lists, as for NV_IF_TARGET. TDLS_DETAIL_GROUP_LANES qualifies the lane
// functions: device functions of the CUDA and HIP dialects, host-device
// functions of nvc++, which select their body with NV_IF_TARGET, and
// plain functions of the device passes of the other models.
#if defined(TDLS_DETAIL_GROUP_NVCXX)
/// Keeps a statement list on the device side only.
#define TDLS_DETAIL_GROUP_ON_DEVICE(code) NV_IF_TARGET(NV_IS_DEVICE, code)
/// Picks the statement list of the side of the pass.
#define TDLS_DETAIL_GROUP_DISPATCH(device, host) NV_IF_TARGET(NV_IS_DEVICE, device, host)
/// Qualifiers of the lane functions.
#define TDLS_DETAIL_GROUP_LANES TDLS_HOST_DEVICE TDLS_FORCEINLINE
#elif defined(TDLS_DETAIL_GROUP_CUDA) || defined(TDLS_DETAIL_GROUP_HIP)
/// Keeps a statement list on the device side only.
#define TDLS_DETAIL_GROUP_ON_DEVICE(code)        TDLS_DETAIL_GROUP_EXPAND code
/// Picks the statement list of the side of the pass.
#define TDLS_DETAIL_GROUP_DISPATCH(device, host) TDLS_DETAIL_GROUP_EXPAND device
/// Qualifiers of the lane functions.
#define TDLS_DETAIL_GROUP_LANES                  __device__ TDLS_FORCEINLINE
#elif defined(TDLS_DETAIL_GROUP_NVPTX) || defined(TDLS_DETAIL_GROUP_AMDGCN)
/// Picks the statement list of the side of the pass.
#define TDLS_DETAIL_GROUP_DISPATCH(device, host) TDLS_DETAIL_GROUP_EXPAND device
/// Qualifiers of the lane functions.
#define TDLS_DETAIL_GROUP_LANES                  inline
#elif defined(TDLS_DETAIL_GROUP_HOST)
/// Picks the statement list of the side of the pass.
#define TDLS_DETAIL_GROUP_DISPATCH(device, host) TDLS_DETAIL_GROUP_EXPAND host
#else
/// No side to pick: the resolution of the barrier refuses the pass.
#define TDLS_DETAIL_GROUP_DISPATCH(device, host)
#endif



namespace tdls {



/// \brief Barrier argument of the cooperative solvers that asks them to
/// deduce the barrier from the compilation target: the default.
struct AutoSync {};

/// \brief Barrier that does nothing, valid only when one thread solves a
/// system.
struct NoSync {
    /// \brief Does nothing.
    TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void operator()() const noexcept {
    }
};

namespace detail {

/// \brief Whether the barrier argument asks for the deduced barrier.
/// \tparam Sync type of the barrier argument
template<typename Sync>
inline constexpr bool is_auto_sync = std::is_same_v<std::remove_cvref_t<Sync>, AutoSync>;

/// \brief Whether a call of the barrier is noexcept, the condition of the
/// noexcept specification of every entry point. Spelled as a
/// noexcept-expression rather than through std::is_nothrow_invocable_v,
/// whose libstdc++ implementation evaluates the call inside a host
/// function, which nvcc 12.0 rejects for a device lambda. The deduced
/// barrier never throws.
/// \tparam Sync callable type of the barrier
template<typename Sync, bool = is_auto_sync<Sync>>
inline constexpr bool nothrow_sync = noexcept(std::declval<Sync&>()());

/// \brief The deduced barrier never throws.
/// \tparam Sync tdls::AutoSync, possibly qualified
template<typename Sync>
inline constexpr bool nothrow_sync<Sync, true> = true;

/// \brief Whether the compilation pass offers the deduced barrier to a
/// group of several threads. Otherwise TDLS_DETAIL_GROUP_NONE holds the
/// reason, appended to the diagnostic.
#if defined(TDLS_DETAIL_GROUP_NONE)
inline constexpr bool group_barrier_available = false;
#else
inline constexpr bool group_barrier_available = true;
/// No reason: the pass offers the deduced barrier.
#define TDLS_DETAIL_GROUP_NONE ""
#endif

/// \brief Largest group the warp of the pass can hold, checked at compile
/// time when the size of the group is (0: no compile-time bound).
#if defined(TDLS_DETAIL_GROUP_CUDA) || defined(TDLS_DETAIL_GROUP_NVPTX)
inline constexpr int group_lanes = 32;
#elif defined(TDLS_DETAIL_GROUP_HIP) || defined(TDLS_DETAIL_GROUP_AMDGCN)
inline constexpr int group_lanes = 64;
#else
inline constexpr int group_lanes = 0;
#endif

/// Message of an incomplete group, printed by its first lane. It takes no
/// argument: a formatted print would reserve a stack frame in local
/// memory for every kernel, even when it never runs.
#define TDLS_DETAIL_GROUP_INCOMPLETE                                                               \
    "TDLS: a group of threads is incomplete in its warp. The threads of a group must call the "    \
    "solver together, as lanes of one warp or wavefront.\n"

/* =========================================================================
   NVIDIA, CUDA API: nvcc, clang CUDA, nvc++ on the device side.
   ========================================================================= */

#if defined(TDLS_DETAIL_GROUP_CUDA) || defined(TDLS_DETAIL_GROUP_NVCXX)

/// \brief Lanes of the calling group: the active lanes sharing its key.
/// Stops the program when the group does not find all its threads.
/// \param[in] key     address identifying the group
/// \param[in] threads number of threads of the group
/// \return the lane mask of the group
TDLS_DETAIL_GROUP_LANES unsigned long long lanes_join(const void* key, const int threads) noexcept {
    unsigned long long lanes = 0;
    // clang-format off: a statement list as the argument of a macro
    TDLS_DETAIL_GROUP_ON_DEVICE((
        const unsigned mask =
            __match_any_sync(__activemask(), reinterpret_cast<unsigned long long>(key));
        const int found = __popc(mask);
        if (found != threads) {
            unsigned lane;
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
            if (lane == static_cast<unsigned>(__ffs(mask) - 1))
                printf(TDLS_DETAIL_GROUP_INCOMPLETE);
            __trap();
        }
        lanes = mask;))
    // clang-format on
    return lanes;
}

/// \brief Barrier of the lanes of a group.
/// \param[in] lanes lane mask of the group, from lanes_join
TDLS_DETAIL_GROUP_LANES void lanes_sync(const unsigned long long lanes) noexcept {
    TDLS_DETAIL_GROUP_ON_DEVICE((__syncwarp(static_cast<unsigned>(lanes));))
}

/* =========================================================================
   AMD, HIP API (ROCm 7.0 or newer). The lanes of a wavefront run in
   lockstep: the barrier only orders the memory operations, as the
   __syncwarp of HIP.
   ========================================================================= */

#elif defined(TDLS_DETAIL_GROUP_HIP)

/// \brief Lanes of the calling group: the active lanes sharing its key.
/// Stops the program when the group does not find all its threads.
/// \param[in] key     address identifying the group
/// \param[in] threads number of threads of the group
/// \return the lane mask of the group
TDLS_DETAIL_GROUP_LANES unsigned long long lanes_join(const void* key, const int threads) noexcept {
    const unsigned long long mask =
        __match_any_sync(__activemask(), reinterpret_cast<unsigned long long>(key));
    const int found = __popcll(mask);
    if (found != threads) {
        if (__lane_id() == static_cast<unsigned>(__builtin_ctzll(mask)))
            printf(TDLS_DETAIL_GROUP_INCOMPLETE);
        __builtin_trap();
    }
    return mask;
}

/// \brief Barrier of the lanes of a group: a wavefront fence around a
/// scheduling barrier.
TDLS_DETAIL_GROUP_LANES void lanes_sync(unsigned long long) noexcept {
    __syncwarp();
}

/* =========================================================================
   NVIDIA, clang builtins: OpenMP offloading and SYCL device passes. The
   active mask is read with the PTX instruction itself, as clang's CUDA
   headers do before __nvvm_activemask (LLVM 19).
   ========================================================================= */

#elif defined(TDLS_DETAIL_GROUP_NVPTX)

/// \brief Lanes of the calling group: the active lanes sharing its key.
/// Stops the program when the group does not find all its threads.
/// \param[in] key     address identifying the group
/// \param[in] threads number of threads of the group
/// \return the lane mask of the group
TDLS_DETAIL_GROUP_LANES unsigned long long lanes_join(const void* key, const int threads) noexcept {
    unsigned active;
    asm volatile("activemask.b32 %0;" : "=r"(active));
    const unsigned mask = __nvvm_match_any_sync_i64(
        active, static_cast<long long>(reinterpret_cast<unsigned long long>(key)));
    const int found = __builtin_popcount(mask);
    if (found != threads) {
#if !defined(__SYCL_DEVICE_ONLY__)
        if (__nvvm_read_ptx_sreg_laneid() == __builtin_ctz(mask))
            printf(TDLS_DETAIL_GROUP_INCOMPLETE);
#endif
        __builtin_trap();
    }
    return mask;
}

/// \brief Barrier of the lanes of a group.
/// \param[in] lanes lane mask of the group, from lanes_join
TDLS_DETAIL_GROUP_LANES void lanes_sync(const unsigned long long lanes) noexcept {
    __nvvm_bar_warp_sync(static_cast<unsigned>(lanes));
}

/* =========================================================================
   AMD, clang builtins: OpenMP offloading and SYCL device passes. The
   match is the loop of HIP's __match_any: the first lane still searching
   broadcasts its key, and the lanes holding it leave with their mask.
   ========================================================================= */

#elif defined(TDLS_DETAIL_GROUP_AMDGCN)

/// \brief Lanes of the calling group: the active lanes sharing its key.
/// Stops the program when the group does not find all its threads.
/// \param[in] key     address identifying the group
/// \param[in] threads number of threads of the group
/// \return the lane mask of the group
TDLS_DETAIL_GROUP_LANES unsigned long long lanes_join(const void* key, const int threads) noexcept {
    const unsigned long long own = reinterpret_cast<unsigned long long>(key);
    unsigned long long mask      = 0;
    bool done                    = false;
    while (__builtin_amdgcn_ballot_w64(!done) != 0) {
        if (!done) {
            const unsigned low  = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(own));
            const unsigned high = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(own >> 32));
            if (((static_cast<unsigned long long>(high) << 32) | low) == own) {
                mask = __builtin_amdgcn_ballot_w64(true);
                done = true;
            }
        }
    }
    const int found = __builtin_popcountll(mask);
    if (found != threads) {
#if !defined(__SYCL_DEVICE_ONLY__)
        const unsigned lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
        if (lane == static_cast<unsigned>(__builtin_ctzll(mask)))
            printf(TDLS_DETAIL_GROUP_INCOMPLETE);
#endif
        __builtin_trap();
    }
    return mask;
}

/// \brief Barrier of the lanes of a group: a wavefront fence around a
/// scheduling barrier.
TDLS_DETAIL_GROUP_LANES void lanes_sync(unsigned long long) noexcept {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

#endif

/* =========================================================================
   Host: a counter and a flag in the last two elements of the workspace.
   The flag alternates between 0 and 1. A thread reads it before
   arriving, so it cannot miss the flip of its own barrier; the last
   thread to arrive resets the counter, then flips the flag.
   ========================================================================= */

#if defined(TDLS_DETAIL_GROUP_HOST) || defined(TDLS_DETAIL_GROUP_NVCXX)

/// \brief Checks the two barrier elements of the workspace on entry: a
/// counter in [0, threads) and a flag of 0 or 1, the state the barrier
/// keeps between calls from a zero-initialized workspace.
/// \tparam T scalar type of the workspace
/// \param[in] slots   the two barrier elements of the workspace
/// \param[in] threads number of threads of the group
template<typename T>
void host_join(T* slots, const int threads) noexcept {
    static_assert(std::atomic_ref<T>::is_always_lock_free,
                  "TDLS: the deduced CPU barrier needs lock-free atomics on the scalar type; "
                  "pass a barrier (sync argument)");
    const T count = std::atomic_ref<T>(slots[0]).load(std::memory_order_relaxed);
    const T flag  = std::atomic_ref<T>(slots[1]).load(std::memory_order_relaxed);
    if (!(count >= T(0) && count < T(threads) && count == T(static_cast<int>(count)) &&
          (flag == T(0) || flag == T(1)))) {
        std::fprintf(stderr,
                     "TDLS: the workspace of a group of %d CPU threads was not zero before "
                     "its first use.\n",
                     threads);
        std::abort();
    }
}

/// \brief Barrier of a group of CPU threads.
/// \tparam T scalar type of the workspace
/// \param[in] slots   the two barrier elements of the workspace
/// \param[in] threads number of threads of the group
template<typename T>
void host_sync(T* slots, const int threads) noexcept {
    std::atomic_ref<T> count(slots[0]);
    std::atomic_ref<T> flag(slots[1]);
    const T seen = flag.load(std::memory_order_acquire);
    if (count.fetch_add(T(1), std::memory_order_acq_rel) == T(threads - 1)) {
        count.store(T(0), std::memory_order_relaxed);
        flag.store(seen == T(0) ? T(1) : T(0), std::memory_order_release);
        return;
    }
    // Spin first, then yield. The clock is read only after the spins, once
    // every 1024 iterations: the barriers of a running group never reach it.
    static std::atomic<bool> warned{false};
    std::chrono::steady_clock::time_point start{};
    for (unsigned spins = 0; flag.load(std::memory_order_acquire) == seen; ++spins) {
        if (spins < 1024) continue;
        std::this_thread::yield();
        if (spins % 1024 != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024)
            start = now;
        else if (now - start > std::chrono::seconds(1) && !warned.exchange(true))
            std::fprintf(stderr,
                         "TDLS: a group of %d CPU threads has waited for 1 s at a barrier. "
                         "The threads of a group must run concurrently: not in a "
                         "worksharing loop, nor in parallel STL or a SYCL kernel on the "
                         "CPU.\n",
                         threads);
    }
}

#endif

#undef TDLS_DETAIL_GROUP_INCOMPLETE

/// \brief The barrier deduced from the compilation target, for a group of
/// several threads. A group of one thread, possible with the runtime
/// solver, does nothing.
/// \tparam T scalar type of the workspace
template<typename T>
class DeducedSync {
  public:
    /// \brief Joins the group: the GPU check of its lanes, or the CPU check
    /// of its workspace.
    /// \param[in] slots   the two barrier elements of the workspace, whose
    ///            address also identifies the group
    /// \param[in] threads number of threads of the group
    TDLS_EXEC_CHECK_DISABLE
    TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr DeducedSync(T* slots, const int threads) noexcept
        : slots(slots), threads(threads) {
        if (threads > 1) {
            TDLS_DETAIL_GROUP_DISPATCH((lanes = lanes_join(slots, threads);),
                                       (host_join(slots, threads);))
        }
    }

    /// \brief Barrier of the group.
    TDLS_EXEC_CHECK_DISABLE
    TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void operator()() const noexcept {
        if (threads > 1) {
            TDLS_DETAIL_GROUP_DISPATCH((lanes_sync(lanes);), (host_sync(slots, threads);))
        }
    }

  private:
    T* slots;
    int threads;
    unsigned long long lanes = 0;
};

/// \brief Resolves the barrier argument of an entry point: the caller's
/// barrier as is, or the deduced barrier.
///
/// With a group size known at compile time, a group of one thread
/// resolves to tdls::NoSync and costs nothing. Otherwise the pass must
/// offer the deduced barrier, and a GPU warp must be able to hold the
/// group.
/// \tparam static_threads size of the group when known at compile time,
///         0 for the runtime solver
/// \tparam Sync           type of the barrier argument
/// \tparam T              scalar type of the workspace
/// \param[in] sync    barrier argument of the entry point
/// \param[in] slots   the two barrier elements of the workspace
/// \param[in] threads size of the group
/// \return a reference to sync, or the deduced barrier
template<int static_threads, typename Sync, typename T>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr decltype(auto)
resolve_sync(Sync& sync, [[maybe_unused]] T* slots, [[maybe_unused]] const int threads) noexcept {
    if constexpr (!is_auto_sync<Sync>) {
        return (sync);
    } else if constexpr (static_threads == 1) {
        return NoSync{};
    } else {
        static_assert(group_barrier_available,
                      "TDLS: this compilation target offers no deduced barrier for a group of "
                      "several threads; pass a barrier (sync argument), or tdls::NoSync with one "
                      "thread per system. " TDLS_DETAIL_GROUP_NONE);
        static_assert(group_lanes == 0 || static_threads <= group_lanes,
                      "TDLS: a group larger than a warp needs a barrier (sync argument)");
        return DeducedSync<T>(slots, threads);
    }
}

} // namespace detail



} // namespace tdls



#undef TDLS_DETAIL_GROUP_DISPATCH
#undef TDLS_DETAIL_GROUP_LANES
#undef TDLS_DETAIL_GROUP_ON_DEVICE
#undef TDLS_DETAIL_GROUP_EXPAND
#undef TDLS_DETAIL_GROUP_CUDA
#undef TDLS_DETAIL_GROUP_HIP
#undef TDLS_DETAIL_GROUP_NVPTX
#undef TDLS_DETAIL_GROUP_AMDGCN
#undef TDLS_DETAIL_GROUP_NVCXX
#undef TDLS_DETAIL_GROUP_HOST
#undef TDLS_DETAIL_GROUP_NONE



#endif // TDLS_CORE_GROUP_HPP
