#ifndef TDLS_CORE_MACROS_HPP
#define TDLS_CORE_MACROS_HPP



/// \file
/// \brief Toolchain detection and portability macros.
/// \author Tristan Chenaille
///
/// Targeted toolchains: plain CPU (gcc, clang, MSVC), CUDA (nvcc, clang),
/// HIP (hipcc/amdclang++), SYCL (icpx, AdaptiveCpp), stdpar (nvc++),
/// OpenMP host/target, Kokkos (through its backend compiler). Only CUDA and
/// HIP require function-space decorations; single-source models (SYCL,
/// stdpar, OpenMP target) decide device callability when kernels are
/// outlined, so no decoration is emitted there. OpenMP-target / OpenACC
/// `declare target` / `routine seq` wrappers belong to the caller's
/// dispatch layer, never to the solvers.
///
/// Every macro is `#ifndef`-guarded: each can be overridden from the
/// command line (e.g. `-DTDLS_FORCEINLINE=inline`) without editing headers.



/// \def TDLS_HOST_DEVICE
/// \brief `__host__ __device__` under CUDA/HIP, empty elsewhere.
///
/// Marks every solver function as callable from host code and from inside
/// device kernels.

#ifndef TDLS_HOST_DEVICE
#if defined(__CUDACC__) || defined(__CUDA__) || defined(__HIPCC__) || defined(__HIP__)
#define TDLS_HOST_DEVICE __host__ __device__
#else
#define TDLS_HOST_DEVICE
#endif
#endif



/// \def TDLS_FORCEINLINE
/// \brief `__forceinline__` under CUDA/HIP, plain `inline` elsewhere.
///
/// On GPU backends, forced inlining is the guard that keeps register tiles
/// alive: an out-of-line call would demote them to slow local memory. On
/// CPU a register spill to the stack is cheap and the heuristic inliner
/// makes better calls than a blanket `always_inline` (which also bloats
/// debug builds and compile times), so a plain `inline` hint is kept there.

#ifndef TDLS_FORCEINLINE
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define TDLS_FORCEINLINE __forceinline__
#else
#define TDLS_FORCEINLINE inline
#endif
#endif



/// \def TDLS_RESTRICT
/// \brief Non-aliasing pointer qualifier (`__restrict__`; `__restrict` on MSVC).

#ifndef TDLS_RESTRICT
#if defined(_MSC_VER) && !defined(__clang__)
#define TDLS_RESTRICT __restrict
#else
#define TDLS_RESTRICT __restrict__
#endif
#endif



/// \def TDLS_UNROLL_FORCE
/// \brief Full-unroll pragma in the local compiler dialect.
///
/// Dialect only: which loops carry it is a compile-time Config knob of each
/// solver, applied through a two-branch `if constexpr` whose no-unroll
/// branch carries no pragma at all. The GCC dialect needs an explicit
/// bound; 64 covers every annotated loop (all bounded by the tile size or
/// by the system dimension).

#ifndef TDLS_UNROLL_FORCE
#if defined(__CUDACC__) || defined(__CUDA__) || defined(__HIPCC__) || defined(__HIP__) ||          \
    defined(__NVCOMPILER) || defined(__clang__)
#define TDLS_UNROLL_FORCE _Pragma("unroll")
#elif defined(__GNUC__)
#define TDLS_UNROLL_FORCE _Pragma("GCC unroll 64")
#else
#define TDLS_UNROLL_FORCE
#endif
#endif



#endif // TDLS_CORE_MACROS_HPP
