#ifndef TDLS_CORE_MACROS_HPP
#define TDLS_CORE_MACROS_HPP



/// \file
/// \brief Toolchain detection and portability macros.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
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
/// command line (e.g. `-DTDLS_FORCE_INLINE_GPU=inline`) without editing
/// headers.

#include <cstddef>

#include <tdls/core/target.hpp>



/// \def TDLS_HOST_DEVICE
/// \brief `__host__ __device__` under CUDA, the equivalent attributes
/// under HIP, empty elsewhere.
///
/// Marks every solver function as callable from host code and from inside
/// device kernels. Under CUDA the `__host__` / `__device__` macros are
/// always predefined by the toolchain; under HIP they only exist once
/// `hip/hip_runtime.h` has been included, which this library does not
/// require, so the HIP branch spells the attributes those macros expand
/// to and works whatever the include order.

#ifndef TDLS_HOST_DEVICE
#if defined(TDLS_DETAIL_LANG_HIP)
#define TDLS_HOST_DEVICE __attribute__((host)) __attribute__((device))
#elif defined(TDLS_DETAIL_FRONTEND_CUDA)
#define TDLS_HOST_DEVICE __host__ __device__
#else
#define TDLS_HOST_DEVICE
#endif
#endif



/// \def TDLS_FORCE_INLINE_GPU
/// \brief Inlining forced on GPU only: `__forceinline__` under CUDA, the
/// equivalent attribute under HIP (see TDLS_HOST_DEVICE for why) and in the
/// device pass of SYCL (`__SYCL_DEVICE_ONLY__`: icpx/DPC++ for Intel GPUs,
/// and its NVIDIA and AMD targets), plain `inline` elsewhere.
///
/// On GPU backends, forced inlining is the guard that keeps register tiles
/// alive: an out-of-line call would demote them to slow local memory. On
/// CPU a register spill to the stack is cheap and the heuristic inliner
/// makes better calls than a blanket `always_inline` (which also bloats
/// debug builds and compile times), so a plain `inline` hint is kept there.
/// Its pair TDLS_FORCE_INLINE_ALL forces the inlining on CPU too.

#ifndef TDLS_FORCE_INLINE_GPU
#if defined(TDLS_DETAIL_LANG_HIP)
#define TDLS_FORCE_INLINE_GPU inline __attribute__((always_inline))
#elif defined(TDLS_DETAIL_FRONTEND_CUDA)
#define TDLS_FORCE_INLINE_GPU __forceinline__
#elif defined(TDLS_DETAIL_LANG_SYCL_DEVICE)
#define TDLS_FORCE_INLINE_GPU inline __attribute__((always_inline))
#else
#define TDLS_FORCE_INLINE_GPU inline
#endif
#endif



/// \def TDLS_FORCE_INLINE_ALL
/// \brief Inlining forced on every backend, CPU included:
/// TDLS_FORCE_INLINE_GPU on GPU, `__forceinline` under MSVC,
/// `always_inline` under the GNU dialect, plain `inline` elsewhere.
///
/// Reserved for the few helpers whose body was factored out of a solver
/// routine: inlined, they leave its generated code as it was, while a call
/// the CPU inliner may decline costs time (see
/// TiledLUppSolverStatic::permute_cycles).

#ifndef TDLS_FORCE_INLINE_ALL
#if defined(TDLS_DETAIL_LANG_HIP) || defined(TDLS_DETAIL_FRONTEND_CUDA)
#define TDLS_FORCE_INLINE_ALL TDLS_FORCE_INLINE_GPU
#elif defined(TDLS_DETAIL_COMPILER_MSVC)
#define TDLS_FORCE_INLINE_ALL __forceinline
#elif defined(TDLS_DETAIL_COMPILER_GNU_ATTRIBUTES)
#define TDLS_FORCE_INLINE_ALL inline __attribute__((always_inline))
#else
#define TDLS_FORCE_INLINE_ALL inline
#endif
#endif



/// \def TDLS_NO_INLINE_GPU
/// \brief Keeps a function out of line on GPU only: `noinline` under CUDA,
/// HIP and in the device pass of SYCL, plain `inline` elsewhere; the pair of
/// TDLS_FORCE_INLINE_GPU.
///
/// For a function that the GPU inliner would merge into a large caller,
/// where the merged function gets a worse register allocation than the two
/// apart (see ArrowheadLUppConfig::streamed_out_of_line). On CPU the
/// inliner is left free.

#ifndef TDLS_NO_INLINE_GPU
#if defined(TDLS_DETAIL_LANG_HIP)
#define TDLS_NO_INLINE_GPU __attribute__((noinline))
#elif defined(TDLS_DETAIL_FRONTEND_CUDA)
#define TDLS_NO_INLINE_GPU __noinline__
#elif defined(TDLS_DETAIL_LANG_SYCL_DEVICE)
#define TDLS_NO_INLINE_GPU __attribute__((noinline))
#else
#define TDLS_NO_INLINE_GPU inline
#endif
#endif



/// \def TDLS_NO_INLINE
/// \brief Keeps a function out of line on every backend.
///
/// For the rare paths of a solver: an inlined rare path would share the
/// registers of the common one and could push it to local memory on GPU
/// (see the second pass of ArrowheadLUppSolverStatic).

#ifndef TDLS_NO_INLINE
#if defined(TDLS_DETAIL_FRONTEND_CUDA)
#define TDLS_NO_INLINE __noinline__
#elif defined(TDLS_DETAIL_COMPILER_MSVC)
#define TDLS_NO_INLINE __declspec(noinline)
#elif defined(TDLS_DETAIL_COMPILER_GNU_ATTRIBUTES)
#define TDLS_NO_INLINE __attribute__((noinline))
#else
#define TDLS_NO_INLINE
#endif
#endif



/// \def TDLS_RESTRICT
/// \brief Non-aliasing pointer qualifier (`__restrict__`; `__restrict` on MSVC;
/// nothing in the code that nvc++ generates).
///
/// From -O2 on, nvc++ 25.11 fuses loops that depend on each other. It draws
/// that right from the qualifier of the pointers that the solvers pass to
/// one another, once inlined. Its host code then returns wrong solutions:
/// the physical row interchanges of TiledLUpp, the blocks of ArrowheadLUpp
/// eliminated in registers, ArrowheadLUpp in its stored and streamed
/// solves. It also stops on an internal error on some TiledLUpp
/// configurations. Without the qualifier, its code is right. Measured on
/// the stored solve of ArrowheadLUpp: the same speed through -stdpar=gpu on
/// an RTX 5080 and through -stdpar=multicore on 12 CPU threads, 0 to 5 %
/// slower on one CPU core. The device pass of nvcc keeps the qualifier,
/// whatever its host compiler.

#ifndef TDLS_RESTRICT
#if defined(TDLS_DETAIL_COMPILER_MSVC)
#define TDLS_RESTRICT __restrict
#elif defined(TDLS_DETAIL_CODEGEN_NVCXX)
#define TDLS_RESTRICT
#else
#define TDLS_RESTRICT __restrict__
#endif
#endif



/// \def TDLS_STRIDED
/// \brief Offset of entry i of a strided object, i times stride s, in
/// 64-bit arithmetic (std::size_t); i and s are nonnegative ints.
///
/// The addressing macros of the solvers use it for every element of an
/// object in remote memory: the offset of the last element of a batch in
/// SoA, its index times the number of systems, may exceed 2^32 on a large
/// GPU, where a 32-bit product would wrap around silently. Measured
/// against the 32-bit product: same results bit for bit, no perceptible
/// cost (RTX 5080, CPU).
///
/// In the code that nvc++ generates, i is extended with its sign rather
/// than with zeros. From
/// -O2 on, nvc++ 25.11 derives the addresses of some loops from an index
/// that the loop never forms, a negative one, extended as this macro
/// extends i. Extended with zeros, every address of such a loop lies 2^32
/// strides beyond the right one: the runtime solvers of TiledLUpp and
/// ArrowheadLUpp read outside their operands and fault. Extended with its
/// sign, every address is right. Both extensions give the same offset for
/// every index the solvers form.

#ifndef TDLS_STRIDED
#if defined(TDLS_DETAIL_CODEGEN_NVCXX)
#define TDLS_STRIDED(i, s)                                                                         \
    (static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i)) * static_cast<unsigned>(s))
#else
#define TDLS_STRIDED(i, s)                                                                         \
    (static_cast<std::size_t>(static_cast<unsigned>(i)) * static_cast<unsigned>(s))
#endif
#endif



/// \def TDLS_STRIDED_WIDE
/// \brief Offset of entry i of a strided object, i times stride s, for a
/// 64-bit stride s (std::size_t); i is a nonnegative int.
///
/// The stride between the columns of a multi right-hand-side block: a
/// column of a batch in SoA lies the dimension times the number of
/// systems away from the previous one, a product that may exceed 2^31
/// where the strides of the entries do not. In the code that nvc++
/// generates, i is extended with its sign, as in TDLS_STRIDED.

#ifndef TDLS_STRIDED_WIDE
#if defined(TDLS_DETAIL_CODEGEN_NVCXX)
#define TDLS_STRIDED_WIDE(i, s) (static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i)) * (s))
#else
#define TDLS_STRIDED_WIDE(i, s) (static_cast<std::size_t>(static_cast<unsigned>(i)) * (s))
#endif
#endif



/// \def TDLS_UNROLL_FORCE
/// \brief Full-unroll pragma in the local compiler dialect.
///
/// Dialect only: which loops carry it is a compile-time Config knob of each
/// solver, applied through a two-branch `if constexpr` whose no-unroll
/// branch carries no pragma at all. The GCC dialect needs an explicit
/// unrolling factor. 64 fully unrolls the loops bounded by the tile size
/// and those bounded by a system dimension up to 64; beyond, GCC unrolls
/// them partially.

#ifndef TDLS_UNROLL_FORCE
#if defined(TDLS_DETAIL_FRONTEND_CUDA) || defined(TDLS_DETAIL_LANG_HIP) ||                         \
    defined(TDLS_DETAIL_COMPILER_NVCXX) || defined(TDLS_DETAIL_COMPILER_CLANG)
#define TDLS_UNROLL_FORCE _Pragma("unroll")
#elif defined(TDLS_DETAIL_COMPILER_GCC)
#define TDLS_UNROLL_FORCE _Pragma("GCC unroll 64")
#else
#define TDLS_UNROLL_FORCE
#endif
#endif



/// \def TDLS_EXEC_CHECK_DISABLE
/// \brief Lifts the execution-space check of nvcc on the next function
/// template; empty elsewhere.
///
/// The CooperativeLUpp solvers call the barrier of the caller, a device
/// function on GPU, from host-device function templates. nvcc checks the
/// execution space of that call in the template itself and rejects it
/// when the instantiation starts from a non-template kernel, although
/// every instantiation actually used runs on the device. The pragma lifts
/// the check for that one function, the practice of Thrust and CUB for
/// user functors. clang, for CUDA and HIP alike, only checks the
/// functions it emits and needs nothing.

#ifndef TDLS_EXEC_CHECK_DISABLE
#if defined(TDLS_DETAIL_FRONTEND_CUDA) && !defined(TDLS_DETAIL_COMPILER_CLANG) &&                  \
    !defined(TDLS_DETAIL_COMPILER_NVCXX)
#define TDLS_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#else
#define TDLS_EXEC_CHECK_DISABLE
#endif
#endif



#endif // TDLS_CORE_MACROS_HPP
