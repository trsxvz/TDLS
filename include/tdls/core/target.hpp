#ifndef TDLS_CORE_TARGET_HPP
#define TDLS_CORE_TARGET_HPP



/// \file
/// \brief Detection of the compiler, of the language and of the side of the
/// compilation pass: the one place where TDLS identifies the toolchain and
/// the pass. Tests of a compiler version or of an architecture level stay
/// where they serve, on top of these macros.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every macro below is internal (TDLS_DETAIL_) and defined when its
/// condition holds; the headers that depend on the target include this
/// one and test these macros instead of those of the toolchains.
///
/// Compilers:
///   - TDLS_DETAIL_COMPILER_CLANG: clang and its derivatives (icpx,
///     amdclang++, clang-cl);
///   - TDLS_DETAIL_COMPILER_GCC: GCC, and the compilers that take its
///     pragmas without being clang (nvc++);
///   - TDLS_DETAIL_COMPILER_MSVC: MSVC itself, not clang-cl;
///   - TDLS_DETAIL_COMPILER_GNU_ATTRIBUTES: a compiler of the GNU
///     attributes (GCC, clang and their derivatives);
///   - TDLS_DETAIL_COMPILER_NVCC: nvcc;
///   - TDLS_DETAIL_COMPILER_NVCXX: nvc++;
///   - TDLS_DETAIL_CODEGEN_NVCXX: code that nvc++ generates, under nvc++
///     and in the host pass of nvcc over nvc++, not in its device pass (the
///     workarounds of nvc++ in core/macros.hpp);
///   - TDLS_DETAIL_COMPILER_HIP_CLANG: the clang of HIP (hipcc,
///     amdclang++, clang -x hip), the compiler of the HIP language on AMD.
///
/// Languages, for the function-space decorations:
///   - TDLS_DETAIL_LANG_HIP: HIP, on AMD (clang) or on NVIDIA (hipcc over
///     nvcc);
///   - TDLS_DETAIL_FRONTEND_CUDA: a CUDA front end (nvcc, clang -x cuda),
///     HIP over nvcc included;
///   - TDLS_DETAIL_LANG_CUDA: CUDA, when not HIP;
///   - TDLS_DETAIL_LANG_SYCL_DEVICE: the device pass of SYCL, whatever the
///     device.
///
/// Side of the compilation pass, exactly one of:
///   - TDLS_DETAIL_PASS_HOST: host code, run by the CPU;
///   - TDLS_DETAIL_PASS_CUDA: the device pass of the CUDA language on
///     NVIDIA (nvcc, clang);
///   - TDLS_DETAIL_PASS_NVPTX: an NVIDIA device pass outside the CUDA
///     language (OpenMP offloading of clang, DPC++ for NVIDIA);
///   - TDLS_DETAIL_PASS_AMDGCN: an AMD device pass (HIP, OpenMP
///     offloading, SYCL for AMD);
///   - TDLS_DETAIL_PASS_SPIRV: a SPIR-V device pass (SYCL, OpenMP
///     offloading): Intel GPUs, or any OpenCL device;
///   - TDLS_DETAIL_PASS_GCC_OFFLOAD: an offloading pass of GCC (nvptx,
///     amdgcn);
///   - TDLS_DETAIL_PASS_NVCXX_CUDA: nvc++ with CUDA (-cuda, -stdpar=gpu),
///     host and device code in one pass, which NV_IF_TARGET tells apart;
///   - TDLS_DETAIL_PASS_NVCXX_OFFLOAD: OpenMP or OpenACC offloading of
///     nvc++ without -cuda;
///   - TDLS_DETAIL_PASS_NVCXX_STDPAR_OPENACC: the parallel algorithms of
///     nvc++ through its OpenACC backend;
///   - TDLS_DETAIL_PASS_ACPP_GENERIC: the generic mode of AdaptiveCpp,
///     host and device code in one pass.
///
/// The order matters. clang defines __CUDA_ARCH__ in the OpenMP device
/// pass for AMDGCN up to clang 18, and DPC++ does not define it for NVPTX:
/// the AMDGCN and NVPTX macros are tested first. AdaptiveCpp defines
/// __NVPTX__ in the host pass of its CUDA mode too, and the host pass of
/// HIP defines __AMDGCN__, plus __SPIRV__ for amdgcnspirv: the device
/// macros count only outside the host passes of these languages. Every
/// other SPIR-V device pass, SYCL or OpenMP, follows. The generic mode of
/// AdaptiveCpp, recognized by the macro its compiler defines, comes last
/// among the device passes.
///
/// Hardware of the pass:
///   - TDLS_DETAIL_TARGET_GPU: 1 when the pass generates GPU code, every
///     side but the host, 0 otherwise. The single-pass compilers
///     (NVCXX_CUDA, ACPP_GENERIC) count as GPU, their code being meant for
///     it, and so does a SPIR-V pass, whatever its device. The defaults
///     that depend on the hardware (the elimination of the blocks of
///     ArrowheadLUpp) read it: a wrong guess costs speed, never results,
///     as they shape only the work of one call, never data kept between
///     calls.



#if defined(__clang__)
/// clang and its derivatives.
#define TDLS_DETAIL_COMPILER_CLANG
#endif
#if defined(__GNUC__) && !defined(__clang__)
/// GCC, and the compilers that take its pragmas without being clang.
#define TDLS_DETAIL_COMPILER_GCC
#endif
#if defined(_MSC_VER) && !defined(__clang__)
/// MSVC itself.
#define TDLS_DETAIL_COMPILER_MSVC
#endif
#if defined(__GNUC__) || defined(__clang__)
/// A compiler of the GNU attributes.
#define TDLS_DETAIL_COMPILER_GNU_ATTRIBUTES
#endif
#if defined(__NVCC__)
/// nvcc.
#define TDLS_DETAIL_COMPILER_NVCC
#endif
#if defined(__NVCOMPILER)
/// nvc++.
#define TDLS_DETAIL_COMPILER_NVCXX
#if !(defined(__NVCC__) && defined(__CUDA_ARCH__))
/// Code that nvc++ generates: not the device pass of nvcc.
#define TDLS_DETAIL_CODEGEN_NVCXX
#endif
#endif
#if defined(__HIP__)
/// The clang of HIP.
#define TDLS_DETAIL_COMPILER_HIP_CLANG
#endif

#if defined(__HIPCC__) || defined(__HIP__)
/// The HIP language.
#define TDLS_DETAIL_LANG_HIP
#endif
#if defined(__CUDACC__) || defined(__CUDA__)
/// A CUDA front end.
#define TDLS_DETAIL_FRONTEND_CUDA
#if !defined(TDLS_DETAIL_LANG_HIP)
/// The CUDA language.
#define TDLS_DETAIL_LANG_CUDA
#endif
#endif
#if defined(__SYCL_DEVICE_ONLY__)
/// The device pass of SYCL.
#define TDLS_DETAIL_LANG_SYCL_DEVICE
#endif

#if defined(__NVCOMPILER)
#if defined(__NVCOMPILER_STDPAR_OPENACC_GPU)
/// The parallel algorithms of nvc++ through OpenACC.
#define TDLS_DETAIL_PASS_NVCXX_STDPAR_OPENACC
#elif defined(_NVHPC_CUDA)
/// nvc++ with CUDA: one pass, NV_IF_TARGET selects the device or the host.
#define TDLS_DETAIL_PASS_NVCXX_CUDA
#elif defined(__NVCOMPILER_OPENMP_GPU) || defined(__NVCOMPILER_OPENACC_GPU)
/// OpenMP or OpenACC offloading of nvc++ without -cuda.
#define TDLS_DETAIL_PASS_NVCXX_OFFLOAD
#else
/// The host.
#define TDLS_DETAIL_PASS_HOST
#endif
#elif defined(__AMDGCN__) && defined(__clang__) &&                                                 \
    (defined(__HIP_DEVICE_COMPILE__) || !defined(__HIP__))
/// An AMD device pass.
#define TDLS_DETAIL_PASS_AMDGCN
#elif defined(__CUDA_ARCH__) && (defined(__CUDACC__) || defined(__CUDA__))
/// The device pass of the CUDA language.
#define TDLS_DETAIL_PASS_CUDA
#elif defined(__NVPTX__) && defined(__clang__) && !defined(__CUDA__)
/// An NVIDIA device pass outside the CUDA language.
#define TDLS_DETAIL_PASS_NVPTX
#elif !defined(__clang__) && (defined(__nvptx__) || defined(__AMDGCN__) || defined(__amdgcn__))
/// An offloading pass of GCC.
#define TDLS_DETAIL_PASS_GCC_OFFLOAD
#elif defined(__SYCL_DEVICE_ONLY__) || ((defined(__SPIR__) || defined(__SPIRV__)) &&               \
                                        (defined(__HIP_DEVICE_COMPILE__) || !defined(__HIP__)))
/// A SPIR-V device pass.
#define TDLS_DETAIL_PASS_SPIRV
#elif defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__) || defined(__HIPSYCL_ENABLE_LLVM_SSCP_TARGET__)
/// The generic mode of AdaptiveCpp.
#define TDLS_DETAIL_PASS_ACPP_GENERIC
#else
/// The host.
#define TDLS_DETAIL_PASS_HOST
#endif

#if defined(TDLS_DETAIL_PASS_HOST)
/// The pass generates CPU code.
#define TDLS_DETAIL_TARGET_GPU 0
#else
/// The pass generates GPU code.
#define TDLS_DETAIL_TARGET_GPU 1
#endif

/// \def TDLS_TARGET_GPU
/// \brief 1 when the code is meant for a GPU, 0 for a CPU: what the
/// defaults that depend on the hardware read
/// (ArrowheadLUppConfig::block_row_interchange Auto,
/// ArrowheadLUppConfig::rhs_pass_width -1). They shape the work of one
/// solve only; TiledLUpp and CooperativeLUpp, whose factors can cross from
/// the device to the host, reject RowInterchange::Auto. Deduced from the side of the
/// pass (TDLS_DETAIL_TARGET_GPU). Where the pass cannot tell, it guesses
/// the GPU: the generic mode of AdaptiveCpp, compiled for the device at
/// run time, and the SPIR-V passes of SYCL and OpenMP, which an OpenCL CPU
/// runs as well. -DTDLS_TARGET_GPU=0 (or 1) overrides the guess; a wrong
/// value costs speed, never results. Under nvc++ with CUDA, one pass for
/// the host and the device, the deduced value holds in constant
/// expressions only: at run time, host code takes the CPU defaults and
/// device code the GPU ones (NV_IF_TARGET), unless TDLS_TARGET_GPU is
/// given.
#ifndef TDLS_TARGET_GPU
#define TDLS_TARGET_GPU TDLS_DETAIL_TARGET_GPU
#if defined(TDLS_DETAIL_PASS_NVCXX_CUDA)
/// nvc++ with CUDA, the hardware deduced: the defaults that depend on it
/// are chosen on each side of NV_IF_TARGET (host code the CPU ones, device
/// code the GPU ones) rather than once for the pass.
#define TDLS_DETAIL_SIDE_DISPATCH
#endif
#endif



#endif // TDLS_CORE_TARGET_HPP
