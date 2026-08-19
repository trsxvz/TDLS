#ifndef TDLS_BENCHMARKS_CASES_PURE_LUPP_VARIANTS_TDLS_CUH
#define TDLS_BENCHMARKS_CASES_PURE_LUPP_VARIANTS_TDLS_CUH



/// \file
/// \brief TDLS variant machinery of the pure-LUpp case, CUDA/HIP
/// backend.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// One variant = one point of the sweep space: solver kind (static or
/// dynamic) x scalar type x tile size x schedule x unroll knob x
/// operand placement, at one system dimension. The grid itself is
/// enumerated in generate_variants.cmake, which emits one translation
/// unit per variant (one register_variant instantiation each): the
/// compile time of a translation unit, recorded by the compiler
/// launcher, is therefore the compile cost of that variant alone.
///
/// Every variant measures solve_inplace, the production entry point
/// (fused forward substitution, one right-hand-side buffer); the
/// two-buffer solve is not benchmarked.
///
/// Placements (batched storage is SoA throughout). The matrix defines
/// the family; the rhs and the pivot follow it or are pinned to
/// thread-local storage by the rreg / preg tags:
///   - reg:  every operand thread-local. The kernel gathers the system
///     from the SoA batch into local arrays and uses the internal
///     residencies of the static solver (compile-time indexing; with
///     unrolling the system lives in registers). The dynamic solver
///     has no internal mode: its reg placement uses the same local
///     arrays at unit stride, and its runtime indexing keeps them in
///     local memory - measuring that gap is the point.
///   - dram: the matrix stays in the SoA batch and is factored in
///     place at batch stride; rhs and pivot each in the batch or
///     thread-local.
///   - shm:  the matrix is staged into dynamic shared memory, one
///     system per thread at element stride ntpb; rhs and pivot each in
///     shared memory or thread-local. No synchronization is involved:
///     shared memory acts as a private extension of each thread's
///     register space.
///
/// The threads-per-block of every variant comes from the O+W launch
/// heuristic of the dispatch layer (candidates 32/64/96/128/256, then
/// the sub-warp candidates when a shared footprint exceeds the budget
/// of a full warp); --ntpb forces a value instead. Variants that
/// cannot run at all (shared footprint above the per-block limit even
/// at one thread per block, device or host memory exhausted by the
/// batch) are recorded as skipped rows, never as crashes.
///
/// The input batches are materialized in place on the device by the
/// counter-based generator (common/batch.hpp): no host batch exists
/// and nothing crosses the PCIe bus. Between timed runs, the consumed
/// inputs are restored, never inside the timing: the matrix batch when
/// it is factored in place (dram family, restored by device-side
/// regeneration) and the in-place right-hand side when it lives in the
/// batch (copied back from the pristine b_d). The validation
/// regenerates its sampled inputs on the host with the same function,
/// which doubles as the host/device identity check of the generator.



#include <chrono>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include <tdls/tdls.hpp>

#include "common/batch.hpp"
#include "common/options.hpp"
#include "common/record.hpp"
#include "common/registry.hpp"
#include "common/stats.hpp"
#include "common/validate.hpp"
#include "dispatch/cuda_or_hip.cuh"

#ifndef TDLS_BENCH_GIT_COMMIT
#define TDLS_BENCH_GIT_COMMIT "unknown"
#endif



namespace tdls_bench {



/// \brief Solver kind axis.
enum class SolverKind {
    static_, ///< TiledLUppSolverStatic (compile-time dimension)
    dynamic_ ///< TiledLUppSolverDynamic (runtime dimension)
};

/// \brief Operand placement axis: the matrix family (reg / dram / shm)
/// with the rhs and the pivot following the matrix unless pinned
/// thread-local by the rreg / preg suffix.
enum class Placement {
    reg,            ///< matrix, rhs and pivot thread-local
    dram,           ///< matrix, rhs and pivot in the SoA batch
    dram_rreg,      ///< matrix and pivot in the batch, rhs thread-local
    dram_preg,      ///< matrix and rhs in the batch, pivot thread-local
    dram_rreg_preg, ///< matrix in the batch, rhs and pivot thread-local
    shm,            ///< matrix, rhs and pivot in shared memory
    shm_rreg,       ///< matrix and pivot in shared memory, rhs thread-local
    shm_preg,       ///< matrix and rhs in shared memory, pivot thread-local
    shm_rreg_preg   ///< matrix in shared memory, rhs and pivot thread-local
};

/// \brief Memory space of the matrix under a placement.
enum class Space {
    reg,  ///< thread-local storage
    dram, ///< device-memory SoA batch
    shm   ///< dynamic shared memory
};

/// \return the matrix space of a placement
constexpr Space matrix_space(const Placement p) {
    if (p == Placement::reg) return Space::reg;
    if (p == Placement::dram || p == Placement::dram_rreg || p == Placement::dram_preg ||
        p == Placement::dram_rreg_preg)
        return Space::dram;
    return Space::shm;
}
/// \return true when the rhs of the placement is thread-local
constexpr bool rhs_local(const Placement p) {
    return p == Placement::reg || p == Placement::dram_rreg || p == Placement::dram_rreg_preg ||
           p == Placement::shm_rreg || p == Placement::shm_rreg_preg;
}
/// \return true when the pivot of the placement is thread-local
constexpr bool piv_local(const Placement p) {
    return p == Placement::reg || p == Placement::dram_preg || p == Placement::dram_rreg_preg ||
           p == Placement::shm_preg || p == Placement::shm_rreg_preg;
}

/// \return the tag token of a placement
constexpr const char* placement_token(const Placement p) {
    switch (p) {
    case Placement::reg:
        return "reg";
    case Placement::dram:
        return "dram";
    case Placement::dram_rreg:
        return "dram-rreg";
    case Placement::dram_preg:
        return "dram-preg";
    case Placement::dram_rreg_preg:
        return "dram-rreg-preg";
    case Placement::shm:
        return "shm";
    case Placement::shm_rreg:
        return "shm-rreg";
    case Placement::shm_preg:
        return "shm-preg";
    default:
        return "shm-rreg-preg";
    }
}

/// \return the CSV residency cell of a space
constexpr const char* space_res(const Space s) {
    return s == Space::reg ? "reg" : (s == Space::dram ? "dram" : "shmem");
}

/// \return the dynamic shared-memory footprint of one thread under a
/// placement: the staged matrix, plus the in-place rhs vector and the
/// pivot vector when they follow it
template<typename T, int N>
constexpr std::size_t shared_bytes_per_thread(const Placement p) {
    if (matrix_space(p) != Space::shm) return 0;
    std::size_t bytes = sizeof(T) * N * N;
    if (!rhs_local(p)) bytes += sizeof(T) * N;
    if (!piv_local(p)) bytes += sizeof(int) * N;
    return bytes;
}

/// \return the scalar tag token of a type
template<typename T>
constexpr const char* scalar_token() {
    return sizeof(T) == 8 ? "f64" : "f32";
}

/// \brief Benchmark configuration: the swept knobs over the fixed
/// defaults.
/// \tparam T  scalar type
/// \tparam TS tile size
/// \tparam S  elimination schedule
/// \tparam U  unroll_inner knob
template<typename T, int TS, tdls::TiledLUppSchedule S, bool U>
struct BenchConfig : tdls::TiledLUppConfig<T, TS, S> {
    static constexpr bool unroll_inner = U; ///< swept unroll knob
};



/* =====================================================================
   Kernels: one thread solves one system in place, through the counting
   solve_inplace entry points so the out-of-tile statistics come out
   with the run. y holds the right-hand side on entry and the solution
   on exit; every kernel deposits the final solution in the global SoA
   buffer y for validation.
   ===================================================================== */

/// \brief Materializes the SoA matrix batch in place on the device
/// from the counter-based generator, one thread per element. Also the
/// restore path of the dram family: regenerating is cheaper than any
/// copy, and no pristine duplicate ever exists.
/// \tparam T scalar type
/// \param[in]  n     system dimension
/// \param[in]  batch number of systems
/// \param[in]  seed  campaign seed
/// \param[in]  bound half-width of the input distribution
/// \param[out] A     matrix batch, SoA (element stride = batch)
template<typename T>
__global__ void kernel_generate_matrices(const int n, const int batch, const std::uint64_t seed,
                                         const double bound, T* TDLS_RESTRICT A) {
    const long long t     = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(batch) * n * n;
    if (t >= total) return;
    const int s                                = static_cast<int>(t / (n * n));
    const int e                                = static_cast<int>(t % (n * n));
    A[static_cast<std::size_t>(e) * batch + s] = static_cast<T>(
        counter_draw(seed, matrix_stream, static_cast<std::uint64_t>(s) * n * n + e, bound));
}

/// \brief Materializes the SoA right-hand-side batch in place on the
/// device, one thread per entry.
/// \tparam T scalar type
/// \param[in]  n     system dimension
/// \param[in]  batch number of systems
/// \param[in]  seed  campaign seed
/// \param[in]  bound half-width of the input distribution
/// \param[out] b     right-hand-side batch, SoA (element stride = batch)
template<typename T>
__global__ void kernel_generate_rhs(const int n, const int batch, const std::uint64_t seed,
                                    const double bound, T* TDLS_RESTRICT b) {
    const long long t     = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(batch) * n;
    if (t >= total) return;
    const int s                                = static_cast<int>(t / n);
    const int i                                = static_cast<int>(t % n);
    b[static_cast<std::size_t>(i) * batch + s] = static_cast<T>(
        counter_draw(seed, rhs_stream, static_cast<std::uint64_t>(s) * n + i, bound));
}

/// \brief Static solver, thread-local operands (internal residencies).
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U>
__global__ void kernel_static_reg(const int batch, const T* TDLS_RESTRICT A_in,
                                  const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT y_out,
                                  int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverStatic<T, N, BenchConfig<T, TS, S, U>>;
    T A[N * N];
    T y[N];
    int piv[N];
    // The gathers follow the unroll knob through the two-branch trick
    // of the solvers: under u1 they must fully unroll (compile-time
    // indexing keeps the local system in registers), under u0 the
    // solver indexes dynamically anyway and the twin variants must
    // differ by the solver knob alone, so no pragma is emitted at all.
    if constexpr (U) {
#pragma unroll
        for (int e = 0; e < N * N; ++e)
            A[e] = A_in[static_cast<std::size_t>(e) * batch + s];
#pragma unroll
        for (int i = 0; i < N; ++i)
            y[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    } else {
        for (int e = 0; e < N * N; ++e)
            A[e] = A_in[static_cast<std::size_t>(e) * batch + s];
        for (int i = 0; i < N; ++i)
            y[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    }
    int count       = 0;
    const bool good = Solver::template solve_inplace<true, true, true>(A, 1, piv, 1, y, 1, count);
    if constexpr (U) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = y[i];
    } else {
        for (int i = 0; i < N; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = y[i];
    }
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Static solver, matrix factored in place in the SoA batch;
/// rhs and pivot in the batch or thread-local.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U, bool RLOCAL, bool PLOCAL>
__global__ void kernel_static_dram(const int batch, T* TDLS_RESTRICT A, const T* TDLS_RESTRICT b_in,
                                   T* TDLS_RESTRICT y, int* TDLS_RESTRICT piv_g,
                                   int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverStatic<T, N, BenchConfig<T, TS, S, U>>;
    int count    = 0;
    bool good;
    T yl[N];
    int pl[N];
    if constexpr (RLOCAL && U) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            yl[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    } else if constexpr (RLOCAL) {
        for (int i = 0; i < N; ++i)
            yl[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    }
    T* y_arg             = RLOCAL ? yl : y + s;
    const int y_stride   = RLOCAL ? 1 : batch;
    int* piv_arg         = PLOCAL ? pl : piv_g + s;
    const int piv_stride = PLOCAL ? 1 : batch;
    good = Solver::template solve_inplace<RLOCAL, PLOCAL, false>(A + s, batch, piv_arg, piv_stride,
                                                                 y_arg, y_stride, count);
    if constexpr (RLOCAL && U) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            y[static_cast<std::size_t>(i) * batch + s] = yl[i];
    } else if constexpr (RLOCAL) {
        for (int i = 0; i < N; ++i)
            y[static_cast<std::size_t>(i) * batch + s] = yl[i];
    }
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Static solver, matrix staged in dynamic shared memory (one
/// system per thread, element stride blockDim.x); rhs and pivot in
/// shared memory or thread-local.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U, bool RLOCAL, bool PLOCAL>
__global__ void kernel_static_shm(const int batch, const T* TDLS_RESTRICT A_in,
                                  const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT y_out,
                                  int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    extern __shared__ unsigned char smem[];
    const int width = static_cast<int>(blockDim.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int s     = static_cast<int>(blockIdx.x) * width + tid;
    T* Ms           = reinterpret_cast<T*>(smem);
    T* ys           = Ms + static_cast<std::size_t>(width) * N * N;
    int* ps = reinterpret_cast<int*>(ys + (RLOCAL ? 0 : static_cast<std::size_t>(width) * N));
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverStatic<T, N, BenchConfig<T, TS, S, U>>;
    for (int e = 0; e < N * N; ++e)
        Ms[e * width + tid] = A_in[static_cast<std::size_t>(e) * batch + s];
    int count = 0;
    bool good;
    T yl[N];
    int pl[N];
    if constexpr (RLOCAL && U) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            yl[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    } else if constexpr (RLOCAL) {
        for (int i = 0; i < N; ++i)
            yl[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    } else {
        for (int i = 0; i < N; ++i)
            ys[i * width + tid] = b_in[static_cast<std::size_t>(i) * batch + s];
    }
    T* y_arg             = RLOCAL ? yl : ys + tid;
    const int y_stride   = RLOCAL ? 1 : width;
    int* piv_arg         = PLOCAL ? pl : ps + tid;
    const int piv_stride = PLOCAL ? 1 : width;
    good                 = Solver::template solve_inplace<RLOCAL, PLOCAL, false>(
        Ms + tid, width, piv_arg, piv_stride, y_arg, y_stride, count);
    if constexpr (RLOCAL && U) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = yl[i];
    } else if constexpr (RLOCAL) {
        for (int i = 0; i < N; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = yl[i];
    } else {
        for (int i = 0; i < N; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = ys[i * width + tid];
    }
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Dynamic solver, thread-local operands at unit stride (runtime
/// indexing: local memory by construction).
template<typename T, int N, int TS, tdls::TiledLUppSchedule S>
__global__ void kernel_dynamic_reg(const int n, const int batch, const T* TDLS_RESTRICT A_in,
                                   const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT y_out,
                                   int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverDynamic<T, BenchConfig<T, TS, S, true>>;
    T A[N * N]; // capacity from the translation unit; extents from n
    T y[N];
    int piv[N];
    for (int e = 0; e < n * n; ++e)
        A[e] = A_in[static_cast<std::size_t>(e) * batch + s];
    for (int i = 0; i < n; ++i)
        y[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    int count       = 0;
    const bool good = Solver::solve_inplace(n, A, 1, piv, 1, y, 1, count);
    for (int i = 0; i < n; ++i)
        y_out[static_cast<std::size_t>(i) * batch + s] = y[i];
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Dynamic solver, matrix factored in place in the SoA batch;
/// rhs and pivot in the batch or thread-local.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool RLOCAL, bool PLOCAL>
__global__ void kernel_dynamic_dram(const int n, const int batch, T* TDLS_RESTRICT A,
                                    const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT y,
                                    int* TDLS_RESTRICT piv_g, int* TDLS_RESTRICT ok,
                                    int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverDynamic<T, BenchConfig<T, TS, S, true>>;
    T yl[N];
    int pl[N];
    if constexpr (RLOCAL) {
        for (int i = 0; i < n; ++i)
            yl[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    }
    T* y_arg             = RLOCAL ? yl : y + s;
    const int y_stride   = RLOCAL ? 1 : batch;
    int* piv_arg         = PLOCAL ? pl : piv_g + s;
    const int piv_stride = PLOCAL ? 1 : batch;
    int count            = 0;
    const bool good =
        Solver::solve_inplace(n, A + s, batch, piv_arg, piv_stride, y_arg, y_stride, count);
    if constexpr (RLOCAL) {
        for (int i = 0; i < n; ++i)
            y[static_cast<std::size_t>(i) * batch + s] = yl[i];
    }
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Dynamic solver, matrix staged in dynamic shared memory; rhs
/// and pivot in shared memory or thread-local.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool RLOCAL, bool PLOCAL>
__global__ void kernel_dynamic_shm(const int n, const int batch, const T* TDLS_RESTRICT A_in,
                                   const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT y_out,
                                   int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    extern __shared__ unsigned char smem[];
    const int width = static_cast<int>(blockDim.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int s     = static_cast<int>(blockIdx.x) * width + tid;
    T* Ms           = reinterpret_cast<T*>(smem);
    T* ys           = Ms + static_cast<std::size_t>(width) * N * N;
    int* ps = reinterpret_cast<int*>(ys + (RLOCAL ? 0 : static_cast<std::size_t>(width) * N));
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverDynamic<T, BenchConfig<T, TS, S, true>>;
    for (int e = 0; e < n * n; ++e)
        Ms[e * width + tid] = A_in[static_cast<std::size_t>(e) * batch + s];
    T yl[N];
    int pl[N];
    if constexpr (RLOCAL) {
        for (int i = 0; i < n; ++i)
            yl[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    } else {
        for (int i = 0; i < n; ++i)
            ys[i * width + tid] = b_in[static_cast<std::size_t>(i) * batch + s];
    }
    T* y_arg             = RLOCAL ? yl : ys + tid;
    const int y_stride   = RLOCAL ? 1 : width;
    int* piv_arg         = PLOCAL ? pl : ps + tid;
    const int piv_stride = PLOCAL ? 1 : width;
    int count            = 0;
    const bool good =
        Solver::solve_inplace(n, Ms + tid, width, piv_arg, piv_stride, y_arg, y_stride, count);
    if constexpr (RLOCAL) {
        for (int i = 0; i < n; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = yl[i];
    } else {
        for (int i = 0; i < n; ++i)
            y_out[static_cast<std::size_t>(i) * batch + s] = ys[i * width + tid];
    }
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}



/* =====================================================================
   Runner: launch selection, batch setup, restore protocol, timing,
   metrics, validation, skip handling.
   ===================================================================== */

/// \return the schedule token of a tag or CSV cell
inline const char* schedule_token(const tdls::TiledLUppSchedule s) {
    return s == tdls::TiledLUppSchedule::RightLooking ? "rl" : "ll";
}

/// \brief Builds the variant tag, the primary key of the CSV rows. Its
/// slash-separated form maps one-to-one onto the underscore-separated
/// stem of the generated translation units (no token contains an
/// underscore), which is how the compile-time log keys join this one.
/// \tparam T scalar type
/// \param[in] kind solver kind
/// \param[in] n    system dimension
/// \param[in] ts   tile size
/// \param[in] s    schedule
/// \param[in] u    unroll knob (ignored for the dynamic solver)
/// \param[in] p    placement
/// \return the tag
template<typename T>
inline std::string make_tag(const SolverKind kind, const int n, const int ts,
                            const tdls::TiledLUppSchedule s, const bool u, const Placement p) {
    std::string tag = "purelupp/";
    tag += kind == SolverKind::static_ ? "static" : "dynamic";
    tag += "/";
    tag += scalar_token<T>();
    tag += "/n" + std::to_string(n) + "/ts" + std::to_string(ts) + "/";
    tag += schedule_token(s);
    if (kind == SolverKind::static_) tag += u ? "/u1" : "/u0";
    tag += "/";
    tag += placement_token(p);
    return tag;
}

/// \brief Measures one variant under one distribution and returns its
/// record; infeasible variants come back as skipped rows, never as
/// crashes.
/// \tparam T scalar type
/// \tparam N system dimension
/// \tparam TS tile size
/// \tparam S schedule
/// \tparam U unroll knob
/// \tparam K solver kind
/// \tparam P operand placement
/// \param[in] opt  command-line options
/// \param[in] dist input distribution
/// \return the filled record
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U, SolverKind K, Placement P>
Record run_variant(const Options& opt, const Distribution dist) {
    const int batch         = opt.batch;
    const DeviceInfo device = query_device_info();

    // The axes and the environment are known before anything runs, so
    // a skipped variant still produces a complete row.
    Record r;
    r.timestamp_utc   = utc_timestamp();
    r.tag             = make_tag<T>(K, N, TS, S, U, P);
    r.case_name       = "pure_lupp";
    r.backend         = backend_name();
    r.solver          = K == SolverKind::static_ ? "static" : "dynamic";
    r.n               = N;
    r.tile_size       = TS;
    r.schedule        = schedule_token(S);
    r.unroll_inner    = K == SolverKind::static_ ? (U ? "1" : "0") : "";
    r.res_matrix      = space_res(matrix_space(P));
    r.res_rhs         = rhs_local(P) ? "reg" : space_res(matrix_space(P));
    r.res_piv         = piv_local(P) ? "reg" : space_res(matrix_space(P));
    r.layout          = P == Placement::reg ? "" : "soa";
    r.scalar          = scalar_token<T>();
    r.distribution    = distribution_name(dist);
    r.batch           = batch;
    r.seed            = opt.seed;
    r.gpu_name        = device.name;
    r.cc              = device.cc;
    r.driver_version  = device.driver_version;
    r.runtime_version = device.runtime_version;
    r.sm_count        = device.sm_count;
    r.smem_per_sm     = device.smem_per_sm;
    r.compiler        = compiler_string();
    r.tdls_version    = TDLS_VERSION_STRING;
    r.git_commit      = TDLS_BENCH_GIT_COMMIT;

    // Kernel resolution and launch selection (O+W heuristic, sub-warp
    // fallback under a shared budget, feasibility verdict).
    constexpr Space MS    = matrix_space(P);
    constexpr bool RLOCAL = rhs_local(P);
    constexpr bool PLOCAL = piv_local(P);
    const auto kernel     = []() {
        if constexpr (K == SolverKind::static_ && MS == Space::reg) {
            return kernel_static_reg<T, N, TS, S, U>;
        } else if constexpr (K == SolverKind::static_ && MS == Space::dram) {
            return kernel_static_dram<T, N, TS, S, U, RLOCAL, PLOCAL>;
        } else if constexpr (K == SolverKind::static_) {
            return kernel_static_shm<T, N, TS, S, U, RLOCAL, PLOCAL>;
        } else if constexpr (MS == Space::reg) {
            return kernel_dynamic_reg<T, N, TS, S>;
        } else if constexpr (MS == Space::dram) {
            return kernel_dynamic_dram<T, N, TS, S, RLOCAL, PLOCAL>;
        } else {
            return kernel_dynamic_shm<T, N, TS, S, RLOCAL, PLOCAL>;
        }
    }();
    // The solvers compute external offsets in unsigned 32-bit
    // arithmetic: at batch stride, the largest matrix offset
    // (N*N-1)*batch must stay below 2^32 (see the solver headers). A
    // batch too large for this dimension is a recorded skip.
    if constexpr (MS == Space::dram) {
        if (static_cast<unsigned long long>(N) * N * batch >= (1ull << 32)) {
            r.status = "skip_offset32";
            return r;
        }
    }
    const std::size_t bytes_per_thread = shared_bytes_per_thread<T, N>(P);
    const LaunchChoice launch_choice =
        choose_launch(kernel, bytes_per_thread, batch, device, opt.ntpb);
    if (!launch_choice.feasible) {
        r.status = "skip_smem";
        return r;
    }
    const int ntpb   = launch_choice.ntpb;
    const int blocks = (batch + ntpb - 1) / ntpb;
    r.ntpb           = ntpb;
    r.blocks         = blocks;
    r.dyn_smem_bytes = launch_choice.dyn_smem;

    const std::size_t A_size = static_cast<std::size_t>(N) * N * batch * sizeof(T);
    const std::size_t v_size = static_cast<std::size_t>(N) * batch * sizeof(T);

    // Device budget, checked predictively against the free memory
    // (with a safety margin) before anything is allocated. The
    // non-fatal allocations below remain the authoritative verdict:
    // they see fragmentation, and the free memory can move under us.
    std::size_t required = A_size + 2 * v_size + 2 * sizeof(int) * static_cast<std::size_t>(batch);
    if constexpr (MS == Space::dram && !PLOCAL)
        required += sizeof(int) * static_cast<std::size_t>(N) * batch;
    if (required + (64ull << 20) > query_free_memory()) {
        r.status = "skip_dram";
        return r;
    }

    // Device buffers; an exhausted device memory is a recorded skip.
    // The inputs are materialized in place by the generation kernels:
    // no host batch exists, and nothing crosses the PCIe bus. b_d
    // holds the pristine right-hand sides; y_d receives the solutions,
    // and doubles as the in-place working buffer when the rhs lives in
    // the batch (restored from b_d before every run). Only the dram
    // family factors the matrix in place and needs its restore (a
    // device-side regeneration) and a device pivot buffer.
    T* A_d             = nullptr;
    T* b_d             = nullptr;
    T* y_d             = nullptr;
    int* piv_d         = nullptr;
    int* ok_d          = nullptr;
    int* oot_d         = nullptr;
    const auto release = [&]() {
        for (void* p :
             {static_cast<void*>(A_d), static_cast<void*>(b_d), static_cast<void*>(y_d),
              static_cast<void*>(piv_d), static_cast<void*>(ok_d), static_cast<void*>(oot_d)})
            if (p != nullptr) GPU_CHECK(gpuFree(p));
    };
    bool allocated = gpu_try_malloc(&A_d, A_size) && gpu_try_malloc(&b_d, v_size) &&
                     gpu_try_malloc(&y_d, v_size) && gpu_try_malloc(&ok_d, sizeof(int) * batch) &&
                     gpu_try_malloc(&oot_d, sizeof(int) * batch);
    if constexpr (MS == Space::dram && !PLOCAL) {
        allocated =
            allocated && gpu_try_malloc(&piv_d, sizeof(int) * static_cast<std::size_t>(N) * batch);
    }
    if (!allocated) {
        release();
        r.status = "skip_dram";
        return r;
    }
    const double bound    = distribution_bound<T>(dist);
    const auto generate_A = [&]() {
        const long long total = static_cast<long long>(batch) * N * N;
        const int gen_blocks  = static_cast<int>((total + 255) / 256);
        kernel_generate_matrices<T><<<gen_blocks, 256>>>(N, batch, opt.seed, bound, A_d);
        GPU_CHECK(gpuGetLastError());
    };
    {
        const long long total = static_cast<long long>(batch) * N;
        const int gen_blocks  = static_cast<int>((total + 255) / 256);
        kernel_generate_rhs<T><<<gen_blocks, 256>>>(N, batch, opt.seed, bound, b_d);
        GPU_CHECK(gpuGetLastError());
    }
    generate_A();
    GPU_CHECK(gpuDeviceSynchronize());
    const auto restore = [&]() {
        if constexpr (MS == Space::dram) {
            generate_A(); // in-place regeneration, cheaper than any copy
            if constexpr (!RLOCAL) {
                GPU_CHECK(gpuMemcpy(y_d, b_d, v_size, gpuMemcpyDeviceToDevice));
            }
        }
    };
    const auto launch_raw = [&]() {
        const std::size_t dyn = launch_choice.dyn_smem;
        if constexpr (K == SolverKind::static_ && MS == Space::reg) {
            kernel_static_reg<T, N, TS, S, U><<<blocks, ntpb>>>(batch, A_d, b_d, y_d, ok_d, oot_d);
        } else if constexpr (K == SolverKind::static_ && MS == Space::dram) {
            kernel_static_dram<T, N, TS, S, U, RLOCAL, PLOCAL>
                <<<blocks, ntpb>>>(batch, A_d, b_d, y_d, piv_d, ok_d, oot_d);
        } else if constexpr (K == SolverKind::static_) {
            kernel_static_shm<T, N, TS, S, U, RLOCAL, PLOCAL>
                <<<blocks, ntpb, dyn>>>(batch, A_d, b_d, y_d, ok_d, oot_d);
        } else if constexpr (MS == Space::reg) {
            kernel_dynamic_reg<T, N, TS, S><<<blocks, ntpb>>>(N, batch, A_d, b_d, y_d, ok_d, oot_d);
        } else if constexpr (MS == Space::dram) {
            kernel_dynamic_dram<T, N, TS, S, RLOCAL, PLOCAL>
                <<<blocks, ntpb>>>(N, batch, A_d, b_d, y_d, piv_d, ok_d, oot_d);
        } else {
            kernel_dynamic_shm<T, N, TS, S, RLOCAL, PLOCAL>
                <<<blocks, ntpb, dyn>>>(N, batch, A_d, b_d, y_d, ok_d, oot_d);
        }
    };
    const auto launch = [&]() {
        launch_raw();
        GPU_CHECK(gpuGetLastError());
    };

    // Launch probe: a configuration the device rejects surfaces here
    // as a recorded row, never as an abort (each variant runs in its
    // own process, so the sticky error state dies with it). The probe
    // is also timed (host clock, synchronous): it prices the variant
    // for the budget policy below.
    restore();
    const auto probe_start = std::chrono::steady_clock::now();
    launch_raw();
    if (TDLS_EXAMPLES_GPU_API(GetLastError)() != gpuSuccess ||
        TDLS_EXAMPLES_GPU_API(DeviceSynchronize)() != gpuSuccess) {
        TDLS_EXAMPLES_GPU_API(GetLastError)(); // clear the sticky state
        release();
        r.status = "error_launch";
        return r;
    }

    const double probe_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - probe_start)
            .count();

    // Budget policy: when a per-variant time budget is set and the
    // probe prices the full protocol above it, the remaining warmups
    // are dropped and the timed runs are cut to what fits, never
    // below one. No schema change: a shortened row simply lists fewer
    // raw runs in its t_runs_ms cell. Long kernels are run-to-run
    // stable, so repetition is cut exactly where it informs least.
    int warmups = opt.warmup;
    int runs    = opt.runs;
    if (opt.budget_s > 0.0) {
        const double projected = probe_ms * (warmups - 1 + runs);
        if (projected > opt.budget_s * 1000.0) {
            warmups       = 1;
            const int fit = static_cast<int>(opt.budget_s * 1000.0 / probe_ms);
            runs          = fit < 1 ? 1 : (fit > opt.runs ? opt.runs : fit);
        }
    }

    // Protocol: untimed warmups (the probe already ran the first),
    // then the timed runs, every run on pristine inputs.
    for (int w = 1; w < warmups; ++w) {
        restore();
        launch();
        GPU_CHECK(gpuDeviceSynchronize());
    }
    GpuTimer timer;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(runs));
    for (int run = 0; run < runs; ++run) {
        restore();
        times.push_back(timer.time_ms(launch));
    }

    // Self-reported kernel metrics under the selected launch.
    const KernelMetrics metrics =
        kernel_metrics(kernel, ntpb, launch_choice.dyn_smem, blocks, device);

    // Outputs of the last run; these host buffers are small (batch
    // vectors, never a matrix batch), an allocation failure is still a
    // recorded skip.
    std::vector<T> y_soa;
    std::vector<int> ok;
    std::vector<int> oot;
    try {
        y_soa.resize(static_cast<std::size_t>(N) * batch);
        ok.resize(batch);
        oot.resize(batch);
    } catch (const std::bad_alloc&) {
        release();
        r.status = "skip_host";
        return r;
    }
    GPU_CHECK(gpuMemcpy(y_soa.data(), y_d, v_size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), ok_d, sizeof(int) * batch, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(oot.data(), oot_d, sizeof(int) * batch, gpuMemcpyDeviceToHost));
    release();

    // Measurement assembly.
    r.regs_per_thread           = metrics.regs_per_thread;
    r.local_bytes_per_thread    = metrics.local_bytes_per_thread;
    r.static_smem_bytes         = metrics.static_smem_bytes;
    r.theoretical_occupancy_pct = metrics.theoretical_occupancy_pct;
    r.wave_fill_pct             = metrics.wave_fill_pct;
    r.t_runs_ms                 = times;
    r.t_ms                      = compute_stats(times);
    r.systems_per_s             = r.t_ms.median > 0.0 ? batch / (r.t_ms.median / 1000.0) : 0.0;
    for (const int flag : ok)
        r.solved += flag;
    r.parity = verdict_parity_sample<T>(N, batch, opt.seed, bound, ok, opt.parity_sample);
    r.be     = backward_error_sample<T>(N, batch, opt.seed, bound, y_soa, ok, opt.validate_sample,
                                        r.validated_systems);
    std::vector<double> oot_values(oot.begin(), oot.end());
    r.oot = compute_stats(std::move(oot_values));
    for (const int count : oot)
        if (count > 0) ++r.oot_systems;
    return r;
}

/// \brief Registers one variant; called once per generated translation
/// unit.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U, SolverKind K, Placement P>
void register_variant() {
    registry().push_back(
        {make_tag<T>(K, N, TS, S, U, P), [](const Options& opt, const Distribution dist) {
             return run_variant<T, N, TS, S, U, K, P>(opt, dist);
         }});
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_CASES_PURE_LUPP_VARIANTS_TDLS_CUH
