#ifndef TDLS_BENCHMARKS_CASES_PURE_LUPP_VARIANTS_TDLS_CUH
#define TDLS_BENCHMARKS_CASES_PURE_LUPP_VARIANTS_TDLS_CUH



/// \file
/// \brief TDLS variant grid of the pure-LUpp case, CUDA/HIP backend.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// One variant = one point of the sweep space: solver kind (static or
/// dynamic) x tile size x schedule x unroll knob x operand placement,
/// at one system dimension. The per-dimension translation units
/// (generated at configure time) call register_dimension<N>() and the
/// main program runs the registered closures.
///
/// Placements of this first version:
///   - reg:  every operand in thread-local storage. The kernel gathers
///     the system from the SoA input batch into local arrays and calls
///     the internal residencies of the static solver (compile-time
///     indexing; with unrolling the system lives in registers). The
///     dynamic solver has no internal mode: its reg placement uses the
///     same local arrays at unit stride, and its runtime indexing keeps
///     them in local memory - measuring that gap is the point.
///   - dram: every operand stays in the SoA batch in device memory,
///     factored in place at batch stride (external residencies).
/// Shared-memory staging, mixed residencies and the AoS/AoSoA layouts
/// are planned extensions of this grid.
///
/// The matrix batch is restored between timed runs (factorization is in
/// place), device-to-device when a pristine copy fits in device memory,
/// from the host otherwise; the restore is never timed.



#include <cstddef>
#include <string>
#include <vector>

#include <tdls/tdls.hpp>

#include "common/options.hpp"
#include "common/record.hpp"
#include "common/registry.hpp"
#include "common/stats.hpp"
#include "common/validate.hpp"
#include "dispatch/cuda_or_hip.cuh"
#include "generators.hpp"

#ifndef TDLS_BENCH_GIT_COMMIT
#define TDLS_BENCH_GIT_COMMIT "unknown"
#endif



namespace tdls_bench {



/// \brief Solver kind axis.
enum class SolverKind {
    static_, ///< TiledLUppSolverStatic (compile-time dimension)
    dynamic_ ///< TiledLUppSolverDynamic (runtime dimension)
};

/// \brief Operand placement axis (uniform over matrix/rhs/pivot in this
/// first version).
enum class Placement {
    reg, ///< thread-local storage (registers when indexing allows)
    dram ///< device-memory SoA batch, walked at batch stride
};

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
   Kernels: one thread solves one system, through the counting solve
   entry points so the out-of-tile statistics come out with the run.
   ===================================================================== */

/// \brief Static solver, thread-local operands (internal residencies).
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U>
__global__ void kernel_static_reg(const int batch, const T* TDLS_RESTRICT A_in,
                                  const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT x_out,
                                  int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverStatic<T, N, BenchConfig<T, TS, S, U>>;
    T A[N * N];
    T b[N];
    T x[N];
    int piv[N];
    // Compile-time-indexed gathers: the local system only stays in
    // registers if these loops fully unroll, like the solver's own.
#pragma unroll
    for (int e = 0; e < N * N; ++e)
        A[e] = A_in[static_cast<std::size_t>(e) * batch + s];
#pragma unroll
    for (int i = 0; i < N; ++i)
        b[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    int count       = 0;
    const bool good = Solver::template solve<true, true, true>(A, 1, piv, 1, b, x, 1, count);
#pragma unroll
    for (int i = 0; i < N; ++i)
        x_out[static_cast<std::size_t>(i) * batch + s] = x[i];
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Static solver, device-memory SoA operands (external
/// residencies, factorization in place at batch stride).
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U>
__global__ void kernel_static_dram(const int batch, T* TDLS_RESTRICT A, const T* TDLS_RESTRICT b,
                                   T* TDLS_RESTRICT x, int* TDLS_RESTRICT piv,
                                   int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver    = tdls::TiledLUppSolverStatic<T, N, BenchConfig<T, TS, S, U>>;
    int count       = 0;
    const bool good = Solver::template solve<false, false, false>(A + s, batch, piv + s, batch,
                                                                  b + s, x + s, batch, count);
    ok[s]           = good ? 1 : 0;
    oot[s]          = count;
}

/// \brief Dynamic solver, thread-local operands at unit stride (runtime
/// indexing: local memory by construction).
template<typename T, int N, int TS, tdls::TiledLUppSchedule S>
__global__ void kernel_dynamic_reg(const int n, const int batch, const T* TDLS_RESTRICT A_in,
                                   const T* TDLS_RESTRICT b_in, T* TDLS_RESTRICT x_out,
                                   int* TDLS_RESTRICT ok, int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver = tdls::TiledLUppSolverDynamic<T, BenchConfig<T, TS, S, true>>;
    T A[N * N]; // capacity from the translation unit; extents from n
    T b[N];
    T x[N];
    int piv[N];
    for (int e = 0; e < n * n; ++e)
        A[e] = A_in[static_cast<std::size_t>(e) * batch + s];
    for (int i = 0; i < n; ++i)
        b[i] = b_in[static_cast<std::size_t>(i) * batch + s];
    int count       = 0;
    const bool good = Solver::solve(n, A, 1, piv, 1, b, x, 1, count);
    for (int i = 0; i < n; ++i)
        x_out[static_cast<std::size_t>(i) * batch + s] = x[i];
    ok[s]  = good ? 1 : 0;
    oot[s] = count;
}

/// \brief Dynamic solver, device-memory SoA operands.
template<typename T, int TS, tdls::TiledLUppSchedule S>
__global__ void kernel_dynamic_dram(const int n, const int batch, T* TDLS_RESTRICT A,
                                    const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x,
                                    int* TDLS_RESTRICT piv, int* TDLS_RESTRICT ok,
                                    int* TDLS_RESTRICT oot) {
    const int s = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    using Solver    = tdls::TiledLUppSolverDynamic<T, BenchConfig<T, TS, S, true>>;
    int count       = 0;
    const bool good = Solver::solve(n, A + s, batch, piv + s, batch, b + s, x + s, batch, count);
    ok[s]           = good ? 1 : 0;
    oot[s]          = count;
}



/* =====================================================================
   Runner: batch setup, restore protocol, timing, metrics, validation.
   ===================================================================== */

/// \return the schedule token of a tag or CSV cell
inline const char* schedule_token(const tdls::TiledLUppSchedule s) {
    return s == tdls::TiledLUppSchedule::RightLooking ? "rl" : "ll";
}

/// \return the placement token of a tag or CSV cell
inline const char* placement_token(const Placement p) {
    return p == Placement::reg ? "reg" : "dram";
}

/// \brief Builds the variant tag, the primary key of the CSV rows.
/// \param[in] kind solver kind
/// \param[in] n    system dimension
/// \param[in] ts   tile size
/// \param[in] s    schedule
/// \param[in] u    unroll knob (ignored for the dynamic solver)
/// \param[in] p    placement
/// \return the tag
inline std::string make_tag(const SolverKind kind, const int n, const int ts,
                            const tdls::TiledLUppSchedule s, const bool u, const Placement p) {
    std::string tag = "purelupp/";
    tag += kind == SolverKind::static_ ? "static" : "dynamic";
    tag += "/n" + std::to_string(n) + "/ts" + std::to_string(ts) + "/";
    tag += schedule_token(s);
    if (kind == SolverKind::static_) tag += u ? "/u1" : "/u0";
    tag += "/";
    tag += placement_token(p);
    return tag;
}

/// \brief Measures one variant under one distribution and returns its
/// record.
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
    const int batch  = opt.batch;
    const int ntpb   = opt.ntpb;
    const int blocks = (batch + ntpb - 1) / ntpb;

    // Reproducible inputs, scattered to the SoA staging buffers.
    const auto host = tdls_tests::make_batch<T>(N, batch, opt.seed, distribution_bound(dist));
    std::vector<T> A_soa(static_cast<std::size_t>(N) * N * batch);
    std::vector<T> b_soa(static_cast<std::size_t>(N) * batch);
    for (int s = 0; s < batch; ++s) {
        for (int e = 0; e < N * N; ++e)
            A_soa[static_cast<std::size_t>(e) * batch + s] = host.matrix(s)[e];
        for (int i = 0; i < N; ++i)
            b_soa[static_cast<std::size_t>(i) * batch + s] = host.rhs(s)[i];
    }

    const DeviceInfo device  = query_device_info();
    const std::size_t A_size = A_soa.size() * sizeof(T);
    const std::size_t v_size = b_soa.size() * sizeof(T);

    // Device buffers. The reg kernels read the matrix batch and never
    // write it back, so they need no restore; the dram kernels factor
    // it in place and restore it before every run, device-to-device
    // when a pristine copy fits, from the host otherwise.
    T* A_d          = nullptr;
    T* A_pristine_d = nullptr;
    T* b_d          = nullptr;
    T* x_d          = nullptr;
    int* piv_d      = nullptr;
    int* ok_d       = nullptr;
    int* oot_d      = nullptr;
    GPU_CHECK(gpuMalloc(&A_d, A_size));
    GPU_CHECK(gpuMalloc(&b_d, v_size));
    GPU_CHECK(gpuMalloc(&x_d, v_size));
    GPU_CHECK(gpuMalloc(&ok_d, sizeof(int) * batch));
    GPU_CHECK(gpuMalloc(&oot_d, sizeof(int) * batch));
    GPU_CHECK(gpuMemcpy(A_d, A_soa.data(), A_size, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(b_d, b_soa.data(), v_size, gpuMemcpyHostToDevice));
    if constexpr (P == Placement::dram) {
        GPU_CHECK(gpuMalloc(&piv_d, sizeof(int) * static_cast<std::size_t>(N) * batch));
        if (query_free_memory() > A_size + (A_size / 8)) {
            GPU_CHECK(gpuMalloc(&A_pristine_d, A_size));
            GPU_CHECK(gpuMemcpy(A_pristine_d, A_d, A_size, gpuMemcpyDeviceToDevice));
        }
    }
    const auto restore = [&]() {
        if constexpr (P == Placement::dram) {
            if (A_pristine_d != nullptr) {
                GPU_CHECK(gpuMemcpy(A_d, A_pristine_d, A_size, gpuMemcpyDeviceToDevice));
            } else {
                GPU_CHECK(gpuMemcpy(A_d, A_soa.data(), A_size, gpuMemcpyHostToDevice));
            }
        }
    };
    const auto launch = [&]() {
        if constexpr (K == SolverKind::static_ && P == Placement::reg) {
            kernel_static_reg<T, N, TS, S, U><<<blocks, ntpb>>>(batch, A_d, b_d, x_d, ok_d, oot_d);
        } else if constexpr (K == SolverKind::static_ && P == Placement::dram) {
            kernel_static_dram<T, N, TS, S, U>
                <<<blocks, ntpb>>>(batch, A_d, b_d, x_d, piv_d, ok_d, oot_d);
        } else if constexpr (K == SolverKind::dynamic_ && P == Placement::reg) {
            kernel_dynamic_reg<T, N, TS, S><<<blocks, ntpb>>>(N, batch, A_d, b_d, x_d, ok_d, oot_d);
        } else {
            kernel_dynamic_dram<T, TS, S>
                <<<blocks, ntpb>>>(N, batch, A_d, b_d, x_d, piv_d, ok_d, oot_d);
        }
        GPU_CHECK(gpuGetLastError());
    };

    // Protocol: untimed warmups, then the timed runs, every run on
    // pristine inputs.
    for (int w = 0; w < opt.warmup; ++w) {
        restore();
        launch();
        GPU_CHECK(gpuDeviceSynchronize());
    }
    GpuTimer timer;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(opt.runs));
    for (int r = 0; r < opt.runs; ++r) {
        restore();
        times.push_back(timer.time_ms(launch));
    }

    // Self-reported kernel metrics.
    KernelMetrics metrics;
    if constexpr (K == SolverKind::static_ && P == Placement::reg) {
        metrics = kernel_metrics(kernel_static_reg<T, N, TS, S, U>, ntpb, 0, blocks, device);
    } else if constexpr (K == SolverKind::static_ && P == Placement::dram) {
        metrics = kernel_metrics(kernel_static_dram<T, N, TS, S, U>, ntpb, 0, blocks, device);
    } else if constexpr (K == SolverKind::dynamic_ && P == Placement::reg) {
        metrics = kernel_metrics(kernel_dynamic_reg<T, N, TS, S>, ntpb, 0, blocks, device);
    } else {
        metrics = kernel_metrics(kernel_dynamic_dram<T, TS, S>, ntpb, 0, blocks, device);
    }

    // Outputs of the last run.
    std::vector<T> x_soa(b_soa.size());
    std::vector<int> ok(batch);
    std::vector<int> oot(batch);
    GPU_CHECK(gpuMemcpy(x_soa.data(), x_d, v_size, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), ok_d, sizeof(int) * batch, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(oot.data(), oot_d, sizeof(int) * batch, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(A_d));
    if (A_pristine_d != nullptr) GPU_CHECK(gpuFree(A_pristine_d));
    GPU_CHECK(gpuFree(b_d));
    GPU_CHECK(gpuFree(x_d));
    if (piv_d != nullptr) GPU_CHECK(gpuFree(piv_d));
    GPU_CHECK(gpuFree(ok_d));
    GPU_CHECK(gpuFree(oot_d));

    // Record assembly.
    Record r;
    r.timestamp_utc             = utc_timestamp();
    r.tag                       = make_tag(K, N, TS, S, U, P);
    r.case_name                 = "pure_lupp";
    r.backend                   = backend_name();
    r.solver                    = K == SolverKind::static_ ? "static" : "dynamic";
    r.n                         = N;
    r.tile_size                 = TS;
    r.schedule                  = schedule_token(S);
    r.unroll_inner              = K == SolverKind::static_ ? (U ? "1" : "0") : "";
    r.res_matrix                = placement_token(P);
    r.res_rhs                   = placement_token(P);
    r.res_piv                   = placement_token(P);
    r.layout                    = P == Placement::dram ? "soa" : "";
    r.scalar                    = sizeof(T) == 8 ? "f64" : "f32";
    r.distribution              = distribution_name(dist);
    r.batch                     = batch;
    r.seed                      = opt.seed;
    r.ntpb                      = ntpb;
    r.blocks                    = blocks;
    r.dyn_smem_bytes            = 0;
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
    r.parity = verdict_parity_sample(host, ok, opt.parity_sample);
    r.be     = backward_error_sample(host, x_soa, ok, opt.validate_sample, r.validated_systems);
    std::vector<double> oot_values(oot.begin(), oot.end());
    r.oot = compute_stats(std::move(oot_values));
    for (const int count : oot)
        if (count > 0) ++r.oot_systems;
    r.gpu_name        = device.name;
    r.cc              = device.cc;
    r.driver_version  = device.driver_version;
    r.runtime_version = device.runtime_version;
    r.sm_count        = device.sm_count;
    r.smem_per_sm     = device.smem_per_sm;
    r.compiler        = compiler_string();
    r.tdls_version    = TDLS_VERSION_STRING;
    r.git_commit      = TDLS_BENCH_GIT_COMMIT;
    return r;
}



/* =====================================================================
   Registration: the grid of one dimension.
   ===================================================================== */

/// \brief Registers one variant.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S, bool U, SolverKind K, Placement P>
void register_variant() {
    registry().push_back(
        {make_tag(K, N, TS, S, U, P), [](const Options& opt, const Distribution dist) {
             return run_variant<T, N, TS, S, U, K, P>(opt, dist);
         }});
}

/// \brief Registers the variants of one (dimension, tile size,
/// schedule) cell: the static solver under both unroll settings and
/// both placements, the dynamic solver under both placements.
template<typename T, int N, int TS, tdls::TiledLUppSchedule S>
void register_schedule() {
    register_variant<T, N, TS, S, true, SolverKind::static_, Placement::reg>();
    register_variant<T, N, TS, S, true, SolverKind::static_, Placement::dram>();
    register_variant<T, N, TS, S, false, SolverKind::static_, Placement::reg>();
    register_variant<T, N, TS, S, false, SolverKind::static_, Placement::dram>();
    register_variant<T, N, TS, S, true, SolverKind::dynamic_, Placement::reg>();
    register_variant<T, N, TS, S, true, SolverKind::dynamic_, Placement::dram>();
}

/// \brief Registers both schedules of one (dimension, tile size) cell.
template<typename T, int N, int TS>
void register_tile_size() {
    register_schedule<T, N, TS, tdls::TiledLUppSchedule::RightLooking>();
    register_schedule<T, N, TS, tdls::TiledLUppSchedule::LeftLooking>();
}

/// \brief Registers the full grid of one dimension: tile sizes 1 to 6,
/// both schedules, both unroll settings, both solvers, both placements.
/// \tparam N system dimension
template<int N>
void register_dimension() {
    register_tile_size<double, N, 1>();
    register_tile_size<double, N, 2>();
    register_tile_size<double, N, 3>();
    register_tile_size<double, N, 4>();
    register_tile_size<double, N, 5>();
    register_tile_size<double, N, 6>();
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_CASES_PURE_LUPP_VARIANTS_TDLS_CUH
