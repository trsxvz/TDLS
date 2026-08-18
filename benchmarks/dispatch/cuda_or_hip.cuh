#ifndef TDLS_BENCHMARKS_DISPATCH_CUDA_OR_HIP_CUH
#define TDLS_BENCHMARKS_DISPATCH_CUDA_OR_HIP_CUH



/// \file
/// \brief CUDA/HIP dispatch layer of the benchmark harness: kernel
/// timing, self-reported kernel metrics and device identification.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Single source in the common CUDA/HIP dialect, on top of the
/// portability aliases of examples/common/gpu_runtime.hpp. Everything
/// reported here comes from the runtime API of the process itself: no
/// external profiler is involved (occupancy is therefore the
/// theoretical bound, and the local-memory column is the spill
/// footprint the compiler decided at build time). Deep profiling stays
/// possible offline: any variant can be rerun in isolation under ncu
/// or rocprof through the --filter option of the binaries.



#include <cstddef>
#include <string>

#include <gpu_runtime.hpp>



// Aliases the few runtime names gpu_runtime.hpp does not cover.
#if defined(__HIP__) || defined(__HIPCC__)
using gpuDeviceProp_t                           = hipDeviceProp_t;
using gpuFuncAttributes_t                       = hipFuncAttributes;
using gpuEvent_t                                = hipEvent_t;
constexpr gpuMemcpyKind gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto gpuFuncAttributeMaxDynSmem       = hipFuncAttributeMaxDynamicSharedMemorySize;
#else
using gpuDeviceProp_t                           = cudaDeviceProp;
using gpuFuncAttributes_t                       = cudaFuncAttributes;
using gpuEvent_t                                = cudaEvent_t;
constexpr gpuMemcpyKind gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuFuncAttributeMaxDynSmem       = cudaFuncAttributeMaxDynamicSharedMemorySize;
#endif



namespace tdls_bench {



/// \return the dispatch backend name of this build
inline const char* backend_name() {
#if defined(__HIP__) || defined(__HIPCC__)
    return "hip";
#else
    return "cuda";
#endif
}

/// \return the device compiler identification of this build
inline std::string compiler_string() {
#if defined(__HIP__) || defined(__HIPCC__)
    return "hipcc";
#elif defined(__CUDACC_VER_MAJOR__)
    return "nvcc-" + std::to_string(__CUDACC_VER_MAJOR__) + "." +
           std::to_string(__CUDACC_VER_MINOR__);
#else
    return "unknown";
#endif
}



/// \brief Identity of the selected device, read from the runtime API.
struct DeviceInfo {
    std::string name;               ///< device name
    std::string cc;                 ///< compute capability, "major.minor"
    int driver_version         = 0; ///< driver version
    int runtime_version        = 0; ///< runtime library version
    int sm_count               = 0; ///< multiprocessors
    int max_threads_per_sm     = 0; ///< resident thread capacity per multiprocessor
    std::size_t smem_per_sm    = 0; ///< shared memory per multiprocessor
    std::size_t smem_per_block = 0; ///< largest dynamic shared allocation of one block
    std::size_t total_mem      = 0; ///< device memory
};

/// \return the DeviceInfo of the current device
inline DeviceInfo query_device_info() {
    DeviceInfo info;
    gpuDeviceProp_t prop{};
    GPU_CHECK(TDLS_EXAMPLES_GPU_API(GetDeviceProperties)(&prop, 0));
    info.name               = prop.name;
    info.cc                 = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    info.sm_count           = prop.multiProcessorCount;
    info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info.total_mem          = prop.totalGlobalMem;
#if defined(__HIP__) || defined(__HIPCC__)
    info.smem_per_sm    = prop.maxSharedMemoryPerMultiProcessor;
    info.smem_per_block = prop.sharedMemPerBlock;
#else
    info.smem_per_sm    = prop.sharedMemPerMultiprocessor;
    info.smem_per_block = prop.sharedMemPerBlockOptin;
#endif
    GPU_CHECK(TDLS_EXAMPLES_GPU_API(DriverGetVersion)(&info.driver_version));
    GPU_CHECK(TDLS_EXAMPLES_GPU_API(RuntimeGetVersion)(&info.runtime_version));
    return info;
}

/// \return the free device memory, in bytes
inline std::size_t query_free_memory() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    GPU_CHECK(TDLS_EXAMPLES_GPU_API(MemGetInfo)(&free_bytes, &total_bytes));
    return free_bytes;
}

/// \brief Non-fatal device allocation: an exhausted device memory is a
/// legitimate campaign outcome (recorded as a skipped variant), not a
/// crash.
/// \tparam T element type
/// \param[out] ptr   receives the device pointer (null on failure)
/// \param[in]  bytes allocation size in bytes
/// \return false when the allocation failed
template<typename T>
[[nodiscard]] inline bool gpu_try_malloc(T** ptr, const std::size_t bytes) {
    *ptr = nullptr;
    if (TDLS_EXAMPLES_GPU_API(Malloc)(reinterpret_cast<void**>(ptr), bytes) == gpuSuccess)
        return true;
    TDLS_EXAMPLES_GPU_API(GetLastError)(); // clear the sticky error state
    *ptr = nullptr;
    return false;
}



/// \brief Launch configuration selected for one variant.
struct LaunchChoice {
    bool feasible        = false; ///< false: the variant cannot run on this device
    int ntpb             = 0;     ///< selected threads per block
    std::size_t dyn_smem = 0;     ///< dynamic shared memory per block
};

/// \brief Selects the threads-per-block of a kernel: the candidate of
/// {32, 64, 96, 128, 256} maximizing theoretical occupancy plus wave
/// fill.
///
/// Shared-memory variants consume dyn_smem = ntpb x bytes_per_thread
/// per block. When no candidate of the primary list fits the per-block
/// budget (large dimensions), the sub-warp candidates {16, 8, 4, 2, 1}
/// are probed instead; when even one thread per block does not fit,
/// the variant is reported infeasible and the caller records a skip.
/// A forced ntpb bypasses the scoring but keeps the feasibility check.
/// \tparam Kernel kernel function-pointer type
/// \param[in] kernel           the kernel
/// \param[in] bytes_per_thread dynamic shared memory per thread (0: none)
/// \param[in] batch            systems of the launch (wave-fill term)
/// \param[in] device           device identity
/// \param[in] forced_ntpb      0: apply the heuristic; else this value
/// \return the selected launch configuration
template<typename Kernel>
inline LaunchChoice choose_launch(Kernel kernel, const std::size_t bytes_per_thread,
                                  const int batch, const DeviceInfo& device,
                                  const int forced_ntpb) {
    if (bytes_per_thread > 0) {
        // Raise the dynamic shared-memory cap of the kernel to the
        // device limit (the default cap is lower on NVIDIA devices).
        TDLS_EXAMPLES_GPU_API(FuncSetAttribute)(reinterpret_cast<const void*>(kernel),
                                                gpuFuncAttributeMaxDynSmem,
                                                static_cast<int>(device.smem_per_block));
        TDLS_EXAMPLES_GPU_API(GetLastError)(); // tolerated on devices without the cap
    }
    const auto evaluate = [&](const int candidate, double& score, std::size_t& dyn_smem) {
        dyn_smem = bytes_per_thread * static_cast<std::size_t>(candidate);
        if (bytes_per_thread > 0 && dyn_smem > device.smem_per_block) return false;
        int blocks_per_sm = 0;
        if (TDLS_EXAMPLES_GPU_API(OccupancyMaxActiveBlocksPerMultiprocessor)(
                &blocks_per_sm, kernel, candidate, dyn_smem) != gpuSuccess) {
            TDLS_EXAMPLES_GPU_API(GetLastError)();
            return false;
        }
        if (blocks_per_sm < 1) return false;
        const double occupancy =
            static_cast<double>(blocks_per_sm) * candidate / device.max_threads_per_sm;
        const long blocks = (batch + candidate - 1) / candidate;
        const long wave   = static_cast<long>(blocks_per_sm) * device.sm_count;
        const long waves  = (blocks + wave - 1) / wave;
        const double fill = static_cast<double>(blocks) / static_cast<double>(waves * wave);
        score             = occupancy + fill;
        return true;
    };
    LaunchChoice choice;
    if (forced_ntpb > 0) {
        double score         = 0.0;
        std::size_t dyn_smem = 0;
        if (evaluate(forced_ntpb, score, dyn_smem)) choice = {true, forced_ntpb, dyn_smem};
        return choice;
    }
    constexpr int primary[] = {32, 64, 96, 128, 256};
    constexpr int subwarp[] = {16, 8, 4, 2, 1};
    double best_score       = -1.0;
    for (const int candidate : primary) {
        double score         = 0.0;
        std::size_t dyn_smem = 0;
        if (evaluate(candidate, score, dyn_smem) && score > best_score) {
            best_score = score;
            choice     = {true, candidate, dyn_smem};
        }
    }
    if (!choice.feasible) {
        for (const int candidate : subwarp) {
            double score         = 0.0;
            std::size_t dyn_smem = 0;
            if (evaluate(candidate, score, dyn_smem) && score > best_score) {
                best_score = score;
                choice     = {true, candidate, dyn_smem};
            }
        }
    }
    return choice;
}



/// \brief Self-reported metrics of one kernel under one launch
/// configuration.
struct KernelMetrics {
    int regs_per_thread                = 0;   ///< registers per thread
    std::size_t local_bytes_per_thread = 0;   ///< local (spill) memory per thread
    std::size_t static_smem_bytes      = 0;   ///< static shared memory per block
    double theoretical_occupancy_pct   = 0.0; ///< occupancy upper bound
    double wave_fill_pct               = 0.0; ///< fill ratio of the last wave
};

/// \brief Queries the metrics of a kernel for a launch configuration.
/// \tparam Kernel kernel function-pointer type
/// \param[in] kernel   the kernel
/// \param[in] ntpb     threads per block
/// \param[in] dyn_smem dynamic shared memory per block
/// \param[in] blocks   blocks launched
/// \param[in] device   device identity
/// \return the metrics
template<typename Kernel>
inline KernelMetrics kernel_metrics(Kernel kernel, const int ntpb, const std::size_t dyn_smem,
                                    const int blocks, const DeviceInfo& device) {
    KernelMetrics metrics;
    gpuFuncAttributes_t attributes{};
    GPU_CHECK(TDLS_EXAMPLES_GPU_API(FuncGetAttributes)(&attributes,
                                                       reinterpret_cast<const void*>(kernel)));
    metrics.regs_per_thread        = attributes.numRegs;
    metrics.local_bytes_per_thread = attributes.localSizeBytes;
    metrics.static_smem_bytes      = attributes.sharedSizeBytes;
    int blocks_per_sm              = 0;
    GPU_CHECK(TDLS_EXAMPLES_GPU_API(OccupancyMaxActiveBlocksPerMultiprocessor)(
        &blocks_per_sm, kernel, ntpb, dyn_smem));
    if (device.max_threads_per_sm > 0)
        metrics.theoretical_occupancy_pct =
            100.0 * blocks_per_sm * ntpb / device.max_threads_per_sm;
    const long wave = static_cast<long>(blocks_per_sm) * device.sm_count;
    if (wave > 0) {
        const long waves      = (blocks + wave - 1) / wave;
        metrics.wave_fill_pct = 100.0 * blocks / static_cast<double>(waves * wave);
    }
    return metrics;
}



/// \brief Event-based kernel timer.
class GpuTimer {
  public:
    GpuTimer() {
        GPU_CHECK(TDLS_EXAMPLES_GPU_API(EventCreate)(&start_));
        GPU_CHECK(TDLS_EXAMPLES_GPU_API(EventCreate)(&stop_));
    }
    ~GpuTimer() {
        TDLS_EXAMPLES_GPU_API(EventDestroy)(start_);
        TDLS_EXAMPLES_GPU_API(EventDestroy)(stop_);
    }
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    /// \brief Times one launch closure, synchronizing on completion.
    /// \param[in] launch closure enqueueing the kernel
    /// \return the elapsed device time in milliseconds
    template<typename Launch>
    double time_ms(Launch&& launch) {
        GPU_CHECK(TDLS_EXAMPLES_GPU_API(EventRecord)(start_, nullptr));
        launch();
        GPU_CHECK(TDLS_EXAMPLES_GPU_API(EventRecord)(stop_, nullptr));
        GPU_CHECK(TDLS_EXAMPLES_GPU_API(EventSynchronize)(stop_));
        float elapsed = 0.0f;
        GPU_CHECK(TDLS_EXAMPLES_GPU_API(EventElapsedTime)(&elapsed, start_, stop_));
        return static_cast<double>(elapsed);
    }

  private:
    gpuEvent_t start_; //!< launch-side event
    gpuEvent_t stop_;  //!< completion-side event
};



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_DISPATCH_CUDA_OR_HIP_CUH
