#ifndef TDLS_BENCHMARKS_COMMON_RECORD_HPP
#define TDLS_BENCHMARKS_COMMON_RECORD_HPP



/// \file
/// \brief Result record of one benchmark measurement and its CSV form.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// One Record is one CSV row, and every row is self-sufficient: the
/// sweep axes, the launch configuration, the self-reported kernel
/// metrics, the timings, the numerical verdicts and the environment all
/// travel together, so a results file needs no side table to be
/// analyzed. Cells never contain commas; not-applicable cells are left
/// empty. The raw run timings are kept (semicolon-separated inside one
/// cell) next to their aggregates, so the choice of indicator stays
/// open downstream.



#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "stats.hpp"



namespace tdls_bench {



/// \brief Schema identifier written in every row; bump it on any column
/// change so downstream scripts can dispatch.
inline constexpr int csv_schema_version = 2;

/// \brief One measurement: a variant under one distribution.
struct Record {
    // Identity
    std::string timestamp_utc; ///< ISO-8601 UTC time of the measurement
    std::string tag;           ///< variant tag (primary key with distribution)

    // Sweep axes
    std::string case_name;          ///< benchmark case (pure_lupp, ...)
    std::string backend;            ///< dispatch backend (cuda, hip, ...)
    std::string solver;             ///< static | dynamic
    int n         = 0;              ///< system dimension
    int tile_size = 0;              ///< TiledLUpp tile size
    std::string schedule;           ///< rl | ll
    std::string unroll_inner;       ///< 1 | 0 | empty (not applicable)
    std::string res_matrix;         ///< reg | shmem | dram
    std::string res_rhs;            ///< reg | shmem | dram
    std::string res_piv;            ///< reg | shmem | dram
    std::string layout;             ///< soa | aos | aosoa | empty (register residency)
    std::string scalar;             ///< f64 | f32
    std::string distribution;       ///< default | stress
    int batch               = 0;    ///< systems per measurement
    unsigned long long seed = 0;    ///< generator seed
    std::string status      = "ok"; ///< ok | skip_smem | skip_dram | skip_host | skip_offset32

    // Launch configuration
    int ntpb                   = 0; ///< threads per block
    int blocks                 = 0; ///< blocks launched
    std::size_t dyn_smem_bytes = 0; ///< dynamic shared memory per block

    // Self-reported kernel metrics (runtime API, no external profiler)
    int regs_per_thread                = 0;   ///< registers per thread
    std::size_t local_bytes_per_thread = 0;   ///< local (spill) memory per thread
    std::size_t static_smem_bytes      = 0;   ///< static shared memory per block
    double theoretical_occupancy_pct   = 0.0; ///< occupancy upper bound
    double wave_fill_pct               = 0.0; ///< fill ratio of the last wave

    // Timings (milliseconds)
    std::vector<double> t_runs_ms; ///< raw timed runs, in order
    Stats t_ms;                    ///< statistics over t_runs_ms
    double systems_per_s = 0.0;    ///< batch / median run time

    // Numerical verdicts
    int solved = 0;            ///< systems reported non-singular
    std::string parity;        ///< match | mismatch | empty (check off)
    int validated_systems = 0; ///< systems entering the backward-error stats
    Stats be;                  ///< backward-error statistics
    int oot_systems = 0;       ///< systems that triggered the out-of-tile search
    Stats oot;                 ///< out-of-tile counter statistics (all systems)

    // Environment
    std::string gpu_name;        ///< device name
    std::string cc;              ///< compute capability (major.minor)
    int driver_version      = 0; ///< driver version as reported by the runtime
    int runtime_version     = 0; ///< runtime library version
    int sm_count            = 0; ///< multiprocessors on the device
    std::size_t smem_per_sm = 0; ///< shared memory per multiprocessor
    std::string compiler;        ///< device compiler identification
    std::string tdls_version;    ///< TDLS_VERSION_STRING
    std::string git_commit;      ///< source revision of the harness build
};

/// \return the current UTC time in ISO-8601 form
inline std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

/// \return the CSV header line (no trailing newline)
inline std::string csv_header() {
    return "schema_version,timestamp_utc,tag,"
           "case,backend,solver,n,tile_size,schedule,unroll_inner,"
           "res_matrix,res_rhs,res_piv,layout,scalar,distribution,batch,seed,status,"
           "ntpb,blocks,dyn_smem_bytes,"
           "regs_per_thread,local_bytes_per_thread,static_smem_bytes,"
           "theoretical_occupancy_pct,wave_fill_pct,"
           "t_runs_ms,t_median_ms,t_mean_ms,t_stddev_ms,t_min_ms,t_max_ms,systems_per_s,"
           "solved,parity,validated_systems,be_min,be_max,be_mean,be_median,be_stddev,"
           "oot_systems,oot_min,oot_max,oot_mean,oot_median,"
           "gpu_name,cc,driver_version,runtime_version,sm_count,smem_per_sm,"
           "compiler,tdls_version,git_commit";
}

namespace detail {

/// \return a floating-point cell (%.6g), empty for an empty statistic
inline std::string cell(const double value, const bool is_empty = false) {
    if (is_empty) return "";
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

} // namespace detail

/// \return the CSV row of a record (no trailing newline)
inline std::string csv_row(const Record& r) {
    std::string raw_times;
    for (std::size_t i = 0; i < r.t_runs_ms.size(); ++i) {
        if (i > 0) raw_times += ';';
        raw_times += detail::cell(r.t_runs_ms[i]);
    }
    std::string row;
    const auto add = [&row](const std::string& value) {
        if (!row.empty()) row += ',';
        row += value;
    };
    add(std::to_string(csv_schema_version));
    add(r.timestamp_utc);
    add(r.tag);
    add(r.case_name);
    add(r.backend);
    add(r.solver);
    add(std::to_string(r.n));
    add(std::to_string(r.tile_size));
    add(r.schedule);
    add(r.unroll_inner);
    add(r.res_matrix);
    add(r.res_rhs);
    add(r.res_piv);
    add(r.layout);
    add(r.scalar);
    add(r.distribution);
    add(std::to_string(r.batch));
    add(std::to_string(r.seed));
    add(r.status);
    add(std::to_string(r.ntpb));
    add(std::to_string(r.blocks));
    add(std::to_string(r.dyn_smem_bytes));
    add(std::to_string(r.regs_per_thread));
    add(std::to_string(r.local_bytes_per_thread));
    add(std::to_string(r.static_smem_bytes));
    add(detail::cell(r.theoretical_occupancy_pct));
    add(detail::cell(r.wave_fill_pct));
    add(raw_times);
    add(detail::cell(r.t_ms.median, r.t_ms.empty));
    add(detail::cell(r.t_ms.mean, r.t_ms.empty));
    add(detail::cell(r.t_ms.stddev, r.t_ms.empty));
    add(detail::cell(r.t_ms.min, r.t_ms.empty));
    add(detail::cell(r.t_ms.max, r.t_ms.empty));
    add(detail::cell(r.systems_per_s));
    add(std::to_string(r.solved));
    add(r.parity);
    add(std::to_string(r.validated_systems));
    add(detail::cell(r.be.min, r.be.empty));
    add(detail::cell(r.be.max, r.be.empty));
    add(detail::cell(r.be.mean, r.be.empty));
    add(detail::cell(r.be.median, r.be.empty));
    add(detail::cell(r.be.stddev, r.be.empty));
    add(std::to_string(r.oot_systems));
    add(detail::cell(r.oot.min, r.oot.empty));
    add(detail::cell(r.oot.max, r.oot.empty));
    add(detail::cell(r.oot.mean, r.oot.empty));
    add(detail::cell(r.oot.median, r.oot.empty));
    add(r.gpu_name);
    add(r.cc);
    add(std::to_string(r.driver_version));
    add(std::to_string(r.runtime_version));
    add(std::to_string(r.sm_count));
    add(std::to_string(r.smem_per_sm));
    add(r.compiler);
    add(r.tdls_version);
    add(r.git_commit);
    return row;
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_RECORD_HPP
