/// \file
/// \brief Entry point of the pure-LUpp benchmark executable.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Lists, filters and measures the registered variants; every
/// measurement appends one finished CSV row and prints a one-line
/// human summary. The binary returns 77 when no device is present
/// (registered in ctest as a skipped test, as the GPU examples do).
/// Campaign supervision (timeouts, resume, indexing) lives outside, in
/// scripts/run_sweep.sh.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

#include <tdls/core/version.hpp>

#ifndef TDLS_BENCH_GIT_COMMIT
#define TDLS_BENCH_GIT_COMMIT "unknown"
#endif

#include "common/options.hpp"
#include "common/record.hpp"
#include "common/registry.hpp"
#include "dispatch/cuda_or_hip.cuh"

int main(int argc, char** argv) {
    tdls_bench::Options opt;
    if (!tdls_bench::parse_options(argc, argv, opt)) return 2;

    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    if (opt.meta) {
        const auto device = tdls_bench::query_device_info();
        std::printf("backend          : %s\n", tdls_bench::backend_name());
        std::printf("gpu              : %s\n", device.name.c_str());
        std::printf("compute_capability: %s\n", device.cc.c_str());
        std::printf("driver_version   : %d\n", device.driver_version);
        std::printf("runtime_version  : %d\n", device.runtime_version);
        std::printf("sm_count         : %d\n", device.sm_count);
        std::printf("smem_per_sm      : %zu\n", device.smem_per_sm);
        std::printf("total_mem        : %zu\n", device.total_mem);
        std::printf("compiler         : %s\n", tdls_bench::compiler_string().c_str());
        std::printf("tdls_version     : %s\n", TDLS_VERSION_STRING);
        std::printf("git_commit       : %s\n", TDLS_BENCH_GIT_COMMIT);
        return 0;
    }

    // Registration order follows link order: sort for a deterministic
    // enumeration whatever the dimension list of the build.
    auto& variants = tdls_bench::registry();
    std::sort(
        variants.begin(), variants.end(),
        [](const tdls_bench::Variant& a, const tdls_bench::Variant& b) { return a.tag < b.tag; });

    std::regex filter;
    try {
        filter = std::regex(opt.filter.empty() ? ".*" : opt.filter);
    } catch (const std::regex_error&) {
        std::printf("invalid filter regex: %s\n", opt.filter.c_str());
        return 2;
    }

    if (opt.list) {
        for (const auto& variant : variants)
            if (std::regex_search(variant.tag, filter)) std::printf("%s\n", variant.tag.c_str());
        return 0;
    }

    std::ofstream csv;
    if (!opt.csv_path.empty()) {
        const bool fresh =
            !std::filesystem::exists(opt.csv_path) || std::filesystem::file_size(opt.csv_path) == 0;
        csv.open(opt.csv_path, std::ios::app);
        if (!csv) {
            std::printf("cannot open CSV output file: %s\n", opt.csv_path.c_str());
            return 2;
        }
        if (fresh) csv << tdls_bench::csv_header() << '\n';
    }

    int measured   = 0;
    int mismatches = 0;
    for (const auto& variant : variants) {
        if (!std::regex_search(variant.tag, filter)) continue;
        for (const auto dist : opt.distributions) {
            const tdls_bench::Record record = variant.run(opt, dist);
            if (record.parity == "mismatch") ++mismatches;
            if (csv.is_open()) {
                csv << tdls_bench::csv_row(record) << '\n';
                csv.flush();
            }
            if (record.status != "ok") {
                std::printf("[   %-8s] %-46s %-7s\n", record.status.c_str(), record.tag.c_str(),
                            record.distribution.c_str());
            } else {
                const std::string parity = record.parity.empty() ? "" : "parity=" + record.parity;
                std::printf("[%10.3f ms] %-46s %-7s ntpb=%-4d %10.3e sys/s  be_max=%-9.3g %s\n",
                            record.t_ms.median, record.tag.c_str(), record.distribution.c_str(),
                            record.ntpb, record.systems_per_s,
                            record.be.empty ? 0.0 : record.be.max, parity.c_str());
            }
            std::fflush(stdout);
            ++measured;
        }
    }
    if (measured == 0) {
        std::printf("no variant matches the filter '%s'\n", opt.filter.c_str());
        return 1;
    }
    // A throughput number over wrong solutions must not look like a
    // success: verdict-parity failures fail the whole invocation (and
    // with it the ctest smoke test and any campaign run).
    if (mismatches > 0) {
        std::printf("%d measurement(s) failed the reference-LU parity check\n", mismatches);
        return 3;
    }
    return 0;
}
