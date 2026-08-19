#ifndef TDLS_BENCHMARKS_COMMON_OPTIONS_HPP
#define TDLS_BENCHMARKS_COMMON_OPTIONS_HPP



/// \file
/// \brief Command-line options of the benchmark executables.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every benchmark binary is self-sufficient: it enumerates its
/// registered variants, filters them by regex, measures them under the
/// selected input distributions and appends finished CSV rows. These
/// options drive that loop; campaign supervision (timeouts, resume)
/// lives outside, in scripts/run_sweep.sh.



#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>



namespace tdls_bench {



/// \brief Input distribution of the generated systems.
enum class Distribution {
    uniform, ///< entries in [-0.5, 0.5]: no out-of-tile pivoting triggers
    stress   ///< entries within 1.4x the pivot acceptance threshold of the
             ///< scalar type: nearly every system fires the out-of-tile
             ///< search, on about a third of its columns, and enough
             ///< candidates stay acceptable to exercise the early exit
             ///< (calibrated in double at n = 12, TS = 3: roughly 99% of the
             ///< systems, about a third of the columns)
};

/// \return the CSV name of a distribution
inline const char* distribution_name(const Distribution d) {
    return d == Distribution::uniform ? "default" : "stress";
}

/// \return the half-width of the entry distribution. The stress bound
/// follows the default acceptance threshold of the scalar type (1e-10
/// in double, 1e-4 in float), scaled by the calibrated 1.4 factor.
/// \tparam T scalar type of the benchmarked solver
template<typename T>
constexpr double distribution_bound(const Distribution d) {
    if (d == Distribution::uniform) return 0.5;
    return std::is_same_v<T, float> ? 1.4e-4 : 1.4e-10;
}



/// \brief Parsed command-line options, with the campaign defaults.
struct Options {
    bool list = false;                     ///< print the (filtered) variant tags and exit
    bool meta = false;                     ///< print the device/build metadata and exit
    std::string filter;                    ///< ECMAScript regex over the tags (empty: all)
    std::string csv_path;                  ///< CSV output file, appended (empty: no CSV)
    int batch               = 100000;      ///< systems per measurement
    int runs                = 5;           ///< timed kernel runs per variant
    int warmup              = 1;           ///< untimed warmup runs per variant
    int ntpb                = 0;           ///< threads per block (0: O+W heuristic)
    unsigned long long seed = 20260818ull; ///< generator seed of the batch
    double budget_s         = 0.0;         ///< per-variant time budget, seconds (0: off)
    int validate_sample     = 4096;        ///< systems checked for backward error (0: off)
    int parity_sample       = 32;          ///< systems checked against the reference LU (0: off)
    std::vector<Distribution> distributions = {Distribution::uniform,
                                               Distribution::stress}; ///< regimes to measure
};

/// \brief Prints the usage summary.
/// \param[in] program argv[0]
inline void print_usage(const char* program) {
    std::printf("usage: %s [options]\n"
                "  --list                 print the (filtered) variant tags and exit\n"
                "  --meta                 print the device/build metadata and exit\n"
                "  --filter <regex>       keep only the tags matching the regex\n"
                "  --csv <path>           append the result rows to this CSV file\n"
                "  --batch <n>            systems per measurement (default 100000)\n"
                "  --runs <n>             timed runs per variant (default 5)\n"
                "  --warmup <n>           untimed warmup runs (default 1)\n"
                "  --ntpb <n>             threads per block, 0 = auto by the O+W heuristic "
                "(default 0)\n"
                "  --seed <n>             batch generator seed (default 20260818)\n"
                "  --budget <s>           per-variant time budget in seconds: a variant whose\n"
                "                         probe prices the protocol above it gets fewer timed\n"
                "                         runs, never fewer than one (default 0 = off)\n"
                "  --validate <n>         systems checked for backward error, 0 = off "
                "(default 4096)\n"
                "  --parity <n>           systems checked against the reference LU, 0 = off "
                "(default 32)\n"
                "  --distribution <d>     default | stress | both (default both)\n",
                program);
}

/// \brief Parses the command line into an Options.
/// \param[in]  argc argument count of main()
/// \param[in]  argv argument vector of main()
/// \param[out] opt  parsed options
/// \return false on an unknown or malformed argument (usage is printed).
inline bool parse_options(const int argc, char* const* argv, Options& opt) {
    const auto integer = [](const char* s, int& out) {
        char* end      = nullptr;
        const long val = std::strtol(s, &end, 10);
        if (end == s || *end != '\0' || val < 0 || val > 2000000000L) return false;
        out = static_cast<int>(val);
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        bool ok         = true;
        if (std::strcmp(arg, "--list") == 0) {
            opt.list = true;
        } else if (std::strcmp(arg, "--meta") == 0) {
            opt.meta = true;
        } else if (std::strcmp(arg, "--filter") == 0) {
            const char* v = next();
            ok            = v != nullptr;
            if (ok) opt.filter = v;
        } else if (std::strcmp(arg, "--csv") == 0) {
            const char* v = next();
            ok            = v != nullptr;
            if (ok) opt.csv_path = v;
        } else if (std::strcmp(arg, "--batch") == 0) {
            const char* v = next();
            ok            = v != nullptr && integer(v, opt.batch) && opt.batch >= 1;
        } else if (std::strcmp(arg, "--runs") == 0) {
            const char* v = next();
            ok            = v != nullptr && integer(v, opt.runs) && opt.runs >= 1;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            const char* v = next();
            ok            = v != nullptr && integer(v, opt.warmup);
        } else if (std::strcmp(arg, "--ntpb") == 0) {
            const char* v = next();
            ok            = v != nullptr && integer(v, opt.ntpb);
        } else if (std::strcmp(arg, "--budget") == 0) {
            const char* v = next();
            ok            = v != nullptr;
            if (ok) {
                char* end    = nullptr;
                opt.budget_s = std::strtod(v, &end);
                ok           = end != v && *end == '\0' && opt.budget_s >= 0.0;
            }
        } else if (std::strcmp(arg, "--seed") == 0) {
            const char* v = next();
            ok            = v != nullptr;
            if (ok) opt.seed = std::strtoull(v, nullptr, 10);
        } else if (std::strcmp(arg, "--validate") == 0) {
            const char* v = next();
            ok            = v != nullptr && integer(v, opt.validate_sample);
        } else if (std::strcmp(arg, "--parity") == 0) {
            const char* v = next();
            ok            = v != nullptr && integer(v, opt.parity_sample);
        } else if (std::strcmp(arg, "--distribution") == 0) {
            const char* v = next();
            ok            = v != nullptr;
            if (ok) {
                if (std::strcmp(v, "default") == 0) {
                    opt.distributions = {Distribution::uniform};
                } else if (std::strcmp(v, "stress") == 0) {
                    opt.distributions = {Distribution::stress};
                } else if (std::strcmp(v, "both") == 0) {
                    opt.distributions = {Distribution::uniform, Distribution::stress};
                } else {
                    ok = false;
                }
            }
        } else {
            ok = false;
        }
        if (!ok) {
            std::printf("unrecognized or malformed argument: %s\n", arg);
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_OPTIONS_HPP
