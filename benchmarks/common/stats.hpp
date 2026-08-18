#ifndef TDLS_BENCHMARKS_COMMON_STATS_HPP
#define TDLS_BENCHMARKS_COMMON_STATS_HPP



/// \file
/// \brief Small descriptive-statistics helper of the benchmark harness.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.



#include <algorithm>
#include <cmath>
#include <vector>



namespace tdls_bench {



/// \brief Descriptive statistics of a sample.
struct Stats {
    double min    = 0.0;  ///< smallest value
    double max    = 0.0;  ///< largest value
    double mean   = 0.0;  ///< arithmetic mean
    double median = 0.0;  ///< median (mean of the middle pair for even sizes)
    double stddev = 0.0;  ///< population standard deviation
    bool empty    = true; ///< true when the sample had no value
};

/// \brief Computes the statistics of a sample (empty samples allowed).
/// \param[in] values the sample
/// \return the statistics
inline Stats compute_stats(std::vector<double> values) {
    Stats s;
    if (values.empty()) return s;
    s.empty = false;
    std::sort(values.begin(), values.end());
    s.min             = values.front();
    s.max             = values.back();
    const auto count  = values.size();
    const auto middle = count / 2;
    s.median   = (count % 2 == 1) ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
    double acc = 0.0;
    for (const double v : values)
        acc += v;
    s.mean     = acc / static_cast<double>(count);
    double var = 0.0;
    for (const double v : values)
        var += (v - s.mean) * (v - s.mean);
    s.stddev = std::sqrt(var / static_cast<double>(count));
    return s;
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_STATS_HPP
