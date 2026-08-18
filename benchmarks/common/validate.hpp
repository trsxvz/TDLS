#ifndef TDLS_BENCHMARKS_COMMON_VALIDATE_HPP
#define TDLS_BENCHMARKS_COMMON_VALIDATE_HPP



/// \file
/// \brief Numerical validation of the benchmark measurements.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A throughput number is only worth reporting next to the proof that
/// the kernel solved its systems: every measurement carries the
/// normwise backward error of a sample of solutions, and the verdict
/// parity of a smaller sample against the independent reference LU of
/// the test suites (tests/common/reference_lu.hpp, the anchor of the
/// whole project). Exhaustive correctness is the job of the test
/// suites, not of the harness: samples keep the campaign cost bounded.



#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "generators.hpp"
#include "reference_lu.hpp"
#include "stats.hpp"



namespace tdls_bench {



/// \brief Backward-error statistics of the first `sample` solved
/// systems of a batch.
/// \tparam T scalar type
/// \param[in]  host    generated systems (contiguous per-system storage)
/// \param[in]  x_soa   solutions in SoA form, element stride = batch count
/// \param[in]  ok      per-system non-singular verdicts
/// \param[in]  sample  number of systems to check (clamped to the batch)
/// \param[out] checked number of systems that entered the statistics
/// \return the backward-error statistics
template<typename T>
inline Stats backward_error_sample(const tdls_tests::SystemBatch<T>& host,
                                   const std::vector<T>& x_soa, const std::vector<int>& ok,
                                   const int sample, int& checked) {
    const int count = std::min(sample, host.count);
    std::vector<double> errors;
    errors.reserve(static_cast<std::size_t>(count));
    std::vector<T> x(host.n);
    for (int s = 0; s < count; ++s) {
        if (ok[static_cast<std::size_t>(s)] == 0) continue;
        for (int i = 0; i < host.n; ++i)
            x[i] = x_soa[static_cast<std::size_t>(i) * host.count + s];
        errors.push_back(tdls_tests::backward_error(host.matrix(s), x.data(), host.rhs(s), host.n));
    }
    checked = static_cast<int>(errors.size());
    return compute_stats(std::move(errors));
}

/// \brief Verdict parity of the first `sample` systems against the
/// reference LU.
/// \tparam T scalar type
/// \param[in] host   generated systems (contiguous per-system storage)
/// \param[in] ok     per-system non-singular verdicts of the benchmark
/// \param[in] sample number of systems to check (clamped to the batch)
/// \return "match", "mismatch", or "" when the check is disabled
template<typename T>
inline std::string verdict_parity_sample(const tdls_tests::SystemBatch<T>& host,
                                         const std::vector<int>& ok, const int sample) {
    if (sample <= 0) return "";
    const int count = std::min(sample, host.count);
    std::vector<T> A(static_cast<std::size_t>(host.n) * host.n);
    std::vector<int> piv(host.n);
    for (int s = 0; s < count; ++s) {
        std::copy(host.matrix(s), host.matrix(s) + host.n * host.n, A.begin());
        const bool reference_ok = tdls_tests::reference_factorize(A.data(), piv.data(), host.n);
        if (reference_ok != (ok[static_cast<std::size_t>(s)] != 0)) return "mismatch";
    }
    return "match";
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_VALIDATE_HPP
