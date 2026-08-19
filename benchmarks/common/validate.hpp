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
///
/// The sampled inputs are REGENERATED on the host from the
/// counter-based generator (common/batch.hpp) while the solutions are
/// gathered from the downloaded SoA buffer: on top of validating the
/// solves, this cross-checks that the host and device generators
/// produce identical values, since any divergence would explode the
/// backward error and break the parity.



#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "batch.hpp"
#include "reference_lu.hpp"
#include "stats.hpp"



namespace tdls_bench {



namespace detail {

/// \brief Gathers one system of an SoA buffer into contiguous scratch.
/// \tparam T scalar type
/// \param[in]  soa      SoA buffer, element stride = batch
/// \param[in]  batch    number of systems in the buffer
/// \param[in]  s        system index
/// \param[in]  elements elements per system
/// \param[out] out      contiguous destination
template<typename T>
inline void gather_system(const std::vector<T>& soa, const int batch, const int s,
                          const int elements, std::vector<T>& out) {
    for (int e = 0; e < elements; ++e)
        out[static_cast<std::size_t>(e)] = soa[static_cast<std::size_t>(e) * batch + s];
}

} // namespace detail

/// \brief Backward-error statistics of the first `sample` solved
/// systems of a batch.
/// \tparam T scalar type
/// \param[in]  n       system dimension
/// \param[in]  batch   number of systems
/// \param[in]  seed    campaign seed (inputs are regenerated from it)
/// \param[in]  bound   half-width of the input distribution
/// \param[in]  y_soa   solution batch, SoA
/// \param[in]  ok      per-system non-singular verdicts
/// \param[in]  sample  number of systems to check (clamped to the batch)
/// \param[out] checked number of systems that entered the statistics
/// \return the backward-error statistics
template<typename T>
inline Stats backward_error_sample(const int n, const int batch, const std::uint64_t seed,
                                   const double bound, const std::vector<T>& y_soa,
                                   const std::vector<int>& ok, const int sample, int& checked) {
    const int count = std::min(sample, batch);
    std::vector<double> errors;
    errors.reserve(static_cast<std::size_t>(count));
    std::vector<T> A(static_cast<std::size_t>(n) * n);
    std::vector<T> b(n);
    std::vector<T> y(n);
    for (int s = 0; s < count; ++s) {
        if (ok[static_cast<std::size_t>(s)] == 0) continue;
        generate_matrix_host(n, seed, bound, s, A.data());
        generate_rhs_host(n, seed, bound, s, b.data());
        detail::gather_system(y_soa, batch, s, n, y);
        errors.push_back(tdls_tests::backward_error(A.data(), y.data(), b.data(), n));
    }
    checked = static_cast<int>(errors.size());
    return compute_stats(std::move(errors));
}

/// \brief Verdict parity of the first `sample` systems against the
/// reference LU.
/// \tparam T scalar type
/// \param[in] n      system dimension
/// \param[in] batch  number of systems
/// \param[in] seed   campaign seed (inputs are regenerated from it)
/// \param[in] bound  half-width of the input distribution
/// \param[in] ok     per-system non-singular verdicts of the benchmark
/// \param[in] sample number of systems to check (clamped to the batch)
/// \return "match", "mismatch", or "" when the check is disabled
template<typename T>
inline std::string verdict_parity_sample(const int n, const int batch, const std::uint64_t seed,
                                         const double bound, const std::vector<int>& ok,
                                         const int sample) {
    if (sample <= 0) return "";
    const int count = std::min(sample, batch);
    std::vector<T> A(static_cast<std::size_t>(n) * n);
    std::vector<int> piv(n);
    for (int s = 0; s < count; ++s) {
        generate_matrix_host<T>(n, seed, bound, s, A.data());
        const bool reference_ok = tdls_tests::reference_factorize(A.data(), piv.data(), n);
        if (reference_ok != (ok[static_cast<std::size_t>(s)] != 0)) return "mismatch";
    }
    return "match";
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_VALIDATE_HPP
