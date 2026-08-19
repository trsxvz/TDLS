#ifndef TDLS_BENCHMARKS_COMMON_BATCH_HPP
#define TDLS_BENCHMARKS_COMMON_BATCH_HPP



/// \file
/// \brief Counter-based input generator of the benchmark harness.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The benchmark inputs are drawn by a stateless counter-based
/// generator (the SplitMix64 finalizer, the same mixing constants as
/// the hash01 of the GPU examples): entry = f(seed, stream, index),
/// uniform over [-bound, bound] through the fixed 53-bit mapping. No
/// sequential state means the batch can be generated anywhere, in any
/// order, in parallel: on the device (the GPU harness materializes and
/// restores its batches in place, no host copy, no host-to-device
/// transfer), on the host (validation regenerates its sampled systems;
/// future CPU cases will generate their batches the same way). The
/// mixing is pure 64-bit integer arithmetic and the mapping one exact
/// double multiply, so host and device produce identical values; a
/// divergence would fail the backward-error and reference-parity
/// validation of every measurement.
///
/// Draw indices are logical, layout- and batch-size-independent:
/// matrix entry e of system s draws (stream 0, index s*n*n + e); rhs
/// entry i of system s draws (stream 1, index s*n + i). Float batches
/// cast the double draws, exactly as the previous generator did.
///
/// The test suites are untouched: their mt19937_64 streams remain the
/// bitwise anchor of the library tests. This generator only defines
/// the benchmark inputs, and is recorded by the seed column of the
/// result rows.



#include <cstdint>

#include <tdls/core/macros.hpp>



namespace tdls_bench {



/// \brief Stream tag of the matrix entries.
inline constexpr std::uint64_t matrix_stream = 0;
/// \brief Stream tag of the right-hand-side entries.
inline constexpr std::uint64_t rhs_stream = 1;

/// \brief SplitMix64 finalizer.
/// \param[in] x state to mix
/// \return the mixed value
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/// \brief One draw of the counter-based stream, uniform over
/// [-bound, bound].
/// \param[in] seed   campaign seed (the CSV seed column)
/// \param[in] stream stream tag (matrix_stream or rhs_stream)
/// \param[in] index  logical index of the entry within its stream
/// \param[in] bound  half-width of the distribution
/// \return the drawn value
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr double counter_draw(const std::uint64_t seed,
                                                                const std::uint64_t stream,
                                                                const std::uint64_t index,
                                                                const double bound) noexcept {
    const std::uint64_t base = splitmix64(seed ^ ((stream + 1) * 0x9e3779b97f4a7c15ull));
    const std::uint64_t z    = splitmix64(base + (index + 1) * 0x9e3779b97f4a7c15ull);
    const double unit        = static_cast<double>(z >> 11) * 0x1.0p-53; // [0, 1)
    return (2.0 * unit - 1.0) * bound;
}

/// \brief Regenerates the matrix of one system on the host, contiguous
/// row-major (validation-side counterpart of the device generation).
/// \tparam T scalar type
/// \param[in]  n     system dimension
/// \param[in]  seed  campaign seed
/// \param[in]  bound half-width of the distribution
/// \param[in]  s     system index
/// \param[out] A     destination, n*n elements
template<typename T>
inline void generate_matrix_host(const int n, const std::uint64_t seed, const double bound,
                                 const int s, T* A) {
    const std::uint64_t first = static_cast<std::uint64_t>(s) * n * n;
    for (int e = 0; e < n * n; ++e)
        A[e] = static_cast<T>(counter_draw(seed, matrix_stream, first + e, bound));
}

/// \brief Regenerates the right-hand side of one system on the host.
/// \tparam T scalar type
/// \param[in]  n     system dimension
/// \param[in]  seed  campaign seed
/// \param[in]  bound half-width of the distribution
/// \param[in]  s     system index
/// \param[out] b     destination, n elements
template<typename T>
inline void generate_rhs_host(const int n, const std::uint64_t seed, const double bound,
                              const int s, T* b) {
    const std::uint64_t first = static_cast<std::uint64_t>(s) * n;
    for (int i = 0; i < n; ++i)
        b[i] = static_cast<T>(counter_draw(seed, rhs_stream, first + i, bound));
}



} // namespace tdls_bench



#endif // TDLS_BENCHMARKS_COMMON_BATCH_HPP
