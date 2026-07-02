#ifndef TDLS_TESTS_COMMON_GENERATORS_HPP
#define TDLS_TESTS_COMMON_GENERATORS_HPP



/// \file
/// \brief Reproducible input generators and layout helpers of the test
/// suites.
/// \author Tristan Chenaille
///
/// The generator maps raw mt19937_64 output to [-bound, bound] with a
/// fixed 53-bit multiply. mt19937_64 is specified bit for bit by the C++
/// standard while std::uniform_real_distribution is not, so the streams
/// (and every bitwise expectation built on them) are identical on every
/// standard library implementation.



#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>



namespace tdls_tests {



/// \brief Deterministic uniform generator over [-bound, bound].
class UniformGenerator {
  public:
    /// \param[in] seed  engine seed (document one per test case)
    /// \param[in] bound half-width of the sampling interval
    UniformGenerator(const std::uint64_t seed, const double bound) : engine(seed), bound(bound) {
    }

    /// \return the next value of the stream
    double next() {
        const double unit = static_cast<double>(engine() >> 11) * 0x1.0p-53; // [0, 1)
        return (2.0 * unit - 1.0) * bound;
    }

  private:
    std::mt19937_64 engine;
    double bound;
};

/// \brief A reproducible batch of dense systems in contiguous per-system
/// storage: system s occupies A.data() + s*n*n (row-major) and
/// b.data() + s*n.
/// \tparam T scalar type
template<typename T>
struct SystemBatch {
    int n     = 0;    ///< system dimension
    int count = 0;    ///< number of systems
    std::vector<T> A; ///< matrices, count * n * n elements
    std::vector<T> b; ///< right-hand sides, count * n elements

    /// \return the matrix of system s
    T* matrix(const int s) {
        return A.data() + static_cast<std::size_t>(s) * n * n;
    }
    /// \return the matrix of system s
    const T* matrix(const int s) const {
        return A.data() + static_cast<std::size_t>(s) * n * n;
    }
    /// \return the right-hand side of system s
    T* rhs(const int s) {
        return b.data() + static_cast<std::size_t>(s) * n;
    }
    /// \return the right-hand side of system s
    const T* rhs(const int s) const {
        return b.data() + static_cast<std::size_t>(s) * n;
    }
};

/// \brief Builds a reproducible batch.
/// \tparam T scalar type
/// \param[in] n     system dimension
/// \param[in] count number of systems
/// \param[in] seed  generator seed
/// \param[in] bound half-width of the entry distribution (0.5 = default
///            regime, 5e-10 = the stress regime that exercises the
///            out-of-tile pivot search)
/// \return the batch
template<typename T>
inline SystemBatch<T> make_batch(const int n, const int count, const std::uint64_t seed,
                                 const double bound) {
    SystemBatch<T> batch;
    batch.n     = n;
    batch.count = count;
    batch.A.resize(static_cast<std::size_t>(count) * n * n);
    batch.b.resize(static_cast<std::size_t>(count) * n);
    UniformGenerator gen(seed, bound);
    for (auto& v : batch.A)
        v = static_cast<T>(gen.next());
    for (auto& v : batch.b)
        v = static_cast<T>(gen.next());
    return batch;
}

/// \brief Zeroes one column of one system: a structural singularity that
/// every solver must report.
/// \tparam T scalar type
/// \param[in,out] batch  the batch
/// \param[in]     system system index
/// \param[in]     column column to zero
template<typename T>
inline void zero_column(SystemBatch<T>& batch, const int system, const int column) {
    for (int r = 0; r < batch.n; ++r)
        batch.matrix(system)[r * batch.n + column] = T(0);
}

/// \brief Element index of (element, system) in an SoA batch buffer:
/// element-major, stride = batch count.
/// \param[in] element flat element index inside the object
/// \param[in] system  system index
/// \param[in] count   number of systems in the buffer
/// \return the buffer index
constexpr std::size_t soa_index(const std::size_t element, const std::size_t system,
                                const std::size_t count) {
    return element * count + system;
}

/// \brief Element index of (element, system) in an AoS batch buffer:
/// system-major, objects contiguous.
/// \param[in] element     flat element index inside the object
/// \param[in] system      system index
/// \param[in] object_size number of elements of one object
/// \return the buffer index
constexpr std::size_t aos_index(const std::size_t element, const std::size_t system,
                                const std::size_t object_size) {
    return system * object_size + element;
}

/// \brief Element index of (element, system) in an AoSoA batch buffer of
/// width W (block-interleaved; the buffer must be padded so that full
/// W-wide blocks exist).
/// \param[in] element     flat element index inside the object
/// \param[in] system      system index
/// \param[in] object_size number of elements of one object
/// \param[in] width       number of interleaved systems per block
/// \return the buffer index
constexpr std::size_t aosoa_index(const std::size_t element, const std::size_t system,
                                  const std::size_t object_size, const std::size_t width) {
    return (system / width) * object_size * width + element * width + system % width;
}



} // namespace tdls_tests



#endif // TDLS_TESTS_COMMON_GENERATORS_HPP
