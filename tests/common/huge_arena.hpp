#ifndef TDLS_TESTS_COMMON_HUGE_ARENA_HPP
#define TDLS_TESTS_COMMON_HUGE_ARENA_HPP



/// \file
/// \brief Virtual arena far beyond the memory of the machine, for the
/// offsets past 2^32 elements.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// An anonymous private mapping reserved without backing store
/// (MAP_NORESERVE): only the pages actually touched take memory, so a
/// system spread over tens of gigabytes by huge strides costs a few pages.
/// POSIX systems only; elsewhere, or when the kernel refuses the
/// reservation (strict overcommit), the arena is empty and the suites
/// report the cell as not run.

#include <cstddef>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

namespace tdls_tests {

/// \brief count elements of T in one lazily backed mapping.
/// \tparam T element type
template<typename T>
class HugeArena {
  public:
    /// \brief Reserves count elements, zero-initialized pages.
    /// \param[in] count number of elements
    explicit HugeArena(const std::size_t count) : bytes_(count * sizeof(T)) {
#if (defined(__unix__) || defined(__APPLE__)) && defined(MAP_NORESERVE)
        void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        data_   = p == MAP_FAILED ? nullptr : static_cast<T*>(p);
#endif
    }
    HugeArena(const HugeArena&)            = delete;
    HugeArena& operator=(const HugeArena&) = delete;
    ~HugeArena() {
#if (defined(__unix__) || defined(__APPLE__)) && defined(MAP_NORESERVE)
        if (data_ != nullptr) munmap(data_, bytes_);
#endif
    }
    /// \brief First element, nullptr when the reservation failed.
    [[nodiscard]] T* data() const noexcept {
        return data_;
    }

  private:
    std::size_t bytes_ = 0;
    T* data_           = nullptr;
};

/// \brief v, read back through a volatile: a value the optimizer cannot
/// see through.
///
/// The wide strides reach the solvers as runtime values, as a batch
/// stride does. Known at compile time, a stride whose step exceeds 2^32
/// bytes lets the SLP vectorizer of clang 17, 18 and 20 (-O2 and above)
/// store or load two elements 2^32 + 8 bytes apart as one contiguous
/// pair: the second element lands 4 GiB short. The fault reproduces
/// without TDLS; clang 12 and GCC compile it right.
/// \param[in] v value
/// \return v
inline int opaque(const int v) {
    volatile int x = v;
    return x;
}

} // namespace tdls_tests



#endif // TDLS_TESTS_COMMON_HUGE_ARENA_HPP
