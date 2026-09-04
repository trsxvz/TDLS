/// \file
/// \brief Negative compilation test: with a runtime-sized matrix, a
/// fixed-size right-hand-side block and solution block of different row
/// counts must be rejected by the adaptor entry points.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the block-extent diagnostic
/// of the adaptors: the two row counts are provably inconsistent
/// whatever the runtime dimension of the matrix.

#include <cstddef>
#include <vector>

#include <tdls/tdls.hpp>

namespace {

/// \brief Indexing policy of the runtime matrix mock: extents as data
/// members, zero by default: the shape of tfel::math::matrix.
struct MockRuntimeMatrixPolicy {
    using size_type            = std::size_t;
    static constexpr int arity = 2;
    std::size_t rows           = 0;
    std::size_t cols           = 0;
    constexpr std::size_t size(const int d) const {
        return d == 0 ? rows : cols;
    }
    constexpr std::size_t getIndex(const std::size_t i, const std::size_t j) const {
        return i * cols + j;
    }
};

/// \brief Runtime-sized matrix mock: heap storage, policy instance.
struct MockRuntimeMatrix {
    using indexing_policy = MockRuntimeMatrixPolicy;
    std::vector<double> v;
    MockRuntimeMatrixPolicy policy;
    explicit MockRuntimeMatrix(const std::size_t n) : v(n * n), policy{n, n} {
    }
    double* data() {
        return v.data();
    }
    const double* data() const {
        return v.data();
    }
    MockRuntimeMatrixPolicy getIndexingPolicy() const {
        return policy;
    }
};

/// \brief Indexing policy of the fixed-size block mock, row-major
/// contiguous.
/// \tparam R row count
/// \tparam C column count
template<int R, int C>
struct MockBlockPolicy {
    using size_type            = int;
    static constexpr int arity = 2;
    constexpr int size(const int d) const {
        return d == 0 ? R : C;
    }
    constexpr int getIndex(const int i, const int j) const {
        return i * C + j;
    }
};

/// \brief Fixed-size contiguous block mock.
/// \tparam R row count
/// \tparam C column count
template<int R, int C>
struct MockBlock {
    using indexing_policy = MockBlockPolicy<R, C>;
    double v[R * C];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

} // namespace

int main() {
    MockRuntimeMatrix A(3);
    MockBlock<3, 2> B{};
    MockBlock<4, 2> X{};
    int piv[4];
    return tdls::solve(A, piv, B, X) ? 0 : 1;
}
