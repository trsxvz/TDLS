/// \file
/// \brief Negative compilation test: a fixed-size pivot object whose
/// extent differs from the dimension of a fixed-size matrix must be
/// rejected by the adaptor entry points.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the pivot-extent diagnostic
/// of the adaptors: the factorization writes N permutation entries, so
/// a shorter pivot would overflow.

#include <tdls/tdls.hpp>

namespace {

constexpr int N = 4;

/// \brief Indexing policy of the matrix mock, row-major contiguous.
struct MockMatrixPolicy {
    using size_type            = int;
    static constexpr int arity = 2;
    constexpr int size(const int d) const {
        return d >= 0 ? N : N;
    }
    constexpr int getIndex(const int i, const int j) const {
        return i * N + j;
    }
};

/// \brief Contiguous matrix mock: data() returns a plain pointer.
struct MockMatrix {
    using indexing_policy = MockMatrixPolicy;
    double v[N * N];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

/// \brief Indexing policy of the fixed-size int vector mock.
/// \tparam M extent
template<int M>
struct MockVectorPolicy {
    using size_type            = int;
    static constexpr int arity = 1;
    constexpr int size(const int) const {
        return M;
    }
    constexpr int getIndex(const int i) const {
        return i;
    }
};

/// \brief Fixed-size contiguous int vector mock, used as a dense pivot.
/// \tparam M extent
template<int M>
struct MockIntVector {
    using indexing_policy = MockVectorPolicy<M>;
    int v[M];
    int* data() {
        return v;
    }
    const int* data() const {
        return v;
    }
};

} // namespace

int main() {
    MockMatrix A{};
    MockIntVector<N - 2> piv{};
    return tdls::factorize(A, piv) ? 0 : 1;
}
