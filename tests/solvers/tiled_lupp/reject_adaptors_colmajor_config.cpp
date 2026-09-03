/// \file
/// \brief Negative compilation test: a configuration selecting the
/// column-major layout must be rejected by the adaptor entry points.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the row-major diagnostic of
/// the adaptors. The recognized indexing policies are row-major: a
/// column-major configuration would remap the flat index under a
/// row-major object and silently corrupt the results.

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

constexpr auto config = tdls::TiledLUppConfig<double>{.layout = tdls::MatrixLayout::ColMajor};

} // namespace

int main() {
    MockMatrix A{};
    int piv[N];
    return tdls::factorize<config>(A, piv) ? 0 : 1;
}
