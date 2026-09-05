/// \file
/// \brief Negative compilation test: a gather view holding one pointer
/// per element, hence without data(), must be rejected by the adaptor
/// entry points.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the dense-object diagnostic
/// of the adaptors (the gather mock has the shape of
/// tfel::math::CoalescedView).

#include <tdls/tdls.hpp>

namespace {

constexpr int N = 4;

/// \brief Indexing policy of the gather mock, row-major.
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

/// \brief Gather-like mock: one pointer per element and no data() at
/// all, hence outside the dense contract.
struct MockGatherView {
    using indexing_policy = MockMatrixPolicy;
    double* pointers[N * N];
};

} // namespace

int main() {
    MockGatherView A{};
    int piv[N];
    return tdls::factorize(A, piv) ? 0 : 1;
}
