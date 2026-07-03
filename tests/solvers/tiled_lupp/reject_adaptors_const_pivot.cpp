/// \file
/// \brief Negative compilation test: a pointer-to-const pivot passed to
/// the factorizing adaptor entry points must be rejected.
/// \author Tristan Chenaille
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the pivot-mutability
/// diagnostic of the adaptors (factorize writes the permutation).

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

} // namespace

int main() {
    MockMatrix A{};
    int storage[N]       = {};
    const int* const piv = storage;
    return tdls::factorize(A, piv) ? 0 : 1;
}
