/// \file
/// \brief Bridge suite: every residency combination is equivalent.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The residency booleans only change where the operands live: slices of
/// the threads (internal) or the whole objects walked with a stride
/// (external). For each of the eight combinations and each entry path
/// (solve, factorize + substitute, solve_inplace, factorize +
/// substitute_inplace), the factored rows, the row positions and the
/// solution must be bitwise identical to the all-external baseline (the
/// column-major cell runs the four internal-matrix combinations). The
/// external operands sit in the middle slot of a three-slot arena, so the
/// strides differ from one. Cells: one thread per system, a full group, a
/// group with a mixed slot, and a column-major group; the small systems
/// run the forced unrolling, the larger ones the no-pragma branch (see
/// tdls_tests::test_unroll).

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "group_runner.hpp"
#include "harness.hpp"

namespace {

/// \brief Runs one residency combination against the all-external
/// baseline on every path of one system.
/// \tparam N               system dimension
/// \tparam Config          solver configuration
/// \tparam internal_rhs    residency of the right-hand side and solution
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of the matrix
/// \param[in] A0 original matrix, contiguous row-major
/// \param[in] b0 right-hand side
/// \return the number of paths compared
template<int N, tdls::CooperativeLUppConfig<double> Config, bool internal_rhs, bool internal_piv,
         bool internal_matrix>
int compare_combination(const double* A0, const double* b0) {
    using Baseline = tdls_tests::GroupRunner<double, N, Config, false, false, false>;
    using Runner =
        tdls_tests::GroupRunner<double, N, Config, internal_rhs, internal_piv, internal_matrix>;
    std::vector<double> A_base(N * N), A_other(N * N);
    int piv_base[N], piv_other[N];
    double x_base[N], x_other[N];
    int compared = 0;
    tdls_tests::for_paths<tdls_tests::Path::combined, tdls_tests::Path::split,
                          tdls_tests::Path::fused, tdls_tests::Path::split_inplace>([&](auto tag) {
        constexpr auto path = decltype(tag)::value;
        bool uniform_base = false, uniform_other = false;
        const bool ok_base =
            Baseline::template run<path>(A0, b0, A_base.data(), piv_base, x_base, uniform_base);
        const bool ok_other =
            Runner::template run<path>(A0, b0, A_other.data(), piv_other, x_other, uniform_other);
        TDLS_CHECK(ok_base && ok_other && uniform_base && uniform_other);
        TDLS_CHECK_BITWISE(A_base.data(), A_other.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(piv_base, piv_other, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x_base, x_other, static_cast<std::size_t>(N));
        ++compared;
    });
    return compared;
}

/// \brief Runs the residency combinations on one reproducible batch: all
/// eight, or only the four with an internal matrix, the ones a
/// column-major configuration changes beyond the layout suite (which
/// covers the external column-major matrix).
/// \tparam N                    system dimension
/// \tparam Config               solver configuration
/// \tparam internal_matrix_only restrict to the internal-matrix
///         combinations
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<int N, tdls::CooperativeLUppConfig<double> Config, bool internal_matrix_only = false>
void residency_case(const int count, const std::uint64_t seed) {
    const auto batch = tdls_tests::make_batch<double>(N, count, seed, 0.5);
    int compared     = 0;
    for (int s = 0; s < count; ++s) {
        const double* A0 = batch.matrix(s);
        const double* b0 = batch.rhs(s);
        if constexpr (!internal_matrix_only) {
            compared += compare_combination<N, Config, false, false, false>(A0, b0);
            compared += compare_combination<N, Config, false, true, false>(A0, b0);
            compared += compare_combination<N, Config, true, false, false>(A0, b0);
            compared += compare_combination<N, Config, true, true, false>(A0, b0);
        }
        compared += compare_combination<N, Config, false, false, true>(A0, b0);
        compared += compare_combination<N, Config, false, true, true>(A0, b0);
        compared += compare_combination<N, Config, true, false, true>(A0, b0);
        compared += compare_combination<N, Config, true, true, true>(A0, b0);
    }
    // Floor: every combination, four paths each, on every system.
    TDLS_CHECK(compared == (internal_matrix_only ? 16 : 32) * count);
}

} // namespace

// One thread per system under the forced unrolling, on a small system:
// the unrolled branch of the sequential path keeps a bridge here.
TDLS_TEST_CASE("cooperativelupp/bridge/residencies/double/N=7,rows_per_thread=7") {
    residency_case<7, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 7}>(40, 640707);
}
TDLS_TEST_CASE("cooperativelupp/bridge/residencies/double/N=12,rows_per_thread=3") {
    residency_case<12, tdls::CooperativeLUppConfig<double>{
                           .rows_per_thread = 3,
                           .unroll_loops    = tdls_tests::test_unroll<12, 3>}>(10, 641203);
}
TDLS_TEST_CASE("cooperativelupp/bridge/residencies/double/N=13,rows_per_thread=5") {
    residency_case<13, tdls::CooperativeLUppConfig<double>{
                           .rows_per_thread = 5,
                           .unroll_loops    = tdls_tests::test_unroll<13, 5>}>(10, 641305);
}
TDLS_TEST_CASE("cooperativelupp/bridge/residencies/double/N=13,rows_per_thread=5,colmajor") {
    residency_case<13,
                   tdls::CooperativeLUppConfig<double>{.rows_per_thread = 5,
                                                       .unroll_loops =
                                                           tdls_tests::test_unroll<13, 5>,
                                                       .layout = tdls::MatrixLayout::ColMajor},
                   true>(10, 641315);
}
// A small group under the forced unrolling, with a mixed slot (N = 7 on 3
// threads of 3 rows), in both layouts.
TDLS_TEST_CASE("cooperativelupp/bridge/residencies/double/N=7,rows_per_thread=3") {
    residency_case<7, tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3}>(20, 640703);
}
TDLS_TEST_CASE("cooperativelupp/bridge/residencies/double/N=7,rows_per_thread=3,colmajor") {
    residency_case<7,
                   tdls::CooperativeLUppConfig<double>{.rows_per_thread = 3,
                                                       .layout = tdls::MatrixLayout::ColMajor},
                   true>(20, 640713);
}

TDLS_TEST_MAIN
