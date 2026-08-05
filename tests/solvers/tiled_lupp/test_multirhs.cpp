/// \file
/// \brief Bridge suite of the multi right-hand-side blocks.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. Released under the
/// BSD 3-Clause License (see the LICENSE file).
///
/// The _block entry points document one property: per-column results
/// match the same columns solved one by one, bitwise. The suite checks
/// it on identical inputs for both solvers - substitute_block against
/// NRHS separate substitute calls (the baseline, itself anchored by the
/// oracle suites), substitute_inplace_block against the two-buffer
/// block, the solve_block / solve_inplace_block wrappers against their
/// split forms, the diagnostics-free overloads against the counting
/// ones, and the runtime solver's blocks against the compile-time
/// solver's on the same systems. Internal and external residencies are
/// both exercised on the compile-time solver.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Runs every block equivalence on one reproducible batch.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Schedule elimination schedule
/// \tparam NRHS  number of right-hand-side columns per block
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule, int NRHS>
void multirhs_case(const int count, const double bound, const std::uint64_t seed) {
    using Config  = tdls::TiledLUppConfig<T, TS, Schedule>;
    using Static  = tdls::TiledLUppSolverStatic<T, N, Config>;
    using Dynamic = tdls::TiledLUppSolverDynamic<T, Config>;

    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::UniformGenerator extra(seed + 1, bound);

    std::vector<T> A_split(N * N), A_other(N * N);
    int solved = 0;
    for (int s = 0; s < count; ++s) {
        const T* A0 = batch.matrix(s);

        // NRHS right-hand-side columns: the batch one plus generated ones,
        // stored column after column (the internal block convention).
        T B[NRHS * N];
        for (int i = 0; i < N; ++i)
            B[i] = batch.rhs(s)[i];
        for (int e = N; e < NRHS * N; ++e)
            B[e] = static_cast<T>(extra.next());

        // Baseline: factorize once, NRHS separate single-column
        // substitutes.
        std::copy(A0, A0 + N * N, A_split.begin());
        int piv_split[N], oot_split = 0;
        const bool ok =
            Static::template factorize<true, false>(A_split.data(), 1, piv_split, 1, oot_split);
        if (!ok) continue;
        ++solved;
        T X_ref[NRHS * N];
        for (int w = 0; w < NRHS; ++w)
            Static::template substitute<true, true, false>(A_split.data(), 1, piv_split, 1,
                                                           B + w * N, X_ref + w * N, 1);

        // substitute_block, internal residency.
        {
            T X[NRHS * N];
            Static::template substitute_block<NRHS, true, true, false>(A_split.data(), 1, piv_split,
                                                                       1, B, X, 1, 0);
            TDLS_CHECK_BITWISE(X_ref, X, static_cast<std::size_t>(NRHS) * N);
        }

        // substitute_block, external residency, interleaved columns
        // (element stride NRHS, column stride 1).
        {
            std::vector<T> Bx(static_cast<std::size_t>(NRHS) * N), Xx(Bx.size());
            for (int w = 0; w < NRHS; ++w)
                for (int i = 0; i < N; ++i)
                    Bx[static_cast<std::size_t>(i) * NRHS + w] = B[w * N + i];
            Static::template substitute_block<NRHS, false, true, false>(
                A_split.data(), 1, piv_split, 1, Bx.data(), Xx.data(), NRHS, 1);
            for (int w = 0; w < NRHS; ++w)
                for (int i = 0; i < N; ++i)
                    TDLS_CHECK_BITWISE(&X_ref[w * N + i],
                                       &Xx[static_cast<std::size_t>(i) * NRHS + w], 1);
        }

        // substitute_inplace_block must reproduce the two-buffer block.
        {
            T Y[NRHS * N];
            for (int e = 0; e < NRHS * N; ++e)
                Y[e] = B[e];
            Static::template substitute_inplace_block<NRHS, true, true, false>(
                A_split.data(), 1, piv_split, 1, Y, 1, 0);
            TDLS_CHECK_BITWISE(X_ref, Y, static_cast<std::size_t>(NRHS) * N);
        }

        // solve_block and solve_inplace_block must reproduce factorize +
        // their substitution, counting and diagnostics-free overloads
        // alike.
        {
            std::copy(A0, A0 + N * N, A_other.begin());
            int piv[N], oot = 0;
            T X[NRHS * N];
            const bool ok_solve = Static::template solve_block<NRHS, true, true, false>(
                A_other.data(), 1, piv, 1, B, X, 1, 0, oot);
            TDLS_CHECK(ok_solve);
            TDLS_CHECK_BITWISE(A_split.data(), A_other.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_split, piv, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(X_ref, X, static_cast<std::size_t>(NRHS) * N);
            TDLS_CHECK(oot_split == oot);

            std::copy(A0, A0 + N * N, A_other.begin());
            T Y[NRHS * N];
            for (int e = 0; e < NRHS * N; ++e)
                Y[e] = B[e];
            const bool ok_inpl = Static::template solve_inplace_block<NRHS, true, true, false>(
                A_other.data(), 1, piv, 1, Y, 1, 0);
            TDLS_CHECK(ok_inpl);
            TDLS_CHECK_BITWISE(X_ref, Y, static_cast<std::size_t>(NRHS) * N);
        }

        // The runtime solver's blocks must match the compile-time
        // solver's on the same systems (contiguous storage, column
        // stride N).
        {
            std::copy(A0, A0 + N * N, A_other.begin());
            int piv[N], oot = 0;
            T X[NRHS * N];
            const bool ok_dyn =
                Dynamic::solve_block(N, NRHS, A_other.data(), 1, piv, 1, B, X, 1, N, oot);
            TDLS_CHECK(ok_dyn);
            TDLS_CHECK_BITWISE(A_split.data(), A_other.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_split, piv, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(X_ref, X, static_cast<std::size_t>(NRHS) * N);
            TDLS_CHECK(oot_split == oot);

            T Y[NRHS * N];
            for (int e = 0; e < NRHS * N; ++e)
                Y[e] = B[e];
            Dynamic::substitute_inplace_block(N, NRHS, A_other.data(), 1, piv, 1, Y, 1, N);
            TDLS_CHECK_BITWISE(X_ref, Y, static_cast<std::size_t>(NRHS) * N);

            std::copy(A0, A0 + N * N, A_other.begin());
            for (int e = 0; e < NRHS * N; ++e)
                Y[e] = B[e];
            const bool ok_free =
                Dynamic::solve_inplace_block(N, NRHS, A_other.data(), 1, piv, 1, Y, 1, N);
            TDLS_CHECK(ok_free);
            TDLS_CHECK_BITWISE(X_ref, Y, static_cast<std::size_t>(NRHS) * N);
        }
    }
    // Floor: the generator produces well-conditioned systems, so every
    // one must have been solved (a suite green on zero solved systems
    // would assert nothing).
    TDLS_CHECK(solved == count);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/bridge/multirhs/double/N=12,TS=3,RL,NRHS=4,default") {
    multirhs_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking, 4>(100, 0.5, 220100);
}
TDLS_TEST_CASE("tiledlupp/bridge/multirhs/double/N=12,TS=3,RL,NRHS=4,stress") {
    multirhs_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking, 4>(100, 5e-10, 220200);
}
TDLS_TEST_CASE("tiledlupp/bridge/multirhs/double/N=13,TS=6,LL,NRHS=5,default") {
    multirhs_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking, 5>(100, 0.5, 220300);
}
TDLS_TEST_CASE("tiledlupp/bridge/multirhs/double/N=12,TS=3,RL,NRHS=1,collapse") {
    multirhs_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking, 1>(100, 0.5, 220400);
}
TDLS_TEST_CASE("tiledlupp/bridge/multirhs/double/N=5,TS=8,RL,NRHS=7,single-tile") {
    multirhs_case<double, 5, 8, tdls::TiledLUppSchedule::RightLooking, 7>(100, 0.5, 220500);
}
TDLS_TEST_CASE("tiledlupp/bridge/multirhs/float/N=12,TS=3,RL,NRHS=3,default") {
    multirhs_case<float, 12, 3, tdls::TiledLUppSchedule::RightLooking, 3>(100, 0.5, 220600);
}

TDLS_TEST_MAIN
