/// \file
/// \brief Bridge suite: the compile-time strides of
/// ArrowheadLUppSolverStatic against the runtime ones.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// Every entry point takes the stride of each operand as a template
/// argument too (tdls::dynamic_stride by default), passed on to the
/// out-of-line parts of the solver: the second pass, and the streamed
/// solve under streamed_out_of_line. On the standard batch (dominant and
/// random blocks, singular and pivot-poor ones, so that the second pass
/// runs), each operand has its own stride, so that a stride given to the
/// wrong operand shows: the matrix (or the border block of the recomputed
/// solve) by 3, the workspace by 4, the right-hand sides by 5, the tangent
/// or canonical columns by 6, the gaps as NaN. Each entry point runs twice:
/// with the runtime strides, then with the strides as template arguments
/// and function arguments of 1, which the solver must ignore. The two give
/// the same buffers, bit for bit, the gaps included, the same verdicts and
/// the same counts of refused blocks: every stored, streamed and
/// recomputed entry point and overload, the tangents, both couplings and
/// layouts, the condensation MinTraffic, the blocks in registers,
/// streamed_out_of_line, and a single stride given at compile time.

#include <cstdint>
#include <limits>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"

namespace {

/// Stride of the matrix, or of the border block of the recomputed solve.
constexpr int SA = 3;
/// Stride of the workspace.
constexpr int SW = 4;
/// Stride of the right-hand sides and solutions.
constexpr int SY = 5;
/// Stride of the tangent and canonical columns.
constexpr int SX = 6;
/// The default of the stride template parameters.
constexpr int dyn = tdls::dynamic_stride;

/// \brief A configuration from its paths.
/// \param[in] diag    diagonal coupling
/// \param[in] col     column-major layout
/// \param[in] reg     blocks eliminated in registers
/// \param[in] traffic condensation MinTraffic
/// \param[in] outline streamed_out_of_line
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> path_config(const bool diag, const bool col,
                                                        const bool reg, const bool traffic = false,
                                                        const bool outline = false) {
    tdls::ArrowheadLUppConfig<double> c;
    c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.storage  = tdls::ArrowheadStorage::Packed;
    c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.condensation =
        traffic ? tdls::ArrowheadCondensation::MinTraffic : tdls::ArrowheadCondensation::MinFlops;
    c.block_row_interchange = reg ? tdls::RowInterchange::Physical : tdls::RowInterchange::Logical;
    c.streamed_out_of_line  = outline;
    return c;
}

/// \brief A provider that fills block k from the full matrix of a problem.
/// \tparam diag diagonal coupling (the accessor of the couplings)
/// \param[in] p problem
/// \return the provider
template<bool diag>
auto make_provider(const tdls_tests::ArrowProblem<double>& p) {
    return [&p](const int k, auto& blk) {
        const tdls::ArrowheadShape& s = p.shape;
        const int B = s.border, S = s.block;
        for (int i = 0; i < S; ++i) {
            const int row = B + k * S + i;
            for (int j = 0; j < S; ++j)
                blk.d(i, j) = p.at(row, B + k * S + j);
            for (int c = 0; c < B; ++c)
                blk.akb(i, c) = p.at(row, c);
            blk.y(i) = p.b[static_cast<std::size_t>(row)];
        }
        if constexpr (diag)
            for (int i = 0; i < S; ++i)
                blk.coupling(i) = p.at(s.coupling_row + i, B + k * S + i);
        else
            for (int r = 0; r < B; ++r)
                for (int i = 0; i < S; ++i)
                    blk.abk(r, i) = p.at(r, B + k * S + i);
    };
}

/// \brief The buffers of one call: matrix storage (or border block),
/// workspace, right-hand sides, tangent or canonical columns; the verdict
/// and the count of refused blocks.
struct Run {
    std::vector<double> A;    ///< storage, or the border block
    std::vector<double> work; ///< workspace
    std::vector<double> y;    ///< right-hand sides and solutions
    std::vector<double> x;    ///< columns
    int outcome = -1;         ///< verdict or outcome
    int refused = -1;         ///< refused blocks
};

/// \brief Bitwise equality of two calls.
/// \param[in] a first call
/// \param[in] b second call
void same(const Run& a, const Run& b) {
    TDLS_CHECK(a.outcome == b.outcome);
    TDLS_CHECK(a.refused == b.refused);
    TDLS_CHECK_BITWISE(a.A.data(), b.A.data(), a.A.size());
    TDLS_CHECK_BITWISE(a.work.data(), b.work.data(), a.work.size());
    TDLS_CHECK_BITWISE(a.y.data(), b.y.data(), a.y.size());
    TDLS_CHECK_BITWISE(a.x.data(), b.x.data(), a.x.size());
}

/// \brief A strided vector of NaN.
/// \param[in] entries entries
/// \param[in] stride  stride
/// \return the vector
std::vector<double> strided(const int entries, const int stride) {
    return std::vector<double>(static_cast<std::size_t>(entries) * static_cast<std::size_t>(stride),
                               std::numeric_limits<double>::quiet_NaN());
}

/// \brief Runs a call on a copy of the buffers.
/// \tparam Fn callable (Run&)
/// \param[in] u  the buffers
/// \param[in] fn the call
/// \return the buffers after the call
template<typename Fn>
Run call(Run u, Fn&& fn) {
    fn(u);
    return u;
}

/// \brief Compares every entry point with its compile-time strides against
/// its runtime strides on a standard batch.
/// \tparam Shape  shape of the matrices
/// \tparam Config configuration
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void strides_case(const int count, const std::uint64_t seed) {
    using S             = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Map           = typename S::storage_map;
    using RO            = typename S::RecomputedOutcome;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    constexpr bool col  = Config.layout == tdls::MatrixLayout::ColMajor;
    constexpr int n     = S::dimension;
    constexpr int B     = Shape.border;
    constexpr int NB    = Shape.blocks;
    constexpr int msz   = S::storage_size;
    constexpr int wsz   = S::template multirhs_workspace_size<B>;
    constexpr int rsz   = S::template recomputed_workspace_size<NB, B>;
    constexpr auto xcs  = static_cast<std::size_t>(B) * SX;
    constexpr auto ycs  = static_cast<std::size_t>(n) * SY;
    int refused = 0, solved = 0;
    for (int s = 0; s < count; ++s) {
        const auto p  = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const auto Ms = tdls_tests::store_arrow<Map>(p);
        auto provider = make_provider<diag>(p);
        // the stored matrix with two right-hand sides, then two more columns
        // for the copies of solve_multirhs; the border block alone
        // (streamed); the border block given apart (recomputed)
        Run full{strided(msz, SA), strided(wsz, SW), strided(4 * n, SY), strided(B * B, SX)};
        for (int e = 0; e < msz; ++e)
            full.A[static_cast<std::size_t>(e) * SA] = Ms[static_cast<std::size_t>(e)];
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < n; ++i)
                full.y[static_cast<std::size_t>(m * n + i) * SY] =
                    p.b[static_cast<std::size_t>(i)] * double(m + 1);
        Run border{strided(msz, SA), strided(wsz, SW), strided(n, SY), strided(B * B, SX)};
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c) {
                const auto e     = static_cast<std::size_t>(Map::abb(Shape, r, c));
                border.A[e * SA] = Ms[e];
            }
            border.y[static_cast<std::size_t>(r) * SY] = p.b[static_cast<std::size_t>(r)];
        }
        Run apart{strided(B * B, SA), strided(rsz, SW), strided(n, SY), strided(B * B, SX)};
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c)
                apart.A[static_cast<std::size_t>(col ? c * B + r : r * B + c) * SA] = p.at(r, c);
            apart.y[static_cast<std::size_t>(r) * SY] = p.b[static_cast<std::size_t>(r)];
        }

        // stored
        const Run r1 = call(full, [](Run& u) {
            u.outcome = S::template solve_inplace<false, false, false>(
                u.A.data(), SA, u.work.data(), SW, u.y.data(), SY, u.refused);
        });
        same(r1, call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace<false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, u.refused);
             }));
        // a single stride at compile time, the others at run time
        same(r1, call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace<false, false, false, SA>(
                     u.A.data(), 1, u.work.data(), SW, u.y.data(), SY, u.refused);
             }));
        same(r1, call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace<false, false, false, dyn, SW>(
                     u.A.data(), SA, u.work.data(), 1, u.y.data(), SY, u.refused);
             }));
        same(r1, call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace<false, false, false, dyn, dyn, SY>(
                     u.A.data(), SA, u.work.data(), SW, u.y.data(), 1, u.refused);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve_inplace<false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), SY);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace<false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), 1);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve<false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), u.y.data() + n * SY, SY,
                          u.refused);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve<false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), u.y.data() + n * SY, 1,
                     u.refused);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve<false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), u.y.data() + n * SY, SY);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve<false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), u.y.data() + n * SY, 1);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve_inplace_multirhs<2, false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), SY, ycs, u.refused);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace_multirhs<2, false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, ycs, u.refused);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve_inplace_multirhs<2, false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), SY, ycs);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve_inplace_multirhs<2, false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, ycs);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve_multirhs<2, false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), u.y.data() + 2 * n * SY,
                          SY, ycs, u.refused);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve_multirhs<2, false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), u.y.data() + 2 * n * SY, 1, ycs,
                     u.refused);
             }));
        same(call(full,
                  [](Run& u) {
                      u.outcome = S::template solve_multirhs<2, false, false, false>(
                          u.A.data(), SA, u.work.data(), SW, u.y.data(), u.y.data() + 2 * n * SY,
                          SY, ycs);
                  }),
             call(full, [](Run& u) {
                 u.outcome = S::template solve_multirhs<2, false, false, false, SA, SW, SY>(
                     u.A.data(), 1, u.work.data(), 1, u.y.data(), u.y.data() + 2 * n * SY, 1, ycs);
             }));
        same(
            call(full,
                 [](Run& u) {
                     u.outcome =
                         S::template solve_border_canonical_multirhs<B, false, false, false>(
                             u.A.data(), SA, u.work.data(), SW, 0, u.x.data(), SX, xcs, u.refused);
                 }),
            call(full, [](Run& u) {
                u.outcome =
                    S::template solve_border_canonical_multirhs<B, false, false, false, SA, SW, SX>(
                        u.A.data(), 1, u.work.data(), 1, 0, u.x.data(), 1, xcs, u.refused);
            }));
        same(
            call(full,
                 [](Run& u) {
                     u.outcome =
                         S::template solve_border_canonical_multirhs<B, false, false, false>(
                             u.A.data(), SA, u.work.data(), SW, 0, u.x.data(), SX, xcs);
                 }),
            call(full, [](Run& u) {
                u.outcome =
                    S::template solve_border_canonical_multirhs<B, false, false, false, SA, SW, SX>(
                        u.A.data(), 1, u.work.data(), 1, 0, u.x.data(), 1, xcs);
            }));

        // streamed
        same(call(border,
                  [&provider](Run& u) {
                      u.outcome = S::template solve_inplace_streamed<false, false, false>(
                          provider, u.A.data(), SA, u.work.data(), SW, u.y.data(), SY, u.refused);
                  }),
             call(border, [&provider](Run& u) {
                 u.outcome = S::template solve_inplace_streamed<false, false, false, SA, SW, SY>(
                     provider, u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, u.refused);
             }));
        same(
            call(border,
                 [&provider](Run& u) {
                     u.outcome = S::template solve_border_canonical_multirhs_streamed<B, false,
                                                                                      false, false>(
                         provider, u.A.data(), SA, u.work.data(), SW, 0, u.x.data(), SX, xcs,
                         u.refused);
                 }),
            call(border, [&provider](Run& u) {
                u.outcome = S::template solve_border_canonical_multirhs_streamed<B, false, false,
                                                                                 false, SA, SW, SX>(
                    provider, u.A.data(), 1, u.work.data(), 1, 0, u.x.data(), 1, xcs, u.refused);
            }));
        for (const bool converged : {true, false}) {
            const auto answer = [converged] { return converged; };
            same(call(border,
                      [&provider, &answer](Run& u) {
                          u.outcome = int(
                              S::template solve_inplace_streamed_or_tangent<B, false, false, false>(
                                  provider, answer, u.A.data(), SA, u.work.data(), SW, u.y.data(),
                                  SY, 0, u.x.data(), SX, xcs, u.refused));
                      }),
                 call(border, [&provider, &answer](Run& u) {
                     u.outcome =
                         int(S::template solve_inplace_streamed_or_tangent<B, false, false, false,
                                                                           SA, SW, SY, SX>(
                             provider, answer, u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, 0,
                             u.x.data(), 1, xcs, u.refused));
                 }));
        }

        // recomputed, every block storable
        const Run q1 = call(apart, [&provider](Run& u) {
            u.outcome = int(S::template solve_inplace_recomputed<NB, false, false, false>(
                provider, u.A.data(), SA, u.work.data(), SW, u.y.data(), SY, u.refused));
        });
        same(q1, call(apart, [&provider](Run& u) {
                 u.outcome =
                     int(S::template solve_inplace_recomputed<NB, false, false, false, SA, SW, SY>(
                         provider, u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, u.refused));
             }));
        for (const bool converged : {true, false}) {
            const auto answer = [converged] { return converged; };
            same(call(apart,
                      [&provider, &answer](Run& u) {
                          u.outcome =
                              int(S::template solve_inplace_recomputed_or_tangent<B, NB, false,
                                                                                  false, false>(
                                  provider, answer, u.A.data(), SA, u.work.data(), SW, u.y.data(),
                                  SY, 0, u.x.data(), SX, xcs, u.refused));
                      }),
                 call(apart, [&provider, &answer](Run& u) {
                     u.outcome =
                         int(S::template solve_inplace_recomputed_or_tangent<B, NB, false, false,
                                                                             false, SA, SW, SY, SX>(
                             provider, answer, u.A.data(), 1, u.work.data(), 1, u.y.data(), 1, 0,
                             u.x.data(), 1, xcs, u.refused));
                 }));
        }
        TDLS_CHECK(q1.outcome == int(RO::solved));
        solved += r1.outcome;
        refused += r1.refused;
    }
    // Floors: the batch is invertible, and its injected blocks are refused,
    // so that the second pass runs.
    TDLS_CHECK(solved == count);
    TDLS_CHECK(refused >= 2 * (count / 3));
}

using tdls::ArrowheadShape;
constexpr ArrowheadShape polycrystal{12, 6, 10, 6};
constexpr ArrowheadShape uneven_dense{8, 6, 10, 0};
constexpr ArrowheadShape iwan{6, 1, 24, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/static_strides/dense") {
    strides_case<uneven_dense, path_config(false, false, false)>(9, 800100);
    strides_case<uneven_dense, path_config(false, true, false)>(9, 800200);
    strides_case<uneven_dense, path_config(false, false, false, true)>(9, 800300);
    strides_case<iwan, path_config(false, false, false)>(9, 800400);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_strides/diagonal") {
    strides_case<polycrystal, path_config(true, false, false)>(9, 800500);
    strides_case<polycrystal, path_config(true, true, false)>(9, 800600);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_strides/registers") {
    strides_case<uneven_dense, path_config(false, false, true)>(9, 800700);
    strides_case<polycrystal, path_config(true, true, true)>(9, 800800);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/static_strides/streamed_out_of_line") {
    strides_case<uneven_dense, path_config(false, false, false, false, true)>(9, 800900);
    strides_case<polycrystal, path_config(true, false, true, false, true)>(9, 801000);
}

TDLS_TEST_MAIN
