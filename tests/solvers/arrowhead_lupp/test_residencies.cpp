/// \file
/// \brief Bridge suite: the eight residency combinations of ArrowheadLUpp
/// are equivalent.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// The internal_rhs / internal_work / internal_matrix template booleans
/// select where each operand lives (caller-local array or strided remote
/// storage) but never change the arithmetic: on identical inputs, all
/// eight combinations produce bitwise identical solutions, matrices left
/// in place and counts of refused blocks. The baseline is the fully
/// external combination; the external operands sit in arenas whose
/// strides (3 for the matrix, 2 for the right-hand side, 5 for the
/// workspace) differ from one, and the entries between them must stay
/// untouched. The runtime solver, always external, joins the comparison
/// with the same strides. The standard batch runs the second pass, the
/// one user of the workspace.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "harness.hpp"

namespace {

constexpr int matrix_stride = 3;
constexpr int rhs_stride    = 2;
constexpr int work_stride   = 5;
/// Value of the entries of the arenas between the strided operands: a
/// write there breaks the comparison.
constexpr double guard = -7.25;

/// \brief Solution and matrix of one residency combination, gathered to
/// contiguous arrays, with its verdict and refused count.
struct Outcome {
    bool ok     = false;
    int refused = 0;
    std::vector<double> x, M;
    bool guards = true;
};

/// \brief Solves a problem under one residency combination.
/// \tparam Solver          static solver
/// \tparam internal_rhs    residency of y
/// \tparam internal_work   residency of the workspace
/// \tparam internal_matrix residency of A
/// \param[in] p problem
/// \return the outcome
template<typename Solver, bool internal_rhs, bool internal_work, bool internal_matrix>
Outcome run(const tdls_tests::ArrowProblem<double>& p) {
    const auto M0 = tdls_tests::store_arrow<typename Solver::storage_map>(p);
    const int ms = internal_matrix ? 1 : matrix_stride, ys = internal_rhs ? 1 : rhs_stride,
              ws = internal_work ? 1 : work_stride;
    std::vector<double> M(M0.size() * static_cast<std::size_t>(ms), guard),
        y(p.b.size() * static_cast<std::size_t>(ys), guard),
        w(static_cast<std::size_t>(Solver::workspace_size) * static_cast<std::size_t>(ws), guard);
    for (std::size_t e = 0; e < M0.size(); ++e)
        M[e * static_cast<std::size_t>(ms)] = M0[e];
    for (std::size_t i = 0; i < p.b.size(); ++i)
        y[i * static_cast<std::size_t>(ys)] = p.b[i];
    Outcome o;
    o.ok = Solver::template solve_inplace<internal_rhs, internal_work, internal_matrix>(
        M.data(), ms, w.data(), ws, y.data(), ys, o.refused);
    o.x.resize(p.b.size());
    o.M.resize(M0.size());
    for (std::size_t i = 0; i < p.b.size(); ++i) {
        o.x[i] = y[i * static_cast<std::size_t>(ys)];
        for (int t = 1; t < ys; ++t)
            o.guards = o.guards && y[i * static_cast<std::size_t>(ys) + t] == guard;
    }
    for (std::size_t e = 0; e < M0.size(); ++e) {
        o.M[e] = M[e * static_cast<std::size_t>(ms)];
        for (int t = 1; t < ms; ++t)
            o.guards = o.guards && M[e * static_cast<std::size_t>(ms) + t] == guard;
    }
    for (std::size_t e = 0; e < w.size(); ++e)
        if (e % static_cast<std::size_t>(ws) != 0) o.guards = o.guards && w[e] == guard;
    return o;
}

/// \brief Solves a problem with the runtime solver, external operands at
/// the strides of the baseline.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in] p problem
/// \return the outcome
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
Outcome run_dynamic(const tdls_tests::ArrowProblem<double>& p) {
    using Solver  = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    const auto M0 = tdls_tests::store_arrow<typename Solver::storage_map>(p);
    std::vector<double> M(M0.size() * matrix_stride, guard), y(p.b.size() * rhs_stride, guard),
        w(static_cast<std::size_t>(Solver::workspace_size(Shape)) * work_stride, guard);
    std::vector<int> piv(static_cast<std::size_t>(Solver::pivot_size(Shape)) * 4, -1);
    for (std::size_t e = 0; e < M0.size(); ++e)
        M[e * matrix_stride] = M0[e];
    for (std::size_t i = 0; i < p.b.size(); ++i)
        y[i * rhs_stride] = p.b[i];
    Outcome o;
    o.ok = Solver::solve_inplace(Shape, M.data(), matrix_stride, piv.data(), 4, w.data(),
                                 work_stride, y.data(), rhs_stride, o.refused);
    o.x.resize(p.b.size());
    o.M.resize(M0.size());
    for (std::size_t i = 0; i < p.b.size(); ++i)
        o.x[i] = y[i * rhs_stride];
    for (std::size_t e = 0; e < M0.size(); ++e)
        o.M[e] = M[e * matrix_stride];
    return o;
}

/// \brief Compares an outcome to the baseline.
/// \param[in] a baseline
/// \param[in] b outcome
void same(const Outcome& a, const Outcome& b) {
    TDLS_CHECK(a.ok == b.ok);
    TDLS_CHECK(a.refused == b.refused);
    TDLS_CHECK(b.guards);
    if (a.ok && b.ok) {
        TDLS_CHECK(tdls_tests::same_bits(a.x, b.x));
        TDLS_CHECK(tdls_tests::same_bits(a.M, b.M));
    }
}

/// \brief The eight combinations and the runtime solver on a standard
/// batch.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in] count number of systems
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void residency_case(const int count, const std::uint64_t seed) {
    using Solver        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    int refused_total   = 0;
    for (int s = 0; s < count; ++s) {
        const auto p    = tdls_tests::batch_problem<double>(Shape, diag, s, seed);
        const auto base = run<Solver, false, false, false>(p);
        TDLS_CHECK(base.ok);
        TDLS_CHECK(base.guards);
        refused_total += base.refused;
        same(base, run<Solver, true, false, false>(p));
        same(base, run<Solver, false, true, false>(p));
        same(base, run<Solver, false, false, true>(p));
        same(base, run<Solver, true, true, false>(p));
        same(base, run<Solver, true, false, true>(p));
        same(base, run<Solver, false, true, true>(p));
        same(base, run<Solver, true, true, true>(p));
        same(base, run_dynamic<Shape, Config>(p));
    }
    TDLS_CHECK(refused_total >= 2 * (count / 3));
}

/// \brief A configuration from its coupling, layout and storage.
/// \param[in] diag    diagonal coupling
/// \param[in] col     column-major layout
/// \param[in] storage storage
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> residency_config(const bool diag, const bool col,
                                                             const tdls::ArrowheadStorage storage) {
    tdls::ArrowheadLUppConfig<double> c;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.storage  = storage;
    return c;
}

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/residencies/diagonal") {
    residency_case<tdls::ArrowheadShape{12, 6, 9, 6},
                   residency_config(true, false, tdls::ArrowheadStorage::Packed)>(9, 700100);
    residency_case<tdls::ArrowheadShape{12, 6, 9, 6},
                   residency_config(true, true, tdls::ArrowheadStorage::Bsr)>(9, 700200);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/residencies/dense") {
    residency_case<tdls::ArrowheadShape{8, 6, 10, 0},
                   residency_config(false, false, tdls::ArrowheadStorage::Csr)>(9, 700300);
    residency_case<tdls::ArrowheadShape{8, 6, 10, 0},
                   residency_config(false, true, tdls::ArrowheadStorage::Packed)>(9, 700400);
    residency_case<tdls::ArrowheadShape{6, 1, 24, 0},
                   residency_config(false, false, tdls::ArrowheadStorage::Full)>(9, 700500);
}

TDLS_TEST_MAIN
