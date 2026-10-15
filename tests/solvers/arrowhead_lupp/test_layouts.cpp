/// \file
/// \brief Bridge suite: the AoS, SoA and AoSoA batch layouts of
/// ArrowheadLUpp are equivalent.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. All rights reserved.
/// This project is publicly released under the BSD 3-Clause License
/// (see the LICENSE file). CEA may also distribute it under specific
/// licensing conditions.
///
/// A batch of systems is laid out in one buffer per operand: AoS (each
/// system contiguous, stride 1), SoA (element-major, stride = number of
/// systems) or AoSoA (blocks of W interleaved systems, stride W, the
/// batch padded to full blocks). The caller pre-offsets each pointer to
/// its system and passes the stride: the solver sees one strided system
/// and must produce, bit for bit, the solution and the matrix of the
/// same system solved alone in contiguous arrays. Both solvers, the
/// standard batch (second pass included), a batch size that is not a
/// multiple of the AoSoA width.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "arrowhead_problems.hpp"
#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Batch layout.
enum class Batch { aos, soa, aosoa };

/// Systems of a batch, and the AoSoA width.
constexpr int count = 13;
constexpr int width = 4;

/// \brief Index of (element, system) in a batch buffer.
/// \param[in] batch  layout
/// \param[in] e      element of the object
/// \param[in] s      system
/// \param[in] object number of elements of one object
/// \return the buffer index
std::size_t index(const Batch batch, const int e, const int s, const int object) {
    const auto E = static_cast<std::size_t>(e), Sy = static_cast<std::size_t>(s),
               O = static_cast<std::size_t>(object);
    switch (batch) {
    case Batch::aos:
        return tdls_tests::aos_index(E, Sy, O);
    case Batch::soa:
        return tdls_tests::soa_index(E, Sy, count);
    default:
        return tdls_tests::aosoa_index(E, Sy, O, width);
    }
}

/// \brief Stride of an object in a batch buffer.
/// \param[in] batch layout
/// \return the element stride
int stride(const Batch batch) {
    return batch == Batch::aos ? 1 : batch == Batch::soa ? count : width;
}

/// \brief Solves a batch in one layout with both solvers, against the
/// systems solved one by one in contiguous arrays.
/// \tparam Shape  shape
/// \tparam Config configuration
/// \param[in] batch layout
/// \param[in] seed  generator seed
template<tdls::ArrowheadShape Shape, tdls::ArrowheadLUppConfig<double> Config>
void layout_case(const Batch batch, const std::uint64_t seed) {
    using Static        = tdls::ArrowheadLUppSolverStatic<double, Shape, Config>;
    using Dynamic       = tdls::ArrowheadLUppSolverDynamic<double, Config>;
    constexpr bool diag = Config.coupling == tdls::ArrowheadCoupling::Diagonal;
    constexpr int size = Static::storage_size, n = Static::dimension,
                  wsize = Dynamic::workspace_size(Shape), psize = Dynamic::pivot_size(Shape);
    constexpr int padded = (count + width - 1) / width * width;
    std::vector<tdls_tests::ArrowProblem<double>> problems;
    for (int s = 0; s < count; ++s)
        problems.push_back(tdls_tests::batch_problem<double>(Shape, diag, s, seed));
    // Batch buffers: one set for each solver.
    std::vector<double> A(static_cast<std::size_t>(size) * padded),
        y(static_cast<std::size_t>(n) * padded), w(static_cast<std::size_t>(wsize) * padded);
    std::vector<int> piv(static_cast<std::size_t>(psize) * padded);
    for (int round = 0; round < 2; ++round) {
        for (int s = 0; s < count; ++s) {
            const auto M = tdls_tests::store_arrow<typename Static::storage_map>(problems[s]);
            for (int e = 0; e < size; ++e)
                A[index(batch, e, s, size)] = M[static_cast<std::size_t>(e)];
            for (int i = 0; i < n; ++i)
                y[index(batch, i, s, n)] = problems[s].b[static_cast<std::size_t>(i)];
        }
        const int st = stride(batch);
        for (int s = 0; s < count; ++s) {
            int refused = 0;
            const bool ok =
                round == 0 ? Static::template solve_inplace<false, false, false>(
                                 &A[index(batch, 0, s, size)], st, &w[index(batch, 0, s, wsize)],
                                 st, &y[index(batch, 0, s, n)], st, refused)
                           : Dynamic::solve_inplace(Shape, &A[index(batch, 0, s, size)], st,
                                                    &piv[index(batch, 0, s, psize)], st,
                                                    &w[index(batch, 0, s, wsize)], st,
                                                    &y[index(batch, 0, s, n)], st, refused);
            // The same system alone, contiguous.
            auto M1 = tdls_tests::store_arrow<typename Static::storage_map>(problems[s]);
            auto y1 = problems[s].b;
            std::vector<double> w1(Static::workspace_size);
            int refused1   = 0;
            const bool ok1 = Static::template solve_inplace<true, true, true>(
                M1.data(), 1, w1.data(), 1, y1.data(), 1, refused1);
            TDLS_CHECK(ok && ok1);
            TDLS_CHECK(refused == refused1);
            std::vector<double> xb(static_cast<std::size_t>(n)), Mb(static_cast<std::size_t>(size));
            for (int i = 0; i < n; ++i)
                xb[static_cast<std::size_t>(i)] = y[index(batch, i, s, n)];
            for (int e = 0; e < size; ++e)
                Mb[static_cast<std::size_t>(e)] = A[index(batch, e, s, size)];
            TDLS_CHECK(tdls_tests::same_bits(xb, y1));
            TDLS_CHECK(tdls_tests::same_bits(Mb, M1));
        }
    }
}

/// \brief A configuration from its coupling and layout, Packed storage.
/// \param[in] diag diagonal coupling
/// \param[in] col  column-major layout
/// \return the configuration
constexpr tdls::ArrowheadLUppConfig<double> layout_config(const bool diag, const bool col) {
    tdls::ArrowheadLUppConfig<double> c;
    c.block_row_interchange = tdls::RowInterchange::Logical;
    c.coupling = diag ? tdls::ArrowheadCoupling::Diagonal : tdls::ArrowheadCoupling::Dense;
    c.layout   = col ? tdls::MatrixLayout::ColMajor : tdls::MatrixLayout::RowMajor;
    c.storage  = tdls::ArrowheadStorage::Packed;
    return c;
}

constexpr tdls::ArrowheadShape polycrystal{12, 6, 10, 6};
constexpr tdls::ArrowheadShape dense_shape{8, 6, 10, 0};

} // namespace

TDLS_TEST_CASE("arrowheadlupp/bridge/layouts/aos") {
    layout_case<polycrystal, layout_config(true, false)>(Batch::aos, 800100);
    layout_case<dense_shape, layout_config(false, true)>(Batch::aos, 800200);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/layouts/soa") {
    layout_case<polycrystal, layout_config(true, true)>(Batch::soa, 800300);
    layout_case<dense_shape, layout_config(false, false)>(Batch::soa, 800400);
}

TDLS_TEST_CASE("arrowheadlupp/bridge/layouts/aosoa") {
    layout_case<polycrystal, layout_config(true, false)>(Batch::aosoa, 800500);
    layout_case<dense_shape, layout_config(false, true)>(Batch::aosoa, 800600);
}

TDLS_TEST_MAIN
