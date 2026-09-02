# TDLS - Tiny Device-callable Linear Solvers

TDLS is a header-only C++20 library of direct solvers for small
general linear systems. It is written to be callable from device code:
one thread solves one system, on CPU as well as inside a CUDA, HIP,
SYCL, Kokkos, RAJA, OpenMP, OpenACC or parallel STL kernel. The
solvers are designed for maximum GPU performance. The library has no
dependency.

The only solver family available today is TiledLUpp, an LU
factorization with logical partial pivoting on a tile grid,
in two variants:

- `tdls::TiledLUppSolverStatic<T, N, Config>`: the dimension N is a
  compile-time constant. Residency template booleans let the operands
  live in registers (compile-time indexing) or in external memory
  (strided accesses).
- `tdls::TiledLUppSolverDynamic<T, Config>`: the dimension n is a
  runtime value. The placement of the operands is expressed through
  runtime element strides.

Both variants address every operand through a single element stride.
This covers the three batch layouts alike: AoS (contiguous storage,
stride 1), SoA (stride = batch size) and AoSoA (tiled hybrid); the SoA
and AoSoA layouts both provide memory coalescence. The layout never
appears in the calls: every solver argument is one (pointer, stride)
pair. The table gives this pair for system b of a batch of B systems.
g is the base pointer of the batch buffer. M is the element count of
one object: N * N for a matrix, N for a right-hand side or a pivot.

| layout           | pointer of system b   | element stride |
|------------------|-----------------------|----------------|
| AoS              | `g + b*M`             | `1`            |
| SoA              | `g + b`               | `B`            |
| AoSoA of width W | `g + (b/W)*M*W + b%W` | `W`            |

An AoSoA batch is padded to a multiple of W systems, so that every
block is full and the stride stays uniform.

A taste of the interface:

```cpp
#include <limits>

#include <tdls/tdls.hpp>

// Solver configuration, shared by both variants: a constexpr value,
// every knob spelled out. Designated initializers override individual
// knobs; tdls::TiledLUppConfig<double>{} alone keeps the ready-made
// defaults.
constexpr tdls::TiledLUppConfig<double> config{
    // int: size of the tiles used for the LU factorization
    .tile_size = 3,
    // tdls::TiledLUppSchedule: elimination schedule, RightLooking or LeftLooking
    .schedule = tdls::TiledLUppSchedule::RightLooking,
    // double (the scalar type T): a pivot at least this large is accepted
    // without searching outside the tile
    .oot_threshold = 1e-10,
    // double (the scalar type T): the factorization is declared singular
    // when the best pivot falls below this floor
    .singular_eps = std::numeric_limits<double>::min(),
    // bool: the out-of-tile search stops at the first acceptable pivot
    .oot_first_acceptable = true,
    // bool: forced unrolling of the in-tile loops
    .unroll_inner = true,
    // tdls::TiledLUppLayout: matrix layout, RowMajor or ColMajor
    .layout = tdls::TiledLUppLayout::RowMajor};

// LUpp solver with compile-time matrix size.
using StaticSolver  = tdls::TiledLUppSolverStatic<double, 9, config>;
// LUpp solver with runtime matrix size.
using DynamicSolver = tdls::TiledLUppSolverDynamic<double, config>;

// Both variants offer two interfaces:
// - split: factorize once, then one substitute call per right-hand
//   side, reusing the factorization;
// - one-call: solve (two buffers) or solve_inplace (single buffer).
// Each substitution and solve entry point also has a _multirhs twin
// taking several right-hand-side columns per call (A X = B), every
// tile loaded once for the whole block.

// Static variant, split interface, contiguous storage (stride 1).
// substitute reads b and writes x; substitute_inplace overwrites its
// single buffer y instead.
StaticSolver::factorize<true, true>(M, 1, piv, 1);
StaticSolver::substitute<true, true, true>(M, 1, piv, 1, b, x, 1);
StaticSolver::substitute_inplace<true, true, true>(M, 1, piv, 1, y, 1);

// Runtime variant, one-call interface, on a structure-of-arrays batch
// of interleaved systems: element k of the system number s sits at
// A[k * stride + s], where the stride is the number of systems. The
// pivot array stays local to the caller (stride 1): placements can be
// mixed freely, one stride per operand. solve reads b and writes x;
// solve_inplace overwrites its single buffer y and folds the forward
// pass into the factorization.
DynamicSolver::solve_inplace(matrixSize, A + s, stride, piv, 1, y + s, stride);
```

## Diagnostics

Every factorizing entry point (`factorize`, `solve`, `solve_inplace`
and their `_multirhs` twins) has an overload taking a trailing
`int& oot_count` argument. It counts the columns whose best in-tile
pivot fell below `oot_threshold`, because this is what triggers the
out-of-tile pivot search. Without the argument, the diagnostic is
compiled out entirely and costs nothing. The adaptors expose the same
overloads: a trailing `int&` on `factorize`, `solve` and
`solve_inplace`.

```cpp
// 9 x 9 matrix, its pivot vector, and the out-of-tile counter
double M[9 * 9] = ...;
int piv[9];
int oot_count;

// Factorization of the matrix M with out-of-tile diagnostics enabled
StaticSolver::factorize<true, true>(M, 1, piv, 1, oot_count);

// Post-processing
if (oot_count == 0) {
    std::printf("no out-of-tile pivoting\n");
} else {
    // Weak in-tile pivots force the search below the tile: slower execution
    std::printf("%d columns needed the out-of-tile pivot search\n", oot_count);
}
```

## TFEL adaptors

The `tfel::math` objects (matrices, vectors, strided views) can also
be passed directly through the adaptors of `tdls/tfel/adaptors.hpp`,
which recognize them structurally without including `TFEL`; see
{doc}`tfel_interoperability`.

```{toctree}
:maxdepth: 1

Overview <self>
getting_started
examples
tests
tfel_interoperability
api_reference
ai_usage
```
