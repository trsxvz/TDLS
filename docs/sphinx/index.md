# TDLS - Tiny Device-callable Linear Solvers

TDLS is a header-only C++17 library of direct solvers for small
general linear systems, written to be callable from device code: one thread
solves one system, on CPU as well as inside a CUDA, HIP, SYCL, Kokkos,
AdaptiveCpp, stdpar or OpenMP (host and offload) kernel. It has no
dependency and no installation step.

The only solver family available today is TiledLUpp, an LU
factorization with logical partial pivoting operating on a tile grid,
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
and AoSoA layouts both provide memory coalescence.

A taste of the interface, from the examples:

```cpp
#include <tdls/tdls.hpp>

// LUpp solver with compile-time matrix size.
using StaticSolver  = tdls::TiledLUppSolverStatic<double, 9, tdls::TiledLUppConfig<double, 3>>;
// LUpp solver with runtime matrix size.
using DynamicSolver = tdls::TiledLUppSolverDynamic<double>;

// Both variants offer two interfaces:
// - split: factorize once, then one substitute call per right-hand
//   side, reusing the factorization;
// - one-call: solve chains factorize and substitute.

// Static variant, split interface, contiguous storage (stride 1).
// substitute reads b and writes x; substitute_inplace overwrites its
// single buffer y instead.
StaticSolver::factorize<true, true>(M, 1, piv, 1);
StaticSolver::substitute<true, true, true>(M, 1, piv, 1, b, x, 1);
StaticSolver::substitute_inplace<true, true, true>(M, 1, piv, 1, y, 1);

// Runtime variant, one-call interface, on a structure-of-arrays batch
// of B interleaved systems: element k of the system number s sits at
// A[k * B + s]. Every pointer is offset by s, every stride is B.
DynamicSolver::solve(n, A + s, B, piv + s, B, b + s, x + s, B);
```

Dense math objects (matrices, vectors, strided views) can also be
passed directly through the structural adaptors, without naming any
external library; see {doc}`tfel_interoperability`.

```{toctree}
:maxdepth: 1

getting_started
examples
tfel_interoperability
api_reference
```
