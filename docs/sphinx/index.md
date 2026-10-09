# TDLS - Tiny Device-callable Linear Solvers

TDLS is a header-only C++20 library of direct solvers for small
general linear systems. It is written to be callable from device code:
one thread, or a group of threads, solves one system, on CPU as well as
inside a CUDA, HIP, SYCL, Kokkos, RAJA, OpenMP, OpenACC or parallel STL
kernel. The solvers are designed for maximum GPU performance. The
library has no dependency.

TDLS is designed to be embedded in
[TFEL/MFront](https://github.com/thelfer/tfel). The TiledLUpp solvers
accept the `tfel::math` objects directly, without including TFEL.

## Solvers

TDLS implements two solver families, both LU factorizations with
partial pivoting, each in a compile-time dimension and a runtime
dimension variant:

- {doc}`TiledLUpp <tiled_lupp/index>`: one thread per system, logical
  pivoting on a tile grid.
- {doc}`CooperativeLUpp <cooperative_lupp/index>`: a group of threads
  per system, each thread holding some of its rows, synchronized by a
  barrier provided by the caller. Derived from
  [MAGMA](https://github.com/icl-utk-edu/magma/blob/v2.10.0/magmablas/zgesv_batched_small.cu).

## Documentation map

- {doc}`getting_started`: requirements, installation, building the
  tests and examples, a first solve, and the calling convention shared
  by every solver.
- {doc}`TiledLUpp <tiled_lupp/index>`: how the solvers work, their
  configuration, and their snippets: one entry point per snippet, on a
  small system.
- {doc}`CooperativeLUpp <cooperative_lupp/index>`: the same for the
  family of the groups of threads.
- {doc}`TFEL interoperability <tfel/index>`: how the `tfel::math`
  objects are accepted, with snippets per solver family.
- {doc}`tests`: the method, the generic suites, then the suites and
  the example programs of each solver family. The examples are complete
  programs solving a physical problem on every execution scale.
- {doc}`api_reference`: the Doxygen reference of the headers.
- {doc}`ai_usage`: the use of AI assistants in the project.

```{toctree}
:maxdepth: 2
:hidden:

Overview <self>
getting_started
tiled_lupp/index
cooperative_lupp/index
tfel/index
tests
api_reference
ai_usage
```
