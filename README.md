# TDLS - Tiny Device-callable Linear Solvers

[![ci](https://github.com/trsxvz/TDLS/actions/workflows/ci.yml/badge.svg)](https://github.com/trsxvz/TDLS/actions/workflows/ci.yml)
[![doxygen](https://github.com/trsxvz/TDLS/actions/workflows/doxygen.yml/badge.svg)](https://trsxvz.github.io/TDLS/main/api/index.html)
[![docs](https://github.com/trsxvz/TDLS/actions/workflows/docs.yml/badge.svg)](https://trsxvz.github.io/TDLS/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)](https://en.cppreference.com/cpp/20)
[![spack](https://img.shields.io/spack/v/tdls)](https://packages.spack.io/package.html?name=tdls)

TDLS is a header-only C++20 library of direct solvers for small
general linear systems. It is written to be callable from device code:
one thread, or a group of threads, solves one system, on CPU as well
as inside a CUDA, HIP, SYCL, Kokkos, RAJA, OpenMP, OpenACC or parallel
STL kernel. The solvers are designed for maximum GPU performance. The
library has no dependency.

Two solver families are available, both LU factorizations with partial
pivoting:

- TiledLUpp: one thread per system, logical pivoting on a tile grid.
- CooperativeLUpp: a group of threads per system, each thread holding
  some of its rows, synchronized by a barrier deduced from the
  compilation target or provided by the caller; a single thread is a
  valid group. Derived from
  [MAGMA](https://github.com/icl-utk-edu/magma/blob/v2.10.0/magmablas/zgesv_batched_small.cu).

Here are two snippets, one per family, with every knob spelled out.

```cpp
#include <limits>

#include <tdls/tdls.hpp>

// Solver configuration, a constexpr value, every knob spelled out.
// Designated initializers override individual knobs;
// tdls::TiledLUppConfig<double>{} alone keeps the ready-made defaults.
constexpr tdls::TiledLUppConfig<double> config{
    // int: size of the tiles used for the LU factorization
    .tile_size = 3,
    // tdls::Schedule: elimination schedule, RightLooking or LeftLooking
    .schedule = tdls::Schedule::RightLooking,
    // double (the scalar type T): a pivot at least this large is accepted
    // without searching outside the tile
    .oot_pivot_threshold = 1e-10,
    // bool: the out-of-tile search stops at the first acceptable pivot
    .oot_first_acceptable = true,
    // double (the scalar type T): the factorization is declared singular
    // when the best pivot falls below this floor
    .singular_floor = std::numeric_limits<double>::min(),
    // bool: forced unrolling of the in-tile loops, the ones where it pays
    .unroll_loops = true,
    // tdls::MatrixLayout: matrix layout, RowMajor or ColMajor
    .layout = tdls::MatrixLayout::RowMajor};

// LU solver for systems of dimension 9: the 9 x 9 matrix is cut
// into 3 x 3 register tiles. One call: factorize M in place, then
// overwrite y with the solution; false means a singular matrix. The
// residency booleans declare every operand caller-local, so the
// stride arguments (the 1s) are ignored at compile time; with
// external operands they carry the element stride of each array.
using Solver = tdls::TiledLUppSolverStatic<double, 9, config>;
const bool ok = Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1);
```

```cpp
#include <limits>

#include <tdls/tdls.hpp>

// Solver configuration, a constexpr value, every knob spelled out.
// Designated initializers override individual knobs;
// tdls::CooperativeLUppConfig<double>{} alone keeps the defaults.
constexpr tdls::CooperativeLUppConfig<double> config{
    // int: rows of the system held by each thread of the group
    .rows_per_thread = 3,
    // tdls::RowInterchange: Logical (rows never move) or Physical (rows
    // move between the threads, as in LAPACK); same pivots, same operations
    .row_interchange = tdls::RowInterchange::Logical,
    // double (the scalar type T): the row in place keeps the pivot when it
    // reaches this fraction of the largest magnitude of its column; 1 gives
    // the pivots of LAPACK
    .relative_pivot_threshold = 0.1,
    // double (the scalar type T): the factorization is declared singular
    // when the best pivot falls below this floor
    .singular_floor = std::numeric_limits<double>::min(),
    // bool: forced unrolling of the loops over the rows of the thread,
    // the ones where it pays
    .unroll_loops = true,
    // tdls::MatrixLayout: matrix layout, RowMajor or ColMajor
    .layout = tdls::MatrixLayout::RowMajor};

// LU solver for systems of dimension 9, shared by a group of 3 threads
// holding 3 rows each. Every thread of the group makes the same call
// with its rank tx in the group and the workspace of the group
// (Solver::workspace_size elements in memory shared by the group). The
// solver deduces the barrier of the group from the compilation target:
// the lanes of a warp on GPU, threads on CPU. An optional last argument
// passes another barrier, any callable. The residency booleans declare
// every operand external: the whole matrix, pivot and right-hand side,
// reached by the group with an element stride (the 1s). A single thread
// holding every row passes 0 for tx.
using Solver = tdls::CooperativeLUppSolverStatic<double, 9, config>;
const bool ok = Solver::solve_inplace<false, false, false>(tx, M, 1, piv, 1, y, 1, work);
```

The `tfel::math` objects (matrices, vectors, strided views) can also
be passed directly to the TiledLUpp solvers: the adaptors of
`tdls/tfel/adaptors.hpp` are designed for them and recognize them
structurally, without including `TFEL`.

## Using the library

Copy the `include/` directory into a project, add the source tree as
a subdirectory, or install it and use `find_package`:

```cmake
# source tree, through add_subdirectory or FetchContent
add_subdirectory(tdls)
target_link_libraries(my_target PRIVATE tdls::tdls)

# installed tree (cmake --install, or spack install tdls)
find_package(tdls REQUIRED)
target_link_libraries(my_target PRIVATE tdls::tdls)
```

## Tests and examples

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target check   # build and run everything
ctest --test-dir build -L solvers    # rerun the library test suites
ctest --test-dir build -L examples   # rerun the self-checking examples
ctest --test-dir build -L snippets   # rerun the documentation snippets
```

Tests and examples stay out of the default `all` target, as in Eigen
or TFEL: `check` builds and runs them, `buildtests` only builds them.

The examples cover three scientific problems for each solver family,
at four execution scales: sequential, OpenMP, and two GPU placements,
plus SYCL and the parallel STL. The GPU and SYCL examples are opt-in
through `TDLS_BUILD_CUDA_EXAMPLES`, `TDLS_BUILD_HIP_EXAMPLES`,
`TDLS_BUILD_SYCL_EXAMPLES` or `TDLS_BUILD_STDPAR_DEVICE_EXAMPLES`;
nothing probes for a GPU toolchain otherwise.

All the build options (the first three default to ON when TDLS is the
top-level project, OFF when it is consumed through `add_subdirectory`;
the GPU switches always default to OFF):

| Option | Effect |
|--------|--------|
| `TDLS_BUILD_TESTS` | build the test suites |
| `TDLS_BUILD_EXAMPLES` | build the examples and the documentation snippets |
| `TDLS_INSTALL` | generate the install and `find_package(tdls)` rules |
| `TDLS_TESTS_OPTIMIZATION` | optimization flag of the test suites in the Release configuration (`-O1` by default) |
| `TDLS_BUILD_CUDA_EXAMPLES` | build the GPU examples with CUDA (the toolchain becomes required) |
| `CMAKE_CUDA_COMPILER` | CUDA compiler to use when `nvcc` is not in the `PATH` |
| `CMAKE_CUDA_ARCHITECTURES` | target GPU architectures (`native` if unset) |
| `TDLS_BUILD_HIP_EXAMPLES` | build the GPU examples with HIP (the toolchain becomes required) |
| `CMAKE_HIP_COMPILER` | HIP compiler of the AMD toolchain |
| `CMAKE_HIP_ARCHITECTURES` | target GPU architectures (the compiler default if unset) |
| `TDLS_BUILD_SYCL_EXAMPLES` | build the SYCL examples (a SYCL compiler such as `icpx` becomes required) |
| `TDLS_SYCL_FLAGS` | SYCL flags of that compiler (`-fsycl` by default; add `-fsycl-targets=...` for AOT) |
| `TDLS_BUILD_STDPAR_DEVICE_EXAMPLES` | build the parallel STL examples for a GPU (an offloading compiler becomes required) |
| `TDLS_STDPAR_DEVICE_FLAGS` | offload flags of that compiler (`--acpp-stdpar`, `-stdpar=gpu`, `--hipstdpar` ...) |
| `TDLS_BUILD_TFEL_SNIPPETS` | build the TFEL snippets of the documentation (TFEL becomes required, found by `find_package(TFELMath)`) |

## Documentation

The documentation, including the Doxygen API reference, lives at
<https://trsxvz.github.io/TDLS/>.

## Use of AI

TDLS is developed with the help of AI coding assistants, currently
Claude Fable 5.1 (Anthropic) and GPT 5.6 (OpenAI). They are used as
tools: code drafting, refactoring, test scaffolding, documentation
editing. The scientific reasoning is human. The choice of algorithms,
the design of the linear solvers and the interpretation of the
measurements are the author's. All AI-assisted output is reviewed,
tested against the test suites and validated by the author, who takes
full responsibility for it.

Contributions prepared with an AI assistant are welcome under the same
rule: the scientific reasoning and the responsibility for the
correctness of what you submit remain yours.

## License

Copyright (C) 2026 CEA. All rights reserved.

TDLS is publicly released under the BSD 3-Clause License (see the
LICENSE file); CEA may also distribute it under specific licensing
conditions. The kernels of the CooperativeLUpp solvers are derived from
MAGMA and remain subject to the MAGMA license, BSD 3-Clause as well
(see the NOTICE file). TDLS is designed to be embedded in
[TFEL/MFront](https://github.com/thelfer/tfel); the BSD-3-Clause
license places no meaningful restriction on such use.
