# Getting started

## Requirements

Only a C++20 compiler is mandatory: the library itself has no
dependency. The oldest releases known to compile the headers are GCC
10, Clang 12, Visual Studio 2019 16.11, CUDA 12.0 and ROCm 5.3.
Running every example additionally needs an OpenMP runtime, oneTBB or
nvc++ for the parallel STL on the CPU cores, a CUDA or HIP toolchain, a
SYCL compiler and a parallel STL offload compiler.

## Consuming the library

TDLS is header-only. It can be consumed in three ways.

Copy the `include/` directory into a project and add it to the
include path.

Add the source tree as a subdirectory, or through FetchContent:

```cmake
add_subdirectory(tdls)
target_link_libraries(my_target PRIVATE tdls::tdls)
```

Install it, then use `find_package`:

```sh
cmake -S . -B build -DTDLS_BUILD_TESTS=OFF -DTDLS_BUILD_EXAMPLES=OFF
cmake --install build --prefix /opt/tdls
```

```cmake
find_package(tdls REQUIRED)
target_link_libraries(my_target PRIVATE tdls::tdls)
```

The install tree is found through `CMAKE_PREFIX_PATH` or `tdls_DIR`
(`<prefix>/lib/cmake/tdls`). TDLS is also available from the Spack
package repository: `spack install tdls`.

In every case, include the umbrella header:

```cpp
#include <tdls/tdls.hpp>
```

Finer-grained headers exist for the individual pieces
(`tdls/solvers/tiled_lupp/solver_static.hpp`,
`tdls/solvers/tiled_lupp/solver_dynamic.hpp`, `tdls/tfel/adaptors.hpp`).

The headers compile as plain C++ under every programming model. Under
CUDA and HIP the entry points decorate themselves: nothing has to be
defined. The version is exposed by `TDLS_VERSION_MAJOR`,
`TDLS_VERSION_MINOR`, `TDLS_VERSION_PATCH`, `TDLS_VERSION_STRING` and
the comparable `TDLS_VERSION`.

## Building the tests and examples

Both are ordinary CMake targets, enabled by default when TDLS is the
top-level project and disabled when it is consumed through
`add_subdirectory` (options `TDLS_BUILD_TESTS` and
`TDLS_BUILD_EXAMPLES`):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target check   # build and run everything
ctest --test-dir build -L solvers    # rerun the library test suites
ctest --test-dir build -L examples   # rerun the self-checking examples
ctest --test-dir build -L snippets   # rerun the documentation snippets
```

Tests and examples stay out of the default `all` target, as in Eigen
or TFEL: `check` builds and runs them, `buildtests` only builds them.
In the Release configuration the test suites compile at `-O1`, which
keeps the fully unrolled solver templates fast to build;
`TDLS_TESTS_OPTIMIZATION` changes that flag.

The OpenMP examples build when an OpenMP runtime is found, the
parallel STL examples when the compiler offers a backend (nvc++, or
libstdc++ with oneTBB). The GPU examples are opt-in: set
`TDLS_BUILD_CUDA_EXAMPLES` or `TDLS_BUILD_HIP_EXAMPLES` to ON, and the
requested toolchain becomes mandatory (point `CMAKE_CUDA_COMPILER` /
`CMAKE_HIP_COMPILER` at a compiler outside the `PATH`).
`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so set it explicitly
when the build machine has no device. At run time the GPU examples
report themselves as skipped when no device is present. The SYCL
examples are opt-in through `TDLS_BUILD_SYCL_EXAMPLES`: the C++
compiler must then accept the SYCL flags, `-fsycl` by default with
icpx, overridable through `TDLS_SYCL_FLAGS`. No device is needed to
build them; at run time they report themselves skipped when no SYCL
device is present. The parallel STL examples run on a GPU with
`TDLS_BUILD_STDPAR_DEVICE_EXAMPLES`: the offload is a matter of
compiler flags, stated in `TDLS_STDPAR_DEVICE_FLAGS` (`--acpp-stdpar`
with AdaptiveCpp, `-stdpar=gpu` with nvc++, `--hipstdpar` with the
ROCm clang).
The TFEL snippets of the documentation are opt-in through
`TDLS_BUILD_TFEL_SNIPPETS`; TFEL is then found by
`find_package(TFELMath)`.

## A first solve

TiledLUpp is the most general factorization available today, so it
makes the first call. The configuration is the default one.

```cpp
#include <tdls/tdls.hpp>

// dimension 9, default configuration
using Solver = tdls::TiledLUppSolverStatic<double, 9>;

double M[9 * 9] = /* the matrix, row-major */;
double y[9]     = /* the right-hand side */;
int piv[9];

// every operand is a caller-local array: the residency booleans are
// true and the strides, the 1s, are ignored at compile time
const bool ok = Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1);
// M holds the factors and y the solution; false means a singular matrix
```

The {doc}`TiledLUpp <tiled_lupp/index>` page explains the
configuration and the entry points.

## Calling convention

One call solves one system. Every operand is a raw pointer pre-offset
by the caller plus one runtime element stride: the solvers never see a
thread index or a batch layout. The same (pointer, stride) pair covers
the three batch layouts. The table gives it for system b of a batch of
B systems. g is the base pointer of the batch buffer. M is the element
count of one object: N * N for a matrix, N for a right-hand side or a
pivot.

| layout           | pointer of system b   | element stride |
|------------------|-----------------------|----------------|
| AoS              | `g + b*M`             | `1`            |
| SoA              | `g + b`               | `B`            |
| AoSoA of width W | `g + (b/W)*M*W + b%W` | `W`            |

An AoSoA batch is padded to a multiple of W systems, so that every
block is full and the stride stays uniform. The SoA and AoSoA layouts
both give memory coalescence on GPU.

The scalar type `T` is `float`, `double` or `long double`.

The factorizing entry points return `false` on a singular matrix. The
return value is `[[nodiscard]]`. The substitutions return nothing: they
cannot fail on a factorization that succeeded.

Offsets are computed in 32-bit arithmetic. The flat element index of
an object (N*N for a matrix) must stay below 2^31, and every element
offset, index times stride, below 2^32.
