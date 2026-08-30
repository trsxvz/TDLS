# Getting started

## Requirements

Only a C++20 compiler is mandatory: the library itself has no
dependency. The oldest releases known to compile the headers are GCC
10, Clang 12, Visual Studio 2019 16.11, CUDA 12.0 and ROCm 5.3.
Running every example additionally needs an OpenMP runtime and a CUDA
or HIP toolchain.

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
`tdls/solvers/tiled_lupp/solver_dynamic.hpp`, `tdls/core/adaptors.hpp`).

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
```

Tests and examples stay out of the default `all` target, as in Eigen
or TFEL: `check` builds and runs them, `buildtests` only builds them.

The OpenMP examples build when an OpenMP runtime is found, the
parallel STL examples when the compiler offers a backend (nvc++, or
libstdc++ with oneTBB). The GPU examples are opt-in: set
`TDLS_BUILD_CUDA_EXAMPLES` or `TDLS_BUILD_HIP_EXAMPLES` to ON, and the
requested toolchain becomes mandatory (point `CMAKE_CUDA_COMPILER` /
`CMAKE_HIP_COMPILER` at a compiler outside the `PATH`).
`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so set it explicitly
when the build machine has no device. At run time the GPU examples
report themselves as skipped when no device is present. The parallel
STL examples run on a GPU with `TDLS_BUILD_STDPAR_DEVICE_EXAMPLES`:
the offload is a matter of compiler flags, stated in
`TDLS_STDPAR_DEVICE_FLAGS` (`--acpp-stdpar` with AdaptiveCpp,
`-stdpar=gpu` with nvc++, `--hipstdpar` with the ROCm clang).

## Portability model

The headers decorate every entry point with a small set of macros
(`TDLS_HOST_DEVICE`, `TDLS_FORCEINLINE`, `TDLS_RESTRICT`,
`TDLS_UNROLL_FORCE`) that resolve to the right annotation for the
compiler at hand: `__host__ __device__` under CUDA, the equivalent
attributes under HIP (no HIP header needs to be included first), plain
host code elsewhere. Single-source models (SYCL, stdpar, OpenMP
target, Kokkos through its backend compiler) need no decoration at
all. Every macro is `#ifndef`-guarded, so any of them can be
overridden from the command line without editing the headers.
