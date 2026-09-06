# Changelog

All notable changes to TDLS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
versions follow [Semantic Versioning](https://semver.org/). Until 1.0.0
the API may change between minor versions.

## [0.1.0] - 2026-09-07

First release. TDLS is a header-only library of direct solvers for
small linear systems. It is designed for GPU performance and callable
from device code.

### Added

- TiledLUpp: LU factorization with logical partial pivoting on a tile
  grid. The pivot is searched below the tile only when the tile holds
  no acceptable one. Two solvers, one with a compile-time dimension and
  one with a runtime dimension. At equal shape and configuration their
  results are bitwise identical.
- Configuration by a constexpr value: tile size, elimination schedule,
  pivoting thresholds, inner unrolling and matrix layout. One knob
  stops the search below the tile at the first candidate reaching the
  threshold, instead of scanning for the largest.
- Entry points: factorize, substitute and solve, their in-place forms,
  canonical columns, and blocks of right-hand sides cut into passes.
  Overloads count the columns whose pivot was searched below the tile.
- Every operand, matrix, pivots and right-hand side, is a pointer plus
  a stride. Residency booleans declare, per operand, a local array or
  remote memory walked with the stride. The convention covers the AoS,
  SoA and AoSoA batch layouts.
- TFEL adaptors: `tfel::math` objects are accepted by structural
  detection, without a dependency on TFEL.
- Device-callable headers. Example programs in CUDA, HIP, SYCL, OpenMP
  and parallel STL on three physical problems.
- Test suites: independent oracle, bitwise bridging, constexpr
  certificates and compile-time contract checks. The example programs
  and the documentation snippets are self-checking and run under ctest.
- Documentation site with the Doxygen reference and a snippet gallery.
- CMake package for `find_package(tdls)` and the Spack package `tdls`.

[0.1.0]: https://github.com/trsxvz/TDLS/releases/tag/v0.1.0
