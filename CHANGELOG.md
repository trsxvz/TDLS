# Changelog

All notable changes to TDLS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
versions follow [Semantic Versioning](https://semver.org/). Until 1.0.0
the API may change between minor versions.

## [Unreleased]

### Added

- ArrowheadLUpp: `condensation`, `MinFlops` (the default, unchanged
  behaviour) or `MinTraffic` (`ArrowheadCondensation`), the condensation of
  an accepted block under the row-major layout with the dense coupling.
  `MinFlops` solves the forward stage alone, updates the Schur
  complement by the border multipliers of the pivot check and stores U
  for the recovery; `MinTraffic` runs the whole solve, as the diagonal
  coupling does: more flops, but U neither written nor read again. The
  other paths run the whole solve either way. Measured with `MinTraffic`
  on an RTX 5080: 1.46 to 2.2 times faster streamed with blocks of 6,
  1.1 to 1.6 times stored with the systems one after the other (AoS),
  but 0.72 to 0.77 times in float with the systems interleaved (SoA)
  and blocks of 12; no trend on CPU. The same blocks are refused; the
  solutions differ by rounding errors. Both solvers, bitwise identical
  to each other.
- The compile-time solvers (TiledLUpp, CooperativeLUpp, ArrowheadLUpp)
  take the stride of each strided operand as a template argument too,
  `X_static_stride` beside the function argument `X_stride`,
  `tdls::dynamic_stride` by default, after the other template
  parameters but for the deduced types of the callables. A positive
  value, 1 in AoS or the width of a block in AoSoA, is the stride the
  solver uses, the function argument then ignored: the solver runs as
  if the caller had passed the constant, and the out-of-line parts of
  ArrowheadLUpp (the second pass, `streamed_out_of_line`) receive it as
  a template argument. The calls without these arguments are unchanged,
  but for those that spell out the type of the barrier of CooperativeLUpp
  or of a callable of ArrowheadLUpp: the strides now come before it. A
  bool given by mistake where a stride is now expected converts to 0,
  which is rejected, or to 1, which is accepted as a stride.
- TiledLUpp: `row_interchange`, `Logical` (the default) or `Physical`,
  the second knob, after the tile size and the schedule, as in
  CooperativeLUpp.
  Physical makes the whole matrix one tile and interchanges the rows as
  LAPACK does, through a test of every row index, so that every index
  is known at compile time once the loops are unrolled: on GPU a
  caller-local matrix, its pivots and its right-hand sides stay in
  registers. The pivots and the factorization are those of `Logical`
  over a single tile; the pivot array then holds the interchanges. For
  small systems: the matrix must fit in the registers of a thread. Both
  solvers, bitwise identical to each other.
- ArrowheadLUpp: `block_row_interchange`, the `row_interchange` of
  TiledLUpp on the diagonal blocks, `Auto` by default. `Physical`
  factors each block in registers, the block and its right-hand sides
  out of the stack of the thread on GPU; the pivot check, the recovery
  data and the second pass unchanged; the runtime solver makes the same
  choice and runs the same operations, so that both solvers stay
  bitwise identical. `Auto` takes `Physical` up to blocks of 14 in
  float and 10 in double on GPU, 8 in float and 12 in double on CPU,
  long double as double; `Logical`, in tiles, beyond. Measured over
  tiles of 3, blocks that keep their rows: 2 to 6 times faster on the
  recomputed path, 1.1 to 2.9 times streamed, 1 to 1.6 times stored
  (RTX 5080), 1.5 times on CPU (median).
- `RowInterchange::Auto`, the default of the blocks of ArrowheadLUpp
  only: the physical interchanges when the block is small enough to stay
  in registers on the target of the code, the logical ones otherwise.
  TiledLUpp and CooperativeLUpp reject it at compile time, their factors
  being able to cross from the device to the host.
- `tdls/core/target.hpp`: the one place where TDLS identifies the
  toolchain and the compilation pass (version tests stay where they
  serve): the compiler, the language (CUDA, HIP, SYCL), the side
  of the compilation pass (host, CUDA, NVPTX, AMDGCN, SPIR-V, GCC
  offloading, nvc++, AdaptiveCpp) and the hardware (CPU or GPU). The
  portability macros, the group barriers and the defaults that depend
  on the target read it. `TDLS_TARGET_GPU` (0 or 1) gives the hardware
  those defaults assume; it guesses the GPU where the pass cannot tell
  (the generic mode of AdaptiveCpp, SPIR-V), and `-DTDLS_TARGET_GPU=0`
  overrides the guess.
- ArrowheadLUpp: `solve_inplace_recomputed` of the compile-time solver,
  the blocks asked for twice, once to condense them and once to recover
  their unknowns, nothing of an accepted block stored: no matrix
  storage, the border block given alone, a workspace that holds only
  the refused blocks, at most a capacity of them (`deferred` beyond,
  y untouched). `RecomputedBlock::recovery` tells the provider which call
  it serves. `solve_inplace_recomputed_or_tangent`, the pass of a Newton
  iteration: the tangent when converged, the blocks then asked for once
  and y untouched. The runtime solver has both, bit for bit those of the
  compile-time one, with `recomputed_workspace_size(shape, capacity,
  nrhs)`, which does not grow with the number of blocks, and
  `valid_recomputed`.
- Documentation: the getting started page asks for the block size of
  the kernels on AMD GPUs, where HIP otherwise assumes blocks of 1024
  threads and the one-thread solvers spill to scratch memory;
  ArrowheadLUpp `block_tile_size` documents the pivoting over the whole
  block, for blocks that pivot.

### Fixed

- TiledLUpp documentation: a caller-local matrix does not stay in
  registers on GPU. The logical pivoting writes each tile back at the
  physical rows of the pivot array, known at run time only, so the
  matrix and its pivots live in the local memory of the thread; only
  the working tiles stay in registers. The file documentation, the
  `unroll_loops` knob, the Sphinx page, the test page and the GPU and
  parallel STL examples said otherwise. The `tile_size` and
  `unroll_loops` knobs now give the measurements inside implicit MFront
  behaviours, and `block_row_interchange` those on larger blocks.
- nvc++ 25.11 from -O2 on: wrong solutions, crashes and an internal
  compiler error, now worked around in the code that nvc++ generates
  only. The other compilers see the same code as before, the device
  pass of nvcc included. nvc++ fused loops that depend on each other, on
  the strength of the `__restrict__` of the pointers that the solvers
  pass to one another: TiledLUpp with physical row interchanges and
  ArrowheadLUpp returned wrong solutions. `TDLS_RESTRICT` is now empty
  under nvc++, which also ends the internal error. nvc++ also derived
  the addresses of some loops from a negative index that the loop never
  forms, extended with zeros as `TDLS_STRIDED` extends its index: the
  runtime solvers read 2^32 strides beyond their operands and crashed.
  The index is now extended with its sign under nvc++, which makes every
  address right. At -O3, the device code of nvc++ read the local memory
  out of bounds in the recomputed solve of ArrowheadLUpp with the dense
  coupling. The two loops of that solve now share one block under
  nvc++. Measured with nvc++: the same speed through -stdpar=gpu on an
  RTX 5080, 0 to 5 % slower on one CPU core than the former code, which
  was wrong.

### Changed

- TiledLUpp: `tile_size` defaults to 4 (3 before): tiles of 4 or 5 were
  the fastest inside implicit MFront behaviours (N = 12 to 19, double,
  RTX 5080), 3 up to 28 % slower at N = 19. The default configuration of
  the TFEL adaptors follows. On matrices that pivot, the tiles of 4 may
  choose other pivots: the solutions then move by rounding errors from
  those of 0.3.0.
- ArrowheadLUpp: `block_tile_size` defaults to 4 (3 before): tiles of 4
  are 1.1 to 1.3 times faster from blocks of 14 (RTX 5080).
  `block_row_interchange` comes right after it, as `row_interchange` in
  TiledLUpp and CooperativeLUpp.
- ArrowheadLUpp: `rhs_pass_width` defaults to -1, a width chosen for the
  target: passes of 2 on GPU with the blocks in registers, one pass
  otherwise (0, the former default, still forces one pass).
- ArrowheadLUpp: by default (`block_row_interchange = Auto`), blocks that
  fit in registers are eliminated there, with the pivots of one tile of
  the size of the block: the solutions of small blocks move by rounding
  errors from those of 0.3.0, and blocks that pivot are refused less
  often.

## [0.3.0] - 2026-10-14

### Added

- ArrowheadLUpp, a third solver family: arrowhead (bordered
  block-diagonal) systems, a border coupled to diagonal blocks that
  never couple to each other, solved by static condensation in a time
  linear in the number of blocks. Built on TiledLUpp, which factors the
  blocks and solves the border system. A pivot check bounds the
  multipliers of every block, border rows included
  (`condensation_pivot_threshold`); a refused block is eliminated with
  the border rows in a second pass, so that the solver succeeds
  whenever the matrix is invertible, rounding aside: a pivot below the
  singular floor of TiledLUpp, zero or subnormal, counts as singular. A
  compile-time shape solver and a runtime shape one, bitwise identical
  when the compiler contracts multiply-adds alike in both
  (`-ffp-contract=off`, or `--fmad=false` for nvcc, guarantees it); four
  storages (Full, Packed, Csr, Bsr) addressed through
  `ArrowheadStorageMap`, without index arrays at run time; dense or diagonal couplings; both layouts, each
  with its condensation path. Several right-hand sides through one
  condensation (`solve_inplace_multirhs`, `solve_multirhs`), each
  column bitwise identical to `solve_inplace` on it alone under the
  default `rhs_pass_width` (within rounding errors under another), and
  the border block of the inverse by unit columns of the border
  (`solve_border_canonical_multirhs`), the tangent operator of a
  behaviour loaded through the border; it stores no recovery data, so
  that without `schur_in_place` the matrix comes out unchanged. The
  stride between the columns of a block of right-hand sides,
  `xcol_stride`, is a `std::size_t`, as in TiledLUpp. The runtime
  solver checks the shape first, `valid(shape, nrhs)`, and returns
  false on a shape it rejects. Headers
  `tdls/solvers/arrowhead_lupp/{config,storage_map,solver_static,solver_dynamic}.hpp`,
  included by `tdls/tdls.hpp`. Tests, documentation pages, snippets and
  examples.
- Streamed blocks in both ArrowheadLUpp solvers: the blocks come from a
  provider, called exactly once per block and in order when the solver
  condenses it, instead of a stored matrix (`solve_inplace_streamed`,
  `solve_border_canonical_multirhs_streamed`). A refused block is kept
  for the second pass without being asked for again. In a Newton
  iteration, `solve_inplace_streamed_or_tangent` asks `converged()`
  after the last block and then gives the tangent from the border
  system the pass has just condensed, instead of the correction; it
  returns a `StreamedOutcome`. The results are those of the stored
  entry points, and the runtime solver gives those of the compile-time
  one, bit for bit when the compiler contracts multiply-adds alike in
  the two paths. The streamed path is selected at compile time: the
  stored entry points do not carry it. `streamed_out_of_line`, off by
  default, keeps the streamed entry points of the compile-time solver
  out of line on GPU. Bridge suite, documentation page and snippets.
- `relative_pivot_threshold` in TiledLUppConfig, the knob of
  CooperativeLUpp applied to the diagonal tile: the row in place keeps
  the pivot when it reaches this fraction of the best in-tile candidate
  and `oot_pivot_threshold`. It governs the in-tile choice only: the
  out-of-tile recovery keeps its absolute trigger. The in-tile
  multipliers stay bounded by its inverse, and the solvers enforce the
  range (0, 1] at compile time.
- `forward_inplace_multirhs` and `backward_inplace_multirhs` in both
  TiledLUpp solvers: the two stages of `substitute_inplace_multirhs`
  (L^-1 P B, then U^-1; under the column-major layout U^-T B, then
  P^T L^-T), so that the caller can read or update the intermediate
  columns. In a row they match `substitute_inplace_multirhs` bit for bit.
  `substitute_inplace_multirhs` is now built from the same sweeps;
  existing calls keep their results and their speed.
- `TDLS_FORCE_INLINE_ALL`, inlining forced on every backend, CPU
  included, for helpers factored out of a solver routine.
- `TDLS_NO_INLINE`, keeping a rare path out of line on every backend,
  and `TDLS_NO_INLINE_GPU`, the same on GPU only, the pair of
  `TDLS_FORCE_INLINE_GPU`.
- A device suite of TiledLUppSolverDynamic with dimensions known at
  compile time, built with the GPU example toolchain (see Fixed).

### Changed

- Breaking: `TDLS_FORCEINLINE` is renamed `TDLS_FORCE_INLINE_GPU`, the
  pair of `TDLS_FORCE_INLINE_ALL`: the suffix says where the inlining
  is forced. There is no alias: a command-line override of the macro,
  or code that uses it, must use the new name. It now also forces the
  inlining in the device pass of SYCL (`__SYCL_DEVICE_ONLY__`), as
  under CUDA and HIP.
- Element offsets in remote memory, index times stride, are computed in
  64-bit arithmetic (`TDLS_STRIDED`) by every solver. They were 32-bit:
  a batch over 2^32 elements, a SoA batch over 32 GiB of doubles on a
  large GPU, wrapped around silently. Same results bit for bit, no
  perceptible cost.
- The stride between the columns of the multi-right-hand-side entry
  points of TiledLUpp, `xcol_stride`, is a `std::size_t`
  (`TDLS_STRIDED_WIDE`): in SoA, the dimension times the number of
  systems may exceed 2^31. Calls with an int compile unchanged. The
  runtime ArrowheadLUpp solver passes its column strides, block or
  border times the workspace stride, the same way.
- Breaking: TiledLUpp now keeps the row in place by default
  (`relative_pivot_threshold` = 0.1). Fewer rows move, and the pivot
  arrays stay alike across the threads of a warp, whose pivot-driven
  reads then hit the same offsets. The pivots, the factored matrix and
  the last bits of the results may differ from 0.2.0; a threshold of 1
  restores the previous choice, the largest in-tile candidate, bit for
  bit. The new member sits between `oot_first_acceptable` and
  `singular_floor`: a configuration built without designated
  initializers must place it.
- The test suites compile at `-O3` by default in the Release
  configuration, the level of the Release builds that use the library;
  it was `-O1`. `make check` and `spack install --test=root` thus check
  the code that the optimizer transforms the most, for a longer compile
  time. `TDLS_TESTS_OPTIMIZATION=-O1` restores the previous
  flag; a build tree configured before keeps its cached `-O1`, and MSVC
  builds keep their own Release flags.
- The solver suites and the documentation snippets compile with
  `-ffp-contract=off` where the compiler knows the flag. They compare
  code paths bit for bit, which a compiler contracting multiply-adds
  differently in each path (GCC with `-march=native`) would separate in
  the last bits.

### Fixed

- TiledLUppSolverDynamic gave wrong solutions on Blackwell GPUs
  (sm_100 and later) compiled by nvcc when the caller passed a
  dimension known at compile time, single and multiple right-hand
  sides alike (issue #16). The defect is in the optimizer of nvcc for
  these targets: every entry point now hides its dimensions from it
  (`detail::runtime_dimension`, an empty assembly statement in the
  device code of nvcc for `__CUDA_ARCH__` 1000 and later, nothing
  elsewhere), so that the solver is compiled for a runtime dimension,
  as it is meant to be. Other targets and compilers are unchanged.
- The test suites failed when built with GCC and `-march=native`: their
  bitwise comparisons now run without contraction of multiply-adds
  (see Changed).

## [0.2.0] - 2026-10-11

### Added

- CooperativeLUpp: LU factorization with partial pivoting shared by a
  group of threads, one system per group. The arithmetic is the one of
  the small-system kernel of MAGMA 2.10.0, whose license is reproduced
  in the headers, the pivot choice included when
  `relative_pivot_threshold` is 1. Each thread holds `rows_per_thread`
  vectors of the matrix, so that several systems share a warp, and
  every value gives bitwise-identical results. The layout picks the
  vectors: the solver factors the stored matrix read column-major, so
  that a group reads consecutive addresses in both layouts. Column-major
  is the kernel of MAGMA, the threads holding rows. Row-major, the
  default, factors A^T, the threads holding columns, and solves through
  the transposed factors, the mirror of MAGMA's solve: U^T z = b folded
  into the factorization, then L^T w = z, which divides by nothing, one
  barrier per step. Two solvers: one with a compile-time
  dimension, which keeps the rows in registers, and one with a runtime
  dimension, which updates them in place. At equal shape and
  configuration their results are bitwise identical. With one thread
  per system the solver runs sequentially, on CPU as on GPU. The back
  substitution no longer writes a shared entry in the barrier interval
  where every thread reads it, a data race of the original kernel
  outside lockstep execution.
- The barrier of a group of threads is deduced from the compilation
  target when the caller passes none, `tdls::AutoSync`: nothing for
  one thread per system; `__syncwarp` on the lanes of the group on
  NVIDIA GPUs, sm_70 or newer; a wavefront fence on AMD GPUs; a counter
  in the first two elements of the workspace for CPU threads. On GPU,
  the group is identified by its workspace and checked on entry: a
  group whose threads are not together in their warp, or whose
  workspace is private to each thread, stops the program with a
  message, instead of a wrong result or a deadlock. The kernels of
  CUDA, HIP, Kokkos, RAJA, OpenMP and OpenACC offloading get it alike.
  So do SYCL on NVIDIA and AMD devices, AdaptiveCpp in its cuda and
  hip modes included, and the parallel algorithms of nvc++. A caller
  may still pass its own barrier, used as is. `make_sync` returns the
  deduced one for the exchanges of the caller. Some targets have no
  deduced barrier for a group of several threads. There, the
  compilation stops with a message that says why. These targets are
  NVIDIA GPUs before Volta, SPIR-V devices such as Intel GPUs, nvc++
  OpenMP or OpenACC offloading without `-cuda`, the generic mode of
  AdaptiveCpp and GCC offloading. So is a CPU whose standard library
  lacks `std::atomic_ref`, as libc++ before 19. On these targets the
  caller passes its barrier. The runtime solver then needs
  `tdls::NoSync` even for one thread per system. `tdls::NoSync` is
  checked against the size of the group: at compile time by the
  compile-time solver, on entry by the runtime solver, whose group size
  depends on n. The new header `tdls/core/group.hpp` holds the
  barriers, `tdls::NoSync` included.
- Entry points of the new family: factorize, substitute, its in-place
  and canonical forms, solve and solve_inplace, with the stride
  convention of TiledLUpp and, for the compile-time solver, its
  residency booleans. Every entry point runs a fixed sequence of
  barriers, independent of the data.
- Test suites of the new family, which run the groups of threads on
  CPU threads: oracle anchors of both solvers, bitwise bridges between
  them, across rows per thread, entry points, residencies, layouts and
  barriers, singular verdicts, knobs, constexpr certificates and
  compile-time contract checks.
- Documentation of the new family: its page and snippet gallery, a
  TFEL interoperability page, and the three example problems at every
  execution scale, from one thread per system on CPU to groups of
  lanes on GPU and groups of work-items in SYCL.
- Two knobs of the new family. `row_interchange` selects the physical
  row interchanges of LAPACK instead of the logical ones of MAGMA: the
  rows move between the threads, the pivot search publishes one
  candidate per thread, and each column takes two barriers instead of
  three. Both schemes choose the same pivots and run the same
  operations; the factored rows end in the pivoted order.
  `relative_pivot_threshold` keeps the row in place when it reaches a
  fraction of the largest magnitude of its column. Its default 0.1
  saves most row interchanges on matrices that pivot often, up to 30%
  of the solve time on GPU, for a backward error below 1e-13 in double;
  1 gives the pivots of LAPACK and compiles the test out. The default
  `Logical` compiles to the same code as before. The enumeration
  `tdls::RowInterchange` joins the shared vocabulary of the families.
- A NOTICE file reproduces the license of the portions derived from
  MAGMA. The install rules now ship LICENSE and NOTICE under
  `share/doc/tdls`.

### Changed

- The unroll knob of TiledLUppConfig is renamed `unroll_loops`, the name
  every family gives it. Breaking: configurations, and the code
  generated by MFront, must use the new name. Its documentation now
  states that it governs only the loops where the choice can noticeably
  change the performance.
- The out-of-tile threshold of TiledLUppConfig is renamed
  `oot_pivot_threshold`, and so is the solver constant that copies it.
  Breaking: configurations, and the code generated by MFront, must use
  the new name.
- In TiledLUppConfig, `oot_first_acceptable` now follows
  `oot_pivot_threshold`: the knobs of the family come first, then the
  knobs every family shares, `singular_floor`, `unroll_loops` and
  `layout`, in the order of CooperativeLUppConfig. Breaking: designated
  initializers follow the declaration order, so a configuration that
  sets both `singular_floor` and `oot_first_acceptable`, and the code
  generated by MFront, must name `oot_first_acceptable` first.
- TiledLUpp adapts its design to the column-major layout, as
  CooperativeLUpp does to the row-major one. The solver factors the
  stored matrix read row-major, so that tiles are read in address
  order in both layouts. Under the column-major layout, it thus
  factors A^T. The pivoting interchanges columns of A: this is partial
  pivoting of A^T, as stable. The solve runs through the transposed
  factors. First U^T z = b, folded into the factorization by
  solve_inplace. Then L^T w = z, which divides by nothing, and
  x = P^T w. The row-major code is unchanged. Breaking for
  column-major callers: the factored matrix, the pivot array and the
  out-of-tile counter now come from the factorization of A^T. They
  match bit for bit what the row-major solver computes on the same
  array. So does the singular verdict of a matrix near
  `singular_floor`. The solutions solve the same system, within the
  backward error of partial pivoting. They are no longer bitwise the
  former ones.
- The snippet pages of both families open with a getting-started page.
  It shows a first call on CPU, a first call on GPU and the solver of a
  runtime dimension. CooperativeLUpp then adds a page on groups of
  threads, on GPU and on CPU threads.

### Removed

- `TDLS_LAYOUT_INDEX`, the flat-index macro of `solvers/options.hpp`:
  no solver uses it any more.

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

[Unreleased]: https://github.com/trsxvz/TDLS/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/trsxvz/TDLS/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/trsxvz/TDLS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/trsxvz/TDLS/releases/tag/v0.1.0
