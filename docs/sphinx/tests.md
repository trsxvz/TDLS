# Tests

The test tree is split into spaces, each tagged with a ctest label so
they run independently. `common` validates the shared test
infrastructure, `solvers` the library itself. `examples` runs the
self-checking example programs, `snippets` the snippets of the
documentation.

## Running the tests

```sh
cmake --build build --target buildtests
ctest --test-dir build -L solvers      # one space
ctest --test-dir build -R oracle       # filter suites by name
./build/tests/solvers/tdls_test_tiledlupp_oracle_static N=12
```

The last form runs a single suite directly; the optional argument
keeps only the cases whose name contains it. The suites use a small
self-contained harness (`tests/common/harness.hpp`), no external test
framework.

In the Release configuration the suites compile at `-O3` (MSVC keeps
its own Release flags); `-DTDLS_TESTS_OPTIMIZATION=-O1` builds them
faster with GCC and Clang, for quick local iterations. The solver
suites and the documentation snippets also compile with
`-ffp-contract=off` where the compiler accepts it. They compare code
paths bit for bit, and a compiler that contracts multiplications and
additions differently in two paths (GCC with `-march=native` does)
changes the last bits.

## Method

Each solver family rests on two pillars.

The anchor is an independent oracle: a naive textbook implementation
of the same factorization, written against none of the library
internals. Solver results are compared to it through the normwise
backward error, the conditioning-independent measure of a direct
solve. The tolerances are derived from the pivoting policy of the
configuration, not guessed.

Everything else is bitwise bridging: once one path is anchored, every
other path must reproduce it bit for bit. The runtime variant against
the compile-time one, every residency combination, every batch layout,
every entry point equivalence, every tile size. The dimension grid of
the anchor suites crosses every tile boundary (divisible, off by one,
single tile, tile equal to the dimension), so a bridge failure
localizes a divergence exactly.

Two mechanisms complete the picture. The constexpr suites evaluate
whole solves at compile time, which the standard requires to be free
of undefined behaviour: each one is a certificate for the path it
exercises. The `reject_*` tests compile, on purpose, translation units
that the compile-time contracts must reject. Each passes only when the
compiler emits the exact diagnostic of its contract, so a silently
dropped contract turns the suite red. The `stop_*` tests do the same
for the contracts checked at run time: each program must stop with
the message of its contract.

## Examples as tests

The example programs double as living documentation and as tests:
each one is a self-checking program executed by ctest under the
`examples` label.

Every solver family gets three problems, chosen for what they impose
on the solver.

- Compile-time dimension: the dimension is a property of the model and
  of the method, written in the program.
- Runtime dimension: the dimension is a parameter chosen when the
  computation is launched.
- Compile-time dimension, MFront pattern: the dimension is compile-time
  again, and the program has the shape of an MFront-generated
  behaviour. A Newton iteration on a fresh jacobian, then the tangent
  operator from a factorization of the converged one.

For ArrowheadLUpp, the shape of the matrix stands for the dimension.
Its solvers have no separate factorization: the tangent operator comes
from `solve_border_canonical_multirhs` or
`solve_inplace_streamed_or_tangent`, which the snippets of
{doc}`the family <arrowhead_lupp/index>` show, and the examples of its
MFront pattern run the Newton iteration alone.

Each problem runs at every execution scale: sequential, OpenMP,
parallel STL, SYCL, and CUDA or HIP in two variants, which differ by
the placement of the operands or by their layout in device memory.
Moving from one scale to the next changes the parallel harness and the
placement of the operands, never the physics. The build options are in
{doc}`getting_started`.

## Generic suites

| Suite | What it locks |
|---|---|
| `backward_error` (label `common`) | the metric every anchor rests on: the exact solution sits at the noise floor, wrong and non-finite solutions are flagged |
| `adaptors` | structural detection on mocks mirroring the TFEL shapes, accepted and rejected: the contract behind {doc}`tfel/index` |
| `reject_adaptors_*` | the compile-time contracts of the adaptors: dense matrix argument, const-ness, config scalar match, row-major addressing, extent coherence, pivots included |

## TiledLUpp tests

### Suites

The oracle of the family is a naive LU with physical row swaps
(`tests/common/reference_lu.hpp`).

| Suite | What it locks |
|---|---|
| `oracle_static`, `oracle_dynamic` | backward error of `solve` against the naive LU, on the boundary-crossing dimension grid, in float, double and long double |
| `static_vs_dynamic` | bitwise equality of the two variants |
| `residencies` | every residency combination reproduces the anchored path bitwise |
| `layouts` | AoS, SoA and AoSoA addressing, bitwise |
| `colmajor` | the column-major factorization against the row-major one of the same array, bitwise, and each solution by its backward error. Covered: `solve` and `solve_inplace` of both solvers, both schedules, the internal residency, the `unroll_loops` knob off, a strided arena |
| `entry_points` | the documented entry point equivalences, bitwise |
| `inplace_paths` | the two in-place substitution algorithms around their switchover dimensions, in both layouts |
| `multirhs` | the `_multirhs` entry points against the same columns solved one by one, bitwise, on both solvers and in both layouts; the two stages `forward_inplace_multirhs` then `backward_inplace_multirhs` against the in-place block |
| `tile_sizes` | every tile size against the anchored one, including unit tiles (tile size 1) and tile sizes exceeding the dimension |
| `singular` | singular and near-singular systems, including the tiny-but-solvable counter-case |
| `config_knobs` | each configuration knob changes what it should and nothing else, including the in-tile pivot choice under a relative threshold, which leaves the out-of-tile search to its absolute trigger |
| `constexpr` | compile-time certificates on both solvers |
| `cross_full` | full parameter cross-product on two rich shapes |
| `dynamic_edges` | runtime-size edge cases of the dynamic variant |
| `wide_offsets` | element offsets beyond 2^32: one system spread over tens of gigabytes of a lazily backed arena (POSIX with memory overcommit, not run elsewhere) by strides that take the offsets of the matrix, the pivots, the right-hand side and the columns of a block past 2^32, the stride between the columns past 2^32 by itself, against the same system stored contiguously, bitwise, on both solvers and in both layouts |
| `device_constant_dimension` | the runtime solver called from a kernel with dimensions known at compile time, through every substitution entry point, in both layouts and both layouts of a block of right-hand sides, against the same solves on the host within 1e-12; built with the GPU examples (`TDLS_BUILD_CUDA_EXAMPLES` or `TDLS_BUILD_HIP_EXAMPLES`), skipped without a device |
| `reject_config_*` | the threshold contracts of the configuration: ordering and positivity, and a relative pivot threshold in (0, 1] |

For the detail of any suite, the authoritative description is the
`\file` documentation at the top of its source in
`tests/solvers/tiled_lupp/`.

### Example programs

| | CPU, sequential | CPU, OpenMP | parallel STL | SYCL | CUDA or HIP, thread-local memory | CUDA or HIP, device memory |
|---|---|---|---|---|---|---|
| **Compile-time dimension** (stiff chemistry, Radau IIA, N = 9) | `implicit_ode` | `implicit_ode_batch_omp` | `implicit_ode_batch_stdpar` | `implicit_ode_batch_sycl` | `implicit_ode_batch_gpu` | `implicit_ode_batch_gpu_soa` |
| **Runtime dimension** (Love integral equation, Nystroem, n chosen at launch) | `integral_equation` | `integral_equation_batch_omp` | `integral_equation_batch_stdpar` | `integral_equation_batch_sycl` | `integral_equation_batch_gpu` | `integral_equation_batch_gpu_soa` |
| **Compile-time dimension, MFront pattern** (Norton viscoplasticity, N = 7) | `norton_law` | `norton_law_batch_omp` | `norton_law_batch_stdpar` | `norton_law_batch_sycl` | `norton_law_batch_gpu` | `norton_law_batch_gpu_soa` |

The sources live under `examples/tiled_lupp/`, one directory per
execution scale. The GPU examples are single sources in the common
CUDA/HIP dialect, compiled as CUDA or HIP according to the option
enabled at configure time. The parallel STL examples serve the CPU
cores and, with the offload flags of the compiler, a GPU; the device
targets carry a `_gpu` suffix. The SYCL examples run on whatever
device the default selector picks; without one, they report
themselves skipped.

#### Compile-time dimension

The Robertson stiff kinetics is integrated with a 3-stage Radau IIA
method. The Newton systems have size N = stages x species, fixed by
the method and by the chemical mechanism. N = 9 is a property of the
program, so `TiledLUppSolverStatic` applies. The examples show the
canonical reason for the split factorize/substitute interface, the
frozen Jacobian practice. On GPU, they show the two residency choices:

- `implicit_ode_batch_gpu`: residency booleans set to true, the whole
  system lives in thread registers, device memory only holds the
  per-cell inputs and outputs;
- `implicit_ode_batch_gpu_soa`: residency booleans set to false, the
  batch of systems is materialized in device memory
  structure-of-arrays and walked with the batch stride, every access
  coalesced.

#### Runtime dimension

Love's integral equation, the potential of a parallel-plate
capacitor, is discretized with the Nystroem method. The quadrature
resolution n is an accuracy versus cost knob chosen when the
computation is launched. The dimension is a runtime value, so
`TiledLUppSolverDynamic` applies. The batch versions sweep the plate
separation, one dense system per parameter value. The runtime solver
has no residency booleans. The two GPU variants express the placement
through the strides handed to the solver: unit stride on thread-local
arrays, batch stride on the SoA batch.

Every instance factorizes once and substitutes two right-hand sides:
a manufactured one, which the solve must return to solver accuracy
and serves as the self-check, and the physical unit potential.

#### Compile-time dimension, MFront pattern

Norton viscoplasticity is integrated the way the MFront `Implicit` DSL
does it: implicit Euler, with the elastic strain increment and the
viscoplastic multiplier as unknowns. Newton iterates on a 7 x 7 system
with the analytic jacobian of the MFront tutorial. N = 7 is fixed by
the law and by the modelling hypothesis, so `TiledLUppSolverStatic`
applies. This is the shape of the systems that MFront-generated
behaviours hand to TDLS.

The family shows the entry points the other two do not. `solve_inplace`
serves the Newton corrections, whose jacobian is fresh at every
iteration. `factorize` and `substitute_canonical_multirhs` then give
the consistent tangent operator: the six columns of the inverse
jacobian, solved together on one factorization where MFront runs six
substitutions.

The law needs no tensor library: symmetric tensors are 6-vectors in
the TFEL convention and everything it manipulates is dense. The batch
versions integrate one loading history per integration point,
with amplitudes spanning the elastic and the creep regimes. The
self-checks rest on two independent references. The radial return, the
closed-form solution of the isotropic case, is checked on every point.
Central differences of the integration are checked against the tangent
operator on a sample of points.

## CooperativeLUpp tests

### Suites

The oracle of the family is the naive LU of the TiledLUpp suites
(`tests/common/reference_lu.hpp`). The groups of threads are CPU
threads synchronized by a barrier. They do not run in lockstep, so a
data race of the solver turns into wrong values, which the bridges
catch. The forced unrolling runs on groups of small systems; larger
systems and one thread per system run the no-pragma branch, which
keeps the instrumented builds short.

| Suite | What it locks |
|---|---|
| `oracle_static`, `oracle_dynamic` | backward error of `solve_inplace` against the naive LU, on one thread per system and on groups of threads, in float, double and long double, under both row interchanges and under a relative pivot threshold |
| `static_vs_dynamic` | bitwise equality of the two variants, on every entry point |
| `rows_per_thread` | every mapping of rows to threads against one thread per system, bitwise, from one row per thread to more rows than the dimension, under both row interchanges |
| `row_interchange` | physical row interchanges against logical ones, bitwise, on every entry point: the solutions, and the factored rows once placed |
| `auto_sync` | the deduced barrier against an explicit one, bitwise, on every entry point of both solvers, and the barrier of `make_sync` in the exchanges of the caller; the barrier elements of the workspace back to their state between calls |
| `entry_points` | the documented entry point equivalences, bitwise, and the guarantees on return: every result visible to the whole group, the workspace free; a regression case guards the workspace, which no solver may declare `restrict` |
| `residencies` | every residency combination against the external one, bitwise |
| `wide_offsets` | element offsets beyond 2^32, as in the TiledLUpp suite: matrix, pivots and right-hand side spread by strides past 2^32, against the same system stored contiguously, bitwise, on both solvers and in both layouts |
| `layouts` | the AoS, SoA and AoSoA addressing under each matrix layout, bitwise, on both solvers; A stored row-major and A^T stored column-major, the same array, factored bit for bit the same way, each solution checked by its backward error |
| `singular` | singular and near-singular systems, including the tiny-but-solvable counter-case, with a verdict uniform across the group |
| `config_knobs` | each configuration knob changes what it should and nothing else, including the pivot choice under a relative threshold |
| `constexpr` | compile-time certificates on both solvers, one thread per system, under both row interchanges |
| `reject_*` | the compile-time contracts: positive floor, at least one row per thread, a relative pivot threshold in (0, 1], no `tdls::NoSync` for a group of several threads |
| `stop_*` | the run-time contracts of the barrier: `tdls::NoSync` with the runtime solver on a dimension above `rows_per_thread`, and a workspace that was not zero before a group of CPU threads used it |

For the detail of any suite, the authoritative description is the
`\file` documentation at the top of its source in
`tests/solvers/cooperative_lupp/`.

### Example programs

| | CPU, sequential | CPU, OpenMP | parallel STL | SYCL | CUDA or HIP, registers or shared memory | CUDA or HIP, device memory |
|---|---|---|---|---|---|---|
| **Compile-time dimension** (stiff chemistry, Radau IIA, N = 9) | `implicit_ode` | `implicit_ode_batch_omp` | `implicit_ode_batch_stdpar` | `implicit_ode_batch_sycl` | `implicit_ode_batch_gpu` | `implicit_ode_batch_gpu_soa` |
| **Runtime dimension** (Love integral equation, Nystroem, n chosen at launch) | `integral_equation` | `integral_equation_batch_omp` | `integral_equation_batch_stdpar` | `integral_equation_batch_sycl` | `integral_equation_batch_gpu` | `integral_equation_batch_gpu_soa` |
| **Compile-time dimension, MFront pattern** (Norton viscoplasticity, N = 7) | `norton_law` | `norton_law_batch_omp` | `norton_law_batch_stdpar` | `norton_law_batch_sycl` | `norton_law_batch_gpu` | `norton_law_batch_gpu_soa` |

The sources live under `examples/cooperative_lupp/`, with the layout
and the build options of the TiledLUpp examples. The problems, the
batches and the self-checks are the same. Only the solver and the
mapping of the systems to threads change.

Each scale picks its group of threads:

- sequential, OpenMP and parallel STL: one thread per system, with no
  barrier. `rows_per_thread` covers the dimension. A barrier between
  CPU threads would cost more than a step of the elimination, and
  `par_unseq` forbids any synchronization. One exception,
  `norton_law_batch_stdpar_groups`, built with nvc++ for a GPU only,
  shares each point between two iterations, which nvc++ runs on two
  lanes of a warp; the deduced barrier checks it on entry;
- SYCL: a group of work-items per system and several systems per
  work-group, with the barrier of the work-group. Every SYCL device
  provides it, whatever the size of its sub-groups;
- CUDA or HIP: a group of lanes per system and several groups per
  warp, with the barrier deduced by the solver, which `make_sync`
  returns for the exchanges of the example.

In a group, every thread keeps its own copy of the state of the
problem and computes the whole residual. It builds only the vectors it
holds, columns under the default row-major layout, so the solver reads
them without a barrier. On return, the
solution is visible to the whole group.

Groups that share a barrier wider than themselves, as in a SYCL
work-group, make the same calls. Their Newton loops and their time
loop run while one of them still iterates. A group that has converged
or failed keeps pace without updating its state.

#### Compile-time dimension

The Radau IIA integration of the TiledLUpp examples runs on
`CooperativeLUppSolverStatic`. On GPU and SYCL, three rows per thread
give groups of three threads. On GPU, the two residency choices:

- `implicit_ode_batch_gpu`: residency booleans set to true, each lane
  holds its three rows in registers, the workspaces live in shared
  memory;
- `implicit_ode_batch_gpu_soa`: residency booleans set to false, the
  batch of systems is materialized in device memory
  structure-of-arrays and walked with the batch stride.

#### Runtime dimension

The Love sweep runs on `CooperativeLUppSolverDynamic`, with four rows
per thread: an instance takes ceil(n / 4) threads. The two GPU
variants express the placement through the pointers handed to the
solver:

- `integral_equation_batch_gpu`: each group assembles its system in
  shared memory and solves it at unit stride, one warp per block;
- `integral_equation_batch_gpu_soa`: an assembly kernel writes the
  batch structure-of-arrays, one thread per instance, and a solve
  kernel walks it with the batch stride, one group per instance.

A barrier separates the two substitutions: every thread reads the
manufactured solution before the second one overwrites it.

#### Compile-time dimension, MFront pattern

The Norton integration runs on `CooperativeLUppSolverStatic`. On GPU
and SYCL, four rows per thread give groups of two threads, the
second of which holds a phantom slot. `solve_inplace` serves the Newton
corrections. `factorize` then `substitute_canonical`, once per
column, give the consistent tangent operator, as MFront does.

## ArrowheadLUpp tests

### Suites

The oracle of the family is the naive LU of the other families
(`tests/common/reference_lu.hpp`), on the full matrix: partial
pivoting over whole columns, the border rows among the candidates. The
generated systems (`tests/solvers/arrowhead_lupp/arrowhead_problems.hpp`)
mix dominant and random blocks, singular blocks and pivot-poor ones,
so that the pivot check refuses blocks and the second pass runs in
every batch.

| Suite | What it locks |
|---|---|
| `oracle_static`, `oracle_dynamic` | backward error of `solve` against the naive LU, on a grid of shapes crossing the structure of the solver (blocks of one unknown, a border smaller than the blocks, a border not a multiple of the block, blocks of 12, blocks cut into tiles of 3 and 1, a polycrystal, 1452 unknowns), both couplings and layouts, in float, double and long double; the singular verdict on a structurally singular matrix, and a floor of refused blocks; the same with the blocks in registers (`register_blocks`) |
| `second_pass` | the refused blocks: zero pivots at the start, the middle and the end of a block, two in one block, as many singular blocks as the couplings have rank (invertible), one more (singular, both solvers must say so), every block refused by strong couplings, a subnormal pivot (a border row takes the pivot, or the matrix is singular, as for TiledLUpp); the backward error at the level of the naive LU |
| `static_vs_dynamic` | bitwise equality of the two variants on every path of the condensation: the four storages, both couplings and layouts, passes, the gather, the Schur complement in place, the unroll policy off, the blocks in registers |
| `storages` | the four storages bitwise identical; the index maps bijective, and matching the canonical CSR and BSR index arrays; `valid` accepting the shapes the solvers accept and rejecting the others without overflowing; the runtime solver rejecting before any access the shapes whose workspace passes 2^31 entries |
| `residencies` | every residency combination against the external one, bitwise, the strided gaps of the arenas untouched; the runtime solver at the same strides |
| `layouts` | the AoS, SoA and AoSoA batch addressing against the systems solved one by one, bitwise, on both solvers |
| `entry_points` | `solve` against `solve_inplace` and the counting overloads against the others, bitwise, on both solvers |
| `config_knobs` | the performance knobs keep the arithmetic, bitwise, on systems whose pivot checks stay clear of the threshold: every pass width, the gather, the Schur complement in place, the unroll policy; the refusals grow with `condensation_pivot_threshold`, 0 refusing only the singular blocks; every threshold of the check and of the blocks keeps the backward error below 1e-12, except a check threshold of 0, which keeps the pivot-poor blocks and their growth, bounded at 1e-8 |
| `wide_offsets` | element offsets beyond 2^32, as in the other families, a refused block making the second pass use the workspace; the column strides of the TiledLUpp calls of the runtime solver past 2^32; the column stride of a block of right-hand sides past 2^32, on both solvers |
| `constexpr` | compile-time certificates on both solvers, through every path and the second pass, and their bitwise agreement |
| `multirhs` | every column of `solve_inplace_multirhs` and `solve_multirhs` against `solve_inplace` on it alone, bitwise, the matrix they leave included: both solvers, internal and strided right-hand sides (the gaps untouched), refused blocks, passes, the gather, the Schur complement in place; `solve_border_canonical_multirhs` against the border rows of `solve_inplace` on the unit columns, in value (a zero may differ in sign), the matrix left unchanged bit for bit (the Schur complement in place aside) |
| `streamed` | the streamed entry points against the stored ones, bitwise, on both solvers: `solve_inplace_streamed` against `solve_inplace`, `solve_border_canonical_multirhs_streamed` against `solve_border_canonical_multirhs`, `solve_inplace_streamed_or_tangent` against both, refused blocks included; the runtime solver against the compile-time one; the provider called exactly once per block, in order, and `converged()` once, after the last block; the same with the blocks in registers |
| `reject_*` | the compile-time contracts: a shape with an empty part, a `coupling_row` under the dense coupling, diagonal coupling rows past the border, a Bsr border that is not a multiple of the block, a `condensation_pivot_threshold` out of [0, 1] on both solvers, an `rhs_pass_width` above border + 1, no right-hand side, more unit columns than the border, stored or streamed, and more tangent columns than the border |

For the detail of any suite, the authoritative description is the
`\file` documentation at the top of its source in
`tests/solvers/arrowhead_lupp/`.

### Example programs

| | CPU, sequential | CPU, OpenMP | parallel STL | SYCL | CUDA or HIP, thread-local memory | CUDA or HIP, device memory |
|---|---|---|---|---|---|---|
| **Compile-time shape** (Iwan multisurface plasticity, 24 surfaces, N = 30) | `iwan` | `iwan_batch_omp` | `iwan_batch_stdpar` | `iwan_batch_sycl` | `iwan_batch_gpu` | `iwan_batch_gpu_soa` |
| **Runtime shape** (Berveiller-Zaoui aggregate, grains chosen at launch, 240 by default, n = 1452) | `aggregate` | `aggregate_batch_omp` | `aggregate_batch_stdpar` | `aggregate_batch_sycl` | `aggregate_batch_gpu` (device memory, AoS) | `aggregate_batch_gpu_soa` |
| **Compile-time shape, MFront pattern** (Berveiller-Zaoui polycrystal, 12 grains, N = 84) | `polycrystal` | `polycrystal_batch_omp` | `polycrystal_batch_stdpar` | `polycrystal_batch_sycl` | `polycrystal_batch_gpu` | `polycrystal_batch_gpu_soa` |

The sources live under `examples/arrowhead_lupp/`, with the layout and
the build options of the TiledLUpp examples, and the physics under
`examples/common/` (`iwan.hpp`, `polycrystal.hpp`). The three problems
are implicit integrations of behaviours whose Newton systems are
arrowhead matrices: a few global unknowns, the border, and many local
ones, the blocks. Every Newton iteration assembles the jacobian
straight into the Packed storage, through `ArrowheadStorageMap`, then
solves it in place. Every program reports its Newton iterations and
the blocks the pivot check refused. The examples solve one right-hand
side per Newton iteration; the snippets of the family show the other
entry points.

#### Compile-time shape

The Iwan model of the cyclic behaviour of soils, integrated as the Iwan
behaviour of the MFront gallery does it. The unknowns are the elastic
strain increment (the border, 6 unknowns) and the plastic multiplier of
each of the 24 surfaces (a block of one unknown each). The flow
direction of a surface couples its multiplier to every component of the
elastic strain: the dense coupling. The status of the surfaces comes
from the elastic prediction and is corrected at convergence.

The surfaces are calibrated on a hyperbolic skeleton curve in simple
shear. The model then follows the piecewise linear skeleton exactly
under monotonic loading, and Masing's rule exactly under reversals,
which gives exact references. The sequential program drives a closed
simple shear cycle through the corners of the curve and checks the
stress at every step. In the batches, each point takes its own
deviatoric direction and amplitude, in uniform steps that cross several
corners at once; its stress must reach the skeleton at the peak, its
opposite at the end, and stay along the direction.

The outer surfaces of a soil have hardening moduli far below the Young
modulus, while their couplings are of order one. Their blocks fail the
pivot check, and the second pass of the solver eliminates them with the
border rows: the sequential cycle refuses about 2000 blocks over 531
Newton iterations. On GPU, the two residency choices:

- `iwan_batch_gpu`: residency booleans set to true, the jacobian, the
  workspace and the residual are local arrays of the thread;
- `iwan_batch_gpu_soa`: residency booleans set to false, the operands
  are materialized in device memory structure-of-arrays and walked
  with the batch stride. The 30 x 30 system is small, and the
  thread-local variant is the faster one.

#### Runtime shape

The physics of the polycrystal below, with a number of grains read on
the command line: 240 by default, 1452 unknowns. The shape is a datum of
the computation, so `ArrowheadLUppSolverDynamic` applies, its workspace
and pivots sized by `workspace_size(shape)` and `pivot_size(shape)`.
The sequential program times the loading history, then the first step
again by a dense LU of the whole jacobian (`TiledLUppSolverDynamic` on
the full matrix): on one CPU core, the 20 steps of the history take a
small fraction of the time of the first step alone by the dense LU.
Both first steps must agree.

The OpenMP program allocates the operands once per thread, sized from
the shape. The parallel STL and SYCL programs give every operand of a
point its slice of a batch, structure-of-arrays. On GPU, no thread holds
a local array of runtime size, so both variants keep the operands in
device memory, and differ by their layout:

- `aggregate_batch_gpu`: each point owns a contiguous slice of each
  batch (array of structures), walked with a stride of 1;
- `aggregate_batch_gpu_soa`: structure-of-arrays, walked with the batch
  stride, every access of a warp coalesced.

Both print the time of their kernel. A point holds about 280 KB of
operands: the batches that fit in device memory hold far fewer points
than the device holds threads, and the SoA layout gains little.
The self-checks are those of the polycrystal, the sampled points being
integrated again on the host by the same runtime solver.

#### Compile-time shape, MFront pattern

A Berveiller-Zaoui polycrystal of 12 face-centred cubic grains with
random orientations, each flowing on its 12 slip systems by a Norton
law, integrated by the implicit Euler scheme as the polycrystal
behaviours of MFront do. The unknowns are the macroscopic elastic and
viscoplastic strains (the border, 12 unknowns) and the viscoplastic
strain of each grain (a block of 6 unknowns). A grain meets the border
only through the average that defines the macroscopic viscoplastic
strain: the diagonal coupling, from `coupling_row` 6. The number of
grains belongs to the behaviour, so `ArrowheadLUppSolverStatic`
applies.

The sequential program integrates the loading history twice, by
`ArrowheadLUppSolverStatic` and by a dense LU of the whole jacobian,
and both must agree. At every step, the viscoplastic strain must stay
deviatoric, since the Schmid tensors are, and every grain must
dissipate, since every slip rate has the sign of its resolved shear
stress. The batches check the last step of every point the same way,
and a sample of points against their whole history integrated again
with the dense LU. On GPU, the two residency choices:

- `polycrystal_batch_gpu`: residency booleans set to true, the state,
  the jacobian (1512 entries), the workspace and the residual are local
  arrays of the thread, in its local memory;
- `polycrystal_batch_gpu_soa`: residency booleans set to false, every
  operand is materialized in device memory structure-of-arrays.

Both run at the same speed here: the assembly of the jacobian, 144 slip
systems and as many powers per iteration, outweighs the placement of
the operands.
