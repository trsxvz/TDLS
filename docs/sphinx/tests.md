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
dropped contract turns the suite red.

## Example programs

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

Each problem is declined on every execution scale: sequential, OpenMP,
parallel STL, SYCL, and CUDA or HIP with the systems in registers or
in device memory. Moving from one scale to the next changes the
parallel harness and the placement of the operands, never the
physics. The build options are in {doc}`getting_started`.

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
| `colmajor` | the column-major layout against the row-major one on transposed storage, bitwise, on both solvers |
| `entry_points` | the documented entry point equivalences, bitwise |
| `inplace_paths` | the two in-place substitution algorithms around their switchover dimensions |
| `multirhs` | the `_multirhs` entry points against the same columns solved one by one, bitwise, on both solvers |
| `tile_sizes` | every tile size against the anchored one, including unit tiles (tile size 1) and tile sizes exceeding the dimension |
| `singular` | singular and near-singular systems, including the tiny-but-solvable counter-case |
| `config_knobs` | each configuration knob changes what it should and nothing else |
| `constexpr` | compile-time certificates on both solvers |
| `cross_full` | full parameter cross-product on two rich shapes |
| `dynamic_edges` | runtime-size edge cases of the dynamic variant |
| `reject_config_*` | the threshold contracts of the configuration: ordering and positivity |

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
