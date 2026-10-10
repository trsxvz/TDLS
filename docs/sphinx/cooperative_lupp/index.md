# CooperativeLUpp

LU factorization with partial pivoting shared by a group of threads,
one system per group. The family is derived from a kernel of
[MAGMA](https://github.com/icl-utk-edu/magma/blob/v2.10.0/magmablas/zgesv_batched_small.cu).
It has one configuration type and two solvers: one with a compile-time
dimension, one with a runtime dimension.

## The layout picks the vectors of a thread

The solver always reads the stored matrix column-major, and factors
that matrix, called S on this page. The threads of a group thus hold
vectors that sit at consecutive addresses: on GPU, the lanes of a group
read one coalesced segment per load. The design of MAGMA rests on this
access pattern.

- Column-major layout (`ColMajor`), the convention of MAGMA: S = A, and
  the threads hold rows of A. The solver is the kernel of MAGMA:
  P A = L U, the forward pass L y = P b folded into the factorization,
  then U x = y.
- Row-major layout (`RowMajor`), the TFEL convention and the default:
  S = A^T, and the threads hold columns of A. The factorization is the
  one of MAGMA, applied to A^T: the pivoting interchanges columns of A,
  which is partial pivoting of A^T, as stable. Since A = U^T L^T P, the
  solve is the mirror of MAGMA's: U^T z = b folded into the
  factorization, then L^T w = z, then x = P^T w. L has a unit
  diagonal, so the backward pass divides by nothing.

The layout is a compile-time knob: the choice costs nothing at run time.
A matrix stored row-major and its transpose stored column-major are the
same array: both configurations factor it the same way, bit for bit,
and solve A x = b and A^T x = b respectively. The rest of this page
speaks of the rows of S, the vectors of a thread.

## Algorithm

The rows of S are shared by the threads of a group. Each thread
holds `rows_per_thread` rows. With T threads in the group, thread `tx`
holds rows `tx`, `tx + T`, `tx + 2T` and so on. Any dimension works
with any `rows_per_thread`. When the dimension is not a multiple of
it, the last slots hold phantom rows: they need no padding of the
matrix, and no operation runs on them. The compile-time solver skips
them at compile time, the runtime solver at run time. A
`rows_per_thread` reaching the dimension gives one thread per system:
the solver then runs sequentially.

The elimination is right-looking, one column per step. At each step,
the threads publish the magnitudes of the column in a workspace. Every
thread finds the same pivot. The owner of the pivot row publishes it,
and every thread eliminates its own rows below it. The substitutions
work the same way, one entry per step.

By default, pivoting is logical: rows never move. Each thread tracks
the position of its rows in the pivoted order, and the pivot array
records it. The factored matrix keeps its physical row order, with L
and U in place.

With `row_interchange = Physical`, rows move between the threads, as
in LAPACK. The row at position k always lives in the same slot of the
same thread, so the owner of each pivot row is known in advance. Each
thread publishes a single candidate of the column instead of one
magnitude per row, and each step of the factorization takes two
barriers instead of three. In exchange, a column whose pivot is not in
place moves two whole rows through the workspace. Which scheme is
faster depends on the device and on how often the matrices pivot. The
factored rows end in the pivoted order, and the pivot array maps each
position to the original index of its row. The workspace size differs.

Both schemes choose the same pivots and run the same operations in the
same order. On a matrix they do not declare singular, their solutions
are bitwise identical when no multiply-add is fused. GPU compilers fuse
them by default, possibly differently in the two schemes: the solutions
may then differ in the last bits. The mapping of rows to threads
changes nothing in the arithmetic either: every value of
`rows_per_thread` produces bitwise identical results.

The pivot of a column follows a relative threshold,
`relative_pivot_threshold`: the row in place keeps the pivot when it
reaches that fraction of the largest magnitude of the column, and the
largest magnitude wins otherwise. Rows then move less often, and the
multipliers stay bounded by the inverse of the threshold. The default
0.1 saves most row interchanges on matrices that pivot often, up to 30%
of the solve time on GPU, for a backward error below 1e-13 in double.
A threshold of 1 is the partial pivoting of LAPACK, with its pivots: on
matrices that never interchange, such as Newton jacobians close to the
identity, it saves the few percent that the test costs. The threshold
is relative to the column, unlike the absolute `oot_pivot_threshold` of
TiledLUpp.

The diagonal of the factored matrix holds the pivots themselves. The
factors are consumed by the substitutions of the family, under the
same configuration. Below `singular_floor`, a pivot declares the
matrix singular and the entry point returns `false`.

The operations are those of the MAGMA kernel applied to S, in the
same order. So is the pivot choice with `relative_pivot_threshold = 1`;
a smaller threshold departs from it on purpose, keeping the row in
place more often, so the pivots and the results then differ. Under the
row-major layout, the substitutions are the mirror of MAGMA's, with one
barrier per step instead of two. The header of the solver lists the
other changes: the mapping of rows to threads, the device-callable
entry points, the pivot output, the singularity criterion, and a data
race of the back substitution, removed.

The portions derived from MAGMA remain subject to its license, BSD
3-Clause. The license is reproduced at the top of each solver header
and in the NOTICE file, installed with the library.

## Groups of threads

Every thread of the group makes the same call, with the same template
arguments, the same operands and its own rank `tx`. The workspace of
the group comes in addition to the operands: `workspace_size`
elements, in memory shared by the group, shared memory on GPU, a plain
array on CPU.

### The deduced barrier

The last argument, `sync`, is the barrier of the group. It is
optional, and most code passes nothing: by default, `tdls::AutoSync`,
the solver deduces the barrier from the target the code is compiled
for.

| Target | Deduced barrier |
|---|---|
| any target, one thread per system | none |
| NVIDIA GPU, sm_70 or newer | `__syncwarp` on the lanes of the group |
| AMD GPU | a wavefront fence: the lanes of a wavefront run in lockstep |
| CPU threads | a counter in the first two elements of the workspace |

On GPU, this covers every model whose kernels the compiler builds as
GPU code:

- CUDA and HIP;
- Kokkos and RAJA;
- OpenMP offloading with clang;
- nvc++ with `-stdpar=gpu`, or with `-cuda` for OpenMP and OpenACC
  offloading;
- SYCL on NVIDIA and AMD devices, AdaptiveCpp in its cuda and hip
  modes included.

The lanes of a group may sit anywhere in their warp or wavefront. On
entry, the active lanes match the address of their workspace, and the
group must find all its threads among them. A group that does not
stops the program with a message, before any exchange. So does a
workspace in the private memory of each thread, whose address is the
same in every lane: the group shares its workspace, in shared or
global memory. Neither a badly placed group nor a private workspace
gives a wrong result or a deadlock.
The threads of a group must therefore call the solver together, from
the same path of the code, as they do when they solve the same system.
The check runs once per call, and the barrier is the one a caller
would write by hand.

On CPU, the threads of a group are any threads that run concurrently:
OpenMP, `std::thread`, Kokkos team threads. The first two elements of
the workspace must be zero before the first call, as in a
`std::vector`, and the barrier keeps them so between calls. They keep
their place whatever the dimension and the configuration, so a
workspace may serve other solves. A group that waits for more than a
second prints a diagnostic once and keeps waiting: the threads of a
group must not run one after the other, as in a worksharing loop or in
the parallel algorithms on CPU. A barrier between CPU threads costs
more than a step of the solve: one thread per system stays the fast
choice on CPU.

### When to pass a barrier

Never for speed: the deduced barrier is the one a caller would write
by hand, and `make_sync` lets a caller run its check on entry once for
several calls (see below). A barrier is needed in two cases only.

**The compiler asks for it.** On a few targets, a group of several
threads has no deduced barrier, and the compilation stops with a
message that says why:

- nvc++ OpenMP or OpenACC offloading without `-cuda`, the OpenACC
  backend of nvc++ for the parallel algorithms, the generic mode of
  AdaptiveCpp and GCC offloading: they offer no warp instruction
  callable without a handle of the kernel;
- NVIDIA GPUs before Volta (sm_70): they lack the instruction that
  matches the lanes;
- SPIR-V devices, Intel GPUs among them, through SYCL or OpenMP
  offloading: they have the instructions, but no standard way to stop
  a kernel that finds its group incomplete, so the check could not
  keep its promise there;
- a CPU whose standard library lacks `std::atomic_ref`, as libc++
  before 19.

On these targets, the runtime solver needs an explicit choice even for
one thread per system: its group size depends on n, unknown to the
compiler. `tdls::NoSync` states that choice, and the solver checks it
on entry (see what TDLS checks, below).

**The group is not one the deduced barrier knows.** The deduced
barrier knows the lanes of one warp or wavefront on GPU, and threads
running concurrently on CPU. Any other group needs the barrier of its
model:

- a group larger than a warp, 32 lanes on NVIDIA or 64 on AMD, which
  the compiler refuses: a block barrier (`__syncthreads`), a
  work-group barrier or a team barrier;
- threads that a runtime runs one after the other, such as the
  work-items of a SYCL kernel on CPU: a work-group barrier. The
  compiler cannot tell this case: the deduced barrier waits, and
  prints a message after a second.

A barrier is any callable taking no argument, called by every thread
of the group, that makes the memory writes of each thread visible to
all of them. A barrier wider than the group is valid too, as the
sequence of barriers below explains. The solver uses it as is,
without any check. `tdls::NoSync`, no barrier at all, serves a group
of one thread; the compile-time solver refuses it for several
threads.

| Programming model | Group | Explicit barrier |
|---|---|---|
| CUDA | lanes of a warp | `__syncwarp` on the lanes of the group |
| HIP | lanes of a wavefront | a wavefront fence and barrier |
| CUDA, HIP | threads of a block | `__syncthreads` |
| SYCL | work-items of a sub-group or a work-group | `sycl::group_barrier` |
| Kokkos | threads of a team | `team.team_barrier()` |
| CPU threads | threads | a thread barrier |

`make_sync(work)`, or `make_sync(n, work)` for the runtime solver,
returns the deduced barrier, checked once. A caller that also needs it
for its own exchanges between the threads of the group builds it once,
calls it, and passes it to the entry points, which then use it as is.

### What TDLS checks

Every misuse that TDLS can tell from a correct use is reported, at
compile time when possible:

- at compile time: a target without a deduced barrier, `tdls::NoSync`
  for a group of several threads of the compile-time solver, a group
  larger than a warp;
- on GPU, with the deduced barrier: a group whose threads are not
  together in their warp, and a workspace in private memory, stop the
  program with a message before any exchange;
- on CPU, with the deduced barrier: a workspace whose first two
  elements hold no barrier state stops the program with a message, and
  a group that waits for more than a second prints one;
- `tdls::NoSync` with the runtime solver on a dimension above
  `rows_per_thread`, a group of several threads, stops the program with
  a message. The targets that cannot stop a kernel let it run: SPIR-V
  devices, the generic mode of AdaptiveCpp, nvc++ offloading without
  `-cuda` and GCC offloading.

Three misuses stay silent, since nothing tells them from a correct
use, and may give wrong results or a deadlock:

- an explicit barrier that does not synchronize the group, since the
  solver uses it as is;
- two groups that share one workspace, except on GPU when their lanes
  share a warp;
- on CPU, a workspace whose first two elements were left with values
  that look like a barrier state.

### The sequence of barriers

Every entry point runs a fixed sequence of barriers. It depends on the
dimension, on `rows_per_thread`, on the row interchanges and on the
entry point, never on the data. A singular matrix runs the whole
sequence too, and the verdict comes at the end. A barrier wider than
the group, a whole warp or a work-group, is therefore valid as well,
provided every thread of its scope makes the same calls.

On entry, each thread reads only the entries of its own rows of S and
of the right-hand side: under the row-major layout, its columns of A.
The vectors a thread builds itself need no barrier before the call. Every entry
point ends with a barrier. On return, all the results are visible to
the whole group, whatever the residency of the operands, and the
workspace is free again.

## Compile-time and runtime dimension

`CooperativeLUppSolverStatic<T, N, Config>` takes the dimension N at
compile time. Each thread keeps its rows in registers. Three residency
template booleans, `internal_rhs`, `internal_piv` and
`internal_matrix`, declare where each operand lives. An internal
operand is the slice of the calling thread: its rows of S, their pivot
entries, their right-hand-side entries, in a local array. The matrix
slice holds the vectors of the thread in the chosen layout: the
N x `rows_per_thread` block of its columns, row-major, or the
`rows_per_thread` x N block of its rows, column-major. With one thread
per system, it is the whole matrix in that layout. An external
operand is the whole object, reachable by the group and walked with a
runtime stride. The booleans have no default: each call site states
what its buffers are.

`CooperativeLUppSolverDynamic<T, Config>` takes the dimension n at run
time. The rows stay in the matrix and are updated in place. Every
operand is a pointer plus a stride, reachable by the group; there are
no residency booleans and no unroll pragma. The number of threads and
the size of the workspace become functions of n. So does the size of
the group, which the deduced barrier checks at run time. With
`tdls::NoSync`, n must not exceed `rows_per_thread`.

At equal shape and configuration, the two solvers execute the same
arithmetic and produce bitwise identical results.

## Configuration

`CooperativeLUppConfig<T>` is an aggregate of six knobs, passed to
the solvers as a constexpr value. Designated initializers override
individual knobs; `CooperativeLUppConfig<double>{}` keeps the
defaults.

| Knob | Default | Role |
|---|---|---|
| `rows_per_thread` | 1 | vectors held by each thread, rows of S: columns of A row-major, rows column-major; the main performance axis: a system of dimension N takes `ceil(N / rows_per_thread)` threads; from N on, one thread per system |
| `row_interchange` | `Logical` | row interchanges, `Logical` (rows never move, as in MAGMA) or `Physical` (rows move between the threads, as in LAPACK); same pivots and operations, different factored formats and workspace sizes |
| `relative_pivot_threshold` | 0.1 | the row in place keeps the pivot when it reaches this fraction of the largest magnitude of its column; 1 is the partial pivoting of LAPACK; in (0, 1] |
| `singular_floor` | `numeric_limits<T>::min()` | a pivot below it is singular; positive |
| `unroll_loops` | `true` | forced unrolling of the loops where offering the choice can noticeably change the performance: those indexing the register rows, whose unrolling keeps them in registers on GPU; the pivot search never carries a pragma; `false` is the choice on CPU; ignored by the runtime solver |
| `layout` | `RowMajor` | matrix storage, which picks the vectors of a thread: `RowMajor`, the threads hold columns and the solve runs through the factors of A^T; `ColMajor`, the threads hold rows, the kernel of MAGMA; in both, a group reads consecutive addresses |

The default `rows_per_thread` is the mapping of MAGMA, one row per
thread. A larger value packs more systems in a warp and leaves fewer
lanes idle, at the price of more registers per thread.

## Snippets

One snippet per entry point or knob. Each one is a complete program:
the system it solves, then the code between the two markers of its
source under `docs/snippets/cooperative_lupp/`. ctest runs them all
under the `snippets` label.

Above the shown code, the program declares what the formula displays:
the matrix as a row-major `double` array, the right-hand sides, and
the expected solution. Below the shown code, it checks its result.

The pages go from the simplest call to the groups of threads, the
knobs and the batches, CPU first, then GPU where it applies. Until the
groups page, one thread solves each system, as with TiledLUpp. The
groups page then starts groups of CPU threads by hand, and the next
pages with `snippets::run_group`, the same code in a function. When a
snippet uses a group, $t_i$ next to the matrix marks the thread that
holds the row.

```{toctree}
:maxdepth: 1

snippets/getting_started
snippets/compile_time_dimension
snippets/runtime_dimension
snippets/groups
snippets/configuration
snippets/batches_and_layouts
```
