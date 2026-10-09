# CooperativeLUpp

LU factorization with partial pivoting shared by a group of threads,
one system per group. The family is derived from a kernel of
[MAGMA](https://github.com/icl-utk-edu/magma/blob/v2.10.0/magmablas/zgesv_batched_small.cu).
It has one configuration type and two solvers: one with a compile-time
dimension, one with a runtime dimension.

## Algorithm

The rows of a system are shared by the threads of a group. Each thread
holds `rows_per_thread` rows. With T threads in the group, thread `tx`
holds rows `tx`, `tx + T`, `tx + 2T` and so on. When the dimension is
not a multiple of `rows_per_thread`, the last slots hold phantom rows,
skipped at compile time. A `rows_per_thread` reaching the dimension
gives one thread per system: the solver then runs sequentially.

The elimination is right-looking, one column per step. At each step,
the threads publish the magnitudes of the column in a workspace. Every
thread finds the same pivot. The owner of the pivot row publishes it,
and every thread eliminates its own rows below it. The substitutions
work the same way, one entry per step.

Pivoting is logical: rows never move. Each thread tracks the position
of its rows in the pivoted order, and the pivot array records it. The
mapping of rows to threads changes nothing in the arithmetic: every
value of `rows_per_thread` produces bitwise identical results.

The factored matrix keeps its physical row order, with L and U in
place. Its diagonal holds the pivots themselves. The factors are
consumed by the substitutions of the family. Below `singular_floor`, a
pivot declares the matrix singular and the entry point returns
`false`.

The arithmetic is the one of the MAGMA kernel: same operations, same
order, same pivot choice. The header of the solver lists the changes:
the mapping of rows to threads, the device-callable entry points, the
pivot output, the singularity criterion, and a data race of the back
substitution, removed.

## Groups of threads

Every thread of the group makes the same call, with the same template
arguments, the same operands and its own rank `tx`. Two arguments come
in addition to the operands:

- the workspace of the group, `workspace_size` elements, in memory
  shared by the group: shared memory on GPU, a plain array on CPU;
- the barrier of the group, `sync`: any callable taking no argument
  that makes the memory writes of each thread visible to all of them.

| Programming model | Group | Barrier |
|---|---|---|
| CUDA | lanes of a warp | `__syncwarp` on the lanes of the group |
| HIP | lanes of a wavefront | a wavefront fence and barrier |
| SYCL | work-items of a work-group | `sycl::group_barrier` on the work-group |
| CPU threads | threads | a thread barrier |
| any model, one thread per system | the thread alone | none, the default `tdls::NoSync` |

The barrier is mandatory as soon as a group has several threads. The
compile-time solver checks it at compile time.

Every entry point runs a fixed sequence of barriers. It depends on the
dimension, on `rows_per_thread` and on the entry point, never on the
data. A singular matrix runs the whole sequence too, and the verdict
comes at the end. A barrier wider than the group, a whole warp or a
work-group, is therefore valid as well, provided every thread of its
scope makes the same calls.

On entry, each thread reads only the entries of its own rows: the rows
a thread builds itself need no barrier before the call. Every entry
point ends with a barrier. On return, all the results are visible to
the whole group, whatever the residency of the operands, and the
workspace is free again.

## Compile-time and runtime dimension

`CooperativeLUppSolverStatic<T, N, Config>` takes the dimension N at
compile time. Each thread keeps its rows in registers. Three residency
template booleans, `internal_rhs`, `internal_piv` and
`internal_matrix`, declare where each operand lives. An internal
operand is the slice of the calling thread: its rows, their pivot
entries, their right-hand-side entries, in a local array. An external
operand is the whole object, reachable by the group and walked with a
runtime stride. The booleans have no default: each call site states
what its buffers are.

`CooperativeLUppSolverDynamic<T, Config>` takes the dimension n at run
time. The rows stay in the matrix and are updated in place. Every
operand is a pointer plus a stride, reachable by the group; there are
no residency booleans and no unroll pragma. The number of threads and
the size of the workspace become functions of n. The barrier cannot be
checked at compile time: with one thread and no barrier, n must not
exceed `rows_per_thread`.

At equal shape and configuration, the two solvers execute the same
arithmetic and produce bitwise identical results.

## Configuration

`CooperativeLUppConfig<T>` is an aggregate of four knobs, passed to
the solvers as a constexpr value. Designated initializers override
individual knobs; `CooperativeLUppConfig<double>{}` keeps the
defaults.

| Knob | Default | Role |
|---|---|---|
| `rows_per_thread` | 1 | rows held by each thread, the main performance axis: a system of dimension N takes `ceil(N / rows_per_thread)` threads; from N on, one thread per system |
| `singular_floor` | `numeric_limits<T>::min()` | a pivot below it is singular; positive |
| `unroll_loops` | `true` | forced unrolling of the loops where offering the choice can noticeably change the performance: those indexing the register rows, whose unrolling keeps them in registers on GPU; the pivot search never carries a pragma; `false` is the choice on CPU; ignored by the runtime solver |
| `layout` | `RowMajor` | matrix storage, `RowMajor` or `ColMajor`; results are bitwise identical |

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

The groups of the snippets are CPU threads. `snippets::run_group`
runs its function on every thread of a group and passes it the rank
of the thread and the barrier of the group. A group of one thread
runs on the calling thread, with no barrier. The formulas mark the
thread that holds each row.

```{toctree}
:maxdepth: 1

snippets/configuration
snippets/compile_time_dimension
snippets/runtime_dimension
snippets/batches_and_layouts
```
