# TiledLUpp

LU factorization with logical partial pivoting on a tile grid, one
system per thread. The family has one configuration type and two
solvers: one with a compile-time dimension, one with a runtime
dimension.

## Algorithm

The matrix is split into a grid of TS x TS tiles and Gaussian
elimination runs on tiles instead of scalars. Only the tiles involved
in the current step live in registers; the full matrix stays in remote
memory (shared, global or plain host memory). Pivoting is logical: a
pivot array maps logical rows to physical rows, and rows are
physically swapped only inside the register-resident diagonal tile.

The pivot is searched inside the diagonal tile first. When the best
in-tile candidate falls below `oot_threshold`, the search extends to
the rows under the tile: out-of-tile pivoting. Each candidate is
corrected on the fly for the eliminations it has not received yet.
With `oot_first_acceptable`, the scan stops at the first candidate
reaching the threshold. Below `singular_floor`, the factorization is
declared singular and the entry point returns `false`.

Two elimination schedules exist. Right-looking factors the diagonal
tile and pushes its updates into the trailing matrix. Left-looking
pulls the updates of the prior tiles when a tile is visited, so each
tile of the trailing matrix is written once per factorization. Both
leave the factors in the same layout, so the substitutions do not
depend on the schedule.

The factored matrix holds L and U in place. Its diagonal holds the
reciprocals of the pivots, not the pivots: the factors are consumed by
the substitutions of TDLS, not by LAPACK routines.

## Compile-time and runtime dimension

`TiledLUppSolverStatic<T, N, Config>` takes the dimension N at compile
time. Three residency template booleans, `internal_rhs`,
`internal_piv` and `internal_matrix`, declare where each operand
lives. A caller-local array is indexed at compile time, so that a
whole system can stay in registers. External memory is walked with a
runtime stride. The booleans have no default: each call site states
what its buffers are.

`TiledLUppSolverDynamic<T, Config>` takes the dimension n at run time.
Every operand is a pointer plus a stride; there are no residency
booleans and no unroll pragma. It is the flexibility path: dimensions
unknown at compile time, faster builds.

At equal shape and configuration, the two solvers execute the same
arithmetic and produce bitwise identical results.

## Configuration

`TiledLUppConfig<T>` is an aggregate of seven knobs, passed to the
solvers as a constexpr value. Designated initializers override
individual knobs; `TiledLUppConfig<double>{}` keeps the defaults.

| Knob | Default | Role |
|---|---|---|
| `tile_size` | 3 | extent of the register tiles, the main performance axis; may exceed the dimension |
| `schedule` | `RightLooking` | elimination schedule, `RightLooking` or `LeftLooking` |
| `oot_threshold` | 1e-10, 1e-4 for `float` | an in-tile pivot at least this large is accepted without searching below the tile |
| `singular_floor` | `numeric_limits<T>::min()` | a pivot below it is singular, after an out-of-tile search or in a trailing tile; positive, at most `oot_threshold` |
| `oot_first_acceptable` | `true` | the search below the tile stops at the first candidate reaching `oot_threshold` |
| `unroll_inner` | `true` | forced unrolling of the in-tile loops, the guard that keeps tiles in registers on GPU; ignored by the runtime solver |
| `layout` | `RowMajor` | matrix storage, `RowMajor` or `ColMajor`; results are bitwise identical |

The two thresholds are written as plain `T` values. They are stored
exactly, as `StructuralReal` values, so that a configuration remains a
template argument on every compiler.

Every factorizing entry point has an overload with a trailing
`int& oot_count`. It counts the columns whose best in-tile pivot fell
below `oot_threshold`, the trigger of the out-of-tile search. A weak
pivot of a trailing tile counts too, although no row is left to
search. Without the argument, the counter is compiled out.

The {doc}`snippets <snippets/index>` show every entry point on a small
system, one snippet each.

```{toctree}
:maxdepth: 2
:hidden:

snippets/index
```
