# ArrowheadLUpp

Static condensation of arrowhead systems, built on TiledLUpp, one
system per thread. An arrowhead matrix, or bordered block-diagonal
matrix, has a border of unknowns coupled to everything, then diagonal
blocks that never couple to each other. The family has one
configuration type and two solvers: one with a compile-time shape, one
with a runtime shape.

$$
A = \left(\begin{array}{c|cccc}
A_{bb} & A_{b0} & A_{b1} & \cdots & A_{b,N_B-1} \\ \hline
A_{0b} & D_0 & & & \\
A_{1b} & & D_1 & & \\
\vdots & & & \ddots & \\
A_{N_B-1,b} & & & & D_{N_B-1}
\end{array}\right)
$$

The border holds $B$ unknowns, ordered first, and each of the $N_B$
blocks $D_k$ holds $S$ unknowns: $n = B + N_B S$. Such systems come out
of the MFront behaviours whose unknowns split into a few global ones
and many local ones. In the implicit integration of a polycrystal, each
phase owns a block and the macroscopic quantities form the border. In a
multisurface plasticity, such as the Iwan model of TFEL, each surface
owns a plastic multiplier, a block of one unknown, coupled to the
elastic strain, the border.

## Algorithm

The block unknowns are eliminated first, one block at a time. What
remains is the border system, $B \times B$, whose matrix and right-hand
side are the Schur complements

$$
\hat A_{bb} = A_{bb} - \sum_k A_{bk} D_k^{-1} A_{kb}, \quad
\hat y_b = y_b - \sum_k A_{bk} D_k^{-1} y_k
$$

Once the border unknowns $x_b$ are known, each block gives its own:
$x_k = D_k^{-1} (y_k - A_{kb} x_b)$. This is static condensation. Its
cost grows linearly with the number of blocks, where a dense LU of the
whole matrix grows with $n^3$. On one CPU core, a border of 12 and
blocks of 6 take about 0.19 µs per block, from 30 blocks to 9600
($n = 57612$).

**TiledLUpp does the dense work.** Each block is copied into a local
array, in the order of its storage, and factored by a
`TiledLUppSolverStatic` of dimension $S$, in registers. The columns
$[A_{kb} \mid y_k]$ are then solved by the multi-right-hand-side
substitution of the same solver, in one pass or in passes of
`rhs_pass_width` columns, and folded into the Schur complement. The
border system is solved by a `TiledLUppSolverStatic` of dimension $B$.
The runtime solver uses `TiledLUppSolverDynamic` the same way. The
knobs `block_tile_size`, `block_relative_pivot_threshold`,
`block_oot_pivot_threshold` and `border_tile_size` configure these
TiledLUpp solvers.

**The pivot check.** A block eliminated alone chooses its pivots among
its own rows. The partial pivoting of the whole matrix would also
consider the border rows that share the column. So each block is
checked on its factors: every multiplier of its elimination, for its
own rows ($L$) and for the border rows ($M = A_{bk} U^{-1}$), must stay
below `1 / condensation_pivot_threshold` in magnitude, 100 by default.
A block that fails, a singular block among others, is refused and left
untouched. This is a delayed pivot, in the words of the multifrontal
solvers.

**The second pass.** When blocks were refused, they are eliminated
after the others, together with the border rows, in a front held by
the workspace of the caller. The pivots are chosen over the $S$ rows of
the block and the $B$ border rows, a row of the block preferred when it
reaches `condensation_pivot_threshold` times the largest candidate. A
border row may take a pivot; a row of the block then takes its place in
the border, which keeps $B$ rows. A refused block never makes the solve
fail: the solver succeeds whenever the whole matrix is invertible,
rounding aside, and returns `false` on a singular matrix. The
workspace is touched by this pass only.

**The recovery.** The border system solved, the blocks are recovered
from the data the condensation left in place of $A_{kb}$ (and of $D_k$
for a refused block), the last block first.

The solver reads the entries of the pattern only: the zeros outside the
blocks are never read and need not be stored (see the storages below).
Like TiledLUpp, it does not detect non-finite entries: they fail the
pivot check and spread to the solution through the second pass. On
exit, y holds the solution and A data of the solve.

## The layout picks the condensation path

Both layouts read the matrix in the order of its storage, and each one
condenses a block the way that suits it. The layout is a compile-time
knob: the choice costs nothing at run time.

- Row-major layout (`RowMajor`), the TFEL convention and the default.
  TiledLUpp factors each block as stored: $P D_k = L U$. The check
  computes the border multipliers $M = A_{bk} U^{-1}$ apart. Under the
  dense coupling, the Schur complement is then updated by $M V_k$, with
  $V_k = L^{-1} P [A_{kb} \mid y_k]$: the forward stage of the solve
  alone, `forward_inplace_multirhs` of TiledLUpp, since $M$ comes from
  the check. The backward stage, on one column, moves to the recovery.
  Under the diagonal coupling, the whole solve
  $W_k = D_k^{-1} [A_{kb} \mid y_k]$ costs less, folded through the few
  coupling entries.
- Column-major layout (`ColMajor`). TiledLUpp factors $D_k^T$, its
  column-major mode. The forward stage $U^{-T} [A_{kb} \mid y_k]$ of the
  solve then holds the border multipliers of the check, which costs
  nothing more: $[A_{kb} \mid y_k]$ is loaded once, checked on the
  forward stage before anything is written, and completed by the
  backward stage, `backward_inplace_multirhs`. The check is the one of
  the elimination of $A^T$, as stable: a block may pass it in one layout
  and fail it in the other.

## Storages and couplings

Four storages hold the entries of the pattern, the zeros inside the
stored blocks included. Each one reads by rows under the row-major
layout and by columns under the column-major one.

| Storage | Entries | Contents |
|---|---|---|
| `Full` | $n^2$ | the whole matrix, zeros included; only the pattern is read |
| `Packed` | $B^2 + C + N_B S (B + S)$ | $A_{bb}$, then the couplings, then the $A_{kb}$, then the $D_k$, without index arrays |
| `Csr` | $B^2 + C + N_B S (B + S)$ | compressed sparse rows (columns under `ColMajor`), border rows first |
| `Bsr` | whole blocks of $S \times S$ | block sparse rows (columns); the border a multiple of $S$ |

$C$ counts the couplings: $N_B S$ under the diagonal coupling, $N_B B S$
under the dense one.

The couplings $A_{bk}$ take one of two patterns:

- `Dense`, the default: every border row may touch every unknown of
  every block.
- `Diagonal`: unknown $i$ of a block touches border row
  `coupling_row + i` only, one entry per block row. This is the pattern
  of a polycrystal, whose macroscopic strain averages the strains of
  the phases: the solver then reads and updates $S$ border rows only.

The positions follow from the shape: the solver never reads an index
array. `ArrowheadStorageMap` computes the index of every entry, so that
the caller fills its matrix through it. For Csr and Bsr it writes the
canonical index arrays, columns in increasing order and the zeros of
the blocks included: a matrix assembled in these formats must follow
them. Packed reads the fewest entries; on GPU the four storages run at
the same speed, and on CPU Packed is the fastest, by up to 11 %.

## Compile-time and runtime shape

`ArrowheadShape` holds the four integers of the pattern: `border`,
`block`, `blocks`, and `coupling_row` under the diagonal coupling.

`ArrowheadLUppSolverStatic<T, Shape, Config>` takes the shape at compile
time. The copy of a block, its solved columns and the accumulated rows
of the Schur complement are local arrays of compile-time size, kept in
registers where they fit. Three residency template booleans,
`internal_rhs`, `internal_work` and `internal_matrix`, declare where
each operand lives, as in TiledLUpp. The workspace holds
`workspace_size` entries.

`ArrowheadLUppSolverDynamic<T, Config>` takes the shape at run time, as
the first argument of its entry points. A device has no local array of
runtime size: every temporary lives in the workspace, of
`workspace_size(shape)` entries, and the pivots of the TiledLUpp solves
in an int array of `pivot_size(shape)` entries. Every operand is a
pointer plus a stride; there are no residency booleans and no unroll
pragma. It is the flexibility path: shapes unknown at compile time,
faster builds.

At equal shape and configuration, the two solvers execute the same
arithmetic and produce bitwise identical results.

## Several right-hand sides

`solve_inplace_multirhs` and `solve_multirhs` solve `nrhs` right-hand
sides through one condensation: each block solves its coupling columns
and its `nrhs` right-hand-side columns together, $[A_{kb} \mid Y_k]$,
and the border system `nrhs` columns. Column m comes out bit for bit as
`solve_inplace` gives it alone. It starts at m `xcol_stride` (m times
the dimension under the internal residency). The compile-time solver
takes `nrhs` as a template argument, the runtime solver as an argument.
The workspace grows with it: `multirhs_workspace_size<nrhs>` entries
for the compile-time solver (`workspace_size` is
`multirhs_workspace_size<1>`), `workspace_size(shape, nrhs)` for the
runtime one.

`solve_border_canonical_multirhs` returns columns `col0` to
`col0 + nrhs - 1` of the border block of $A^{-1}$, the inverse of the
Schur complement of the border: the border rows of the solutions for
the unit columns of the border, `border` entries per column. These
right-hand sides vanish on the blocks, which are condensed without
right-hand side, and nothing is recovered. It is the tangent operator
of a behaviour loaded through the border equations only, the
macroscopic strain of a polycrystal for instance. `nrhs` is in
[1, `border`].

Six right-hand sides cost 1.2 to 2.9 times one, against 6 for six
solves; six unit columns of the border cost 0.65 to 1.4 times one
solve (polycrystals of 30 and 240 grains, dense couplings, Iwan
plasticity; both solvers, gcc and clang on CPU, an RTX 5080).

## Streamed blocks

The entry points above read a stored matrix. When the blocks come out of
a computation, the jacobian of a behaviour assembled block by block for
instance, storing them costs a write and a read of the whole matrix.
The streamed entry points take them instead from a provider, when the
solver condenses them:

| Streamed entry point | Stored counterpart |
|---|---|
| `solve_inplace_streamed` | `solve_inplace` |
| `solve_border_canonical_multirhs_streamed` | `solve_border_canonical_multirhs` |
| `solve_inplace_streamed_or_tangent` | `solve_inplace`, or `solve_border_canonical_multirhs` once converged |

The contract of the provider:

- The caller writes beforehand only the border: $A_{bb}$ in the storage
  and the border rows of y.
- `provider(k, blk)` is called exactly once per block, for k = 0, 1,
  ... in order. It writes block k through the accessors of `blk`:
  `d(i, j)` for $D_k$, `akb(i, c)` for $A_{kb}$, `coupling(i)` (diagonal
  coupling) or `abk(r, i)` (dense coupling) for $A_{bk}$, `y(i)` for the
  rows of the right-hand side on the block (not read by the canonical
  solve). The entries go in logical order, whatever the layout and the
  storage. Called once per block and in order, the provider may
  accumulate the residual as it goes.
- `solve_inplace_streamed_or_tangent` calls `converged()` once, after
  the last block. When it returns true, the border system the pass has
  just condensed gives the tangent instead of the correction: the
  blocks are not asked for again. y then holds intermediate values (the
  condensation has already written the recovery data of the right-hand
  side), not a correction.
- One right-hand side at most rides through the blocks.
- The entry points are `noexcept`: the provider and `converged()` must
  not throw.

A refused block is stored as the provider gave it, for the second pass.
The arithmetic, the pivot check and the second pass are those of the
stored solve: the results are bit for bit the same. A compiler that
fuses multiplications and additions differently in the two code paths
(gcc with `-march=native` does) changes the last bits; build with
`-ffp-contract=off` to compare them bit for bit.

`StreamedBlock` of the compile-time solver is made of the working arrays
of the solver themselves: the factorization and the solve run in place,
without a copy. The one of the runtime solver is a view of the
workspace, with the same accessors, so that a generic lambda serves
both solvers; its streamed entry points need
`streamed_workspace_size(shape)` entries, or
`streamed_workspace_size(shape, nrhs)` for the canonical columns and
the tangent.

The two paths coexist: the stored entry points do not change. The
streamed path is chosen at compile time (`if constexpr`), and the code
of the stored path comes out the same, instruction for instruction
(gcc 13 and clang 20 on CPU, nvcc 13 on GPU, against the solver without
the streamed entry points).

On GPU, the provider is compiled inside the loop of the solver: a call
the compiler leaves out of line in it passes its arguments through the
local memory, which can cost more than the stored matrix saves. Check
the inlining of the provider before comparing the two paths, and try
`streamed_out_of_line`: kept apart from its caller, the streamed solve
gets its own register allocation.

## Configuration

`ArrowheadLUppConfig<T>` is an aggregate of thirteen knobs, passed to the
solvers as a constexpr value. Designated initializers override
individual knobs; `ArrowheadLUppConfig<double>{}` keeps the defaults.

| Knob | Default | Role |
|---|---|---|
| `coupling` | `Dense` | pattern of the couplings $A_{bk}$, `Dense` or `Diagonal` |
| `storage` | `Full` | storage of the matrix, `Full`, `Packed`, `Csr` or `Bsr` |
| `layout` | `RowMajor` | order of the storage, `RowMajor` or `ColMajor`; picks the condensation path (see above) |
| `condensation_pivot_threshold` | 0.01 | a block passes when no multiplier of its elimination exceeds the inverse; 0 refuses only the singular blocks; in [0, 1] |
| `block_tile_size` | 3 | tile size of TiledLUpp on the blocks |
| `block_relative_pivot_threshold` | 0.01 | `relative_pivot_threshold` of TiledLUpp on the blocks: the rows of a block stay in place, and the reads of a warp stay coalesced |
| `block_oot_pivot_threshold` | 1e-12 for `double` and `long double`<br>1e-6 for `float` | `oot_pivot_threshold` of TiledLUpp on the blocks, two orders below its default |
| `border_tile_size` | 3 | tile size of TiledLUpp on the border system |
| `rhs_pass_width` | 0 | columns $[A_{kb} \mid y_k]$ solved per pass, in [0, border + 1]; 0 solves them all at once, the best on CPU; 2 is 18 to 28 % faster on GPU under the row-major layout with a border of 12, for the stored path; the streamed path, whose block is local, can prefer 0 there (8 % faster on a polycrystal of 240 grains) |
| `gather_rhs` | `false` | permutes the columns of a block by a gather into a second buffer instead of cycles in place; changes the register allocation, to measure per shape on GPU |
| `schur_in_place` | `false` | accumulates the Schur complement in the border entries of A and y instead of a local array; fewer registers, more memory traffic |
| `streamed_out_of_line` | `false` | keeps the streamed entry points of the compile-time solver out of line on GPU (CUDA, HIP, SYCL), where the inliner may merge them into a large caller and allocate its registers worse; 19 % faster on a polycrystal of 240 grains (RTX 5080), to measure per provider; ignored elsewhere and by the runtime solver |
| `unroll_loops` | `true` | forced unrolling of the loops over the entries of a block and of the border, passed on to the TiledLUpp solvers; ignored by the runtime solver |

The floating-point knobs are written as plain `T` values. They are stored
exactly, as `StructuralReal` values, so that a configuration remains a
template argument on every compiler.

Every entry point has an overload with a trailing `int& refused_count`,
the number of blocks the pivot check refused, which the second pass
eliminated.

## Snippets

One snippet per entry point, storage or knob. Each one is a complete
program: the system it solves, then the code between the two markers of
its source under `docs/snippets/arrowhead_lupp/`. ctest runs them all
under the `snippets` label.

They solve the same small system, a border of 2 unknowns and two blocks
of 2, whose lines in the formulas separate the border from the blocks:

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5
\end{array}\right), \quad
b = \begin{pmatrix} 21 \\ 25 \\ 20 \\ 23 \\ 46 \\ 37 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

Its couplings are diagonal: unknown $i$ of each block touches border
row $i$ only. The matrix, its variants and the batch of the batch pages
are declared in `docs/snippets/common/arrow.hpp`, with `store`, which
writes a full matrix in any storage through the map.

The pages go from the simplest call to the storages, the knobs and the
batches, CPU first, then GPU where it applies.

```{toctree}
:maxdepth: 1

snippets/getting_started
snippets/compile_time_shape
snippets/runtime_shape
snippets/storages
snippets/configuration
snippets/batches_and_layouts
snippets/streamed_blocks
```
