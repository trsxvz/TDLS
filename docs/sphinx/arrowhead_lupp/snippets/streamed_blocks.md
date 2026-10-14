# Streamed blocks

The streamed entry points take the blocks from the caller, one at a
time, when the solver condenses them: a provider, `provider(k, blk)`,
writes block k through the accessors of `blk`, and the caller never
writes the whole matrix. The storage keeps its full size: the caller
stores beforehand only the border, $A_{bb}$ and the border rows of y,
and the solver writes there the recovery data and the refused blocks.
See {doc}`../index` for the contract of the provider. The system is
the 6 x 6 one of the other pages, its couplings diagonal, its storage
Packed.

## Solve in place, the blocks streamed

The provider writes $D_k$, $A_{kb}$, the coupling of each unknown of the
block (`coupling`, here; `abk(r, i)` under the dense coupling) and the
rows of the right-hand side on the block. The result is bit for bit the
one of `solve_inplace` on the stored matrix, the arithmetic being the
same (see {doc}`../index` for the fusion of multiplications and
additions).

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
b = \begin{pmatrix} 21 \\ 25 \\ 20 \\ 23 \\ 46 \\ 37 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_streamed.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The border block of the inverse, the blocks streamed

`solve_border_canonical_multirhs_streamed` gives columns of the border
block of $A^{-1}$ as `solve_border_canonical_multirhs` does, bit for bit
(the same arithmetic; see {doc}`../index` for the fusion of
multiplications and additions). The right-hand sides are unit columns
of the border, so the provider does not write `y`, and only $A_{bb}$ is
stored beforehand.

$$
(A^{-1})_{bb} = \frac{1}{16464}
\begin{pmatrix} 3349 & -474 \\ -598 & 2700 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_streamed_border_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The tangent from the converged pass

In a Newton iteration, the convergence is known once the residual is
assembled, that is after the last block. `solve_inplace_streamed_or_tangent`
asks `converged()` then: when it returns true, the border system the
pass has just condensed gives columns of the border block of $A^{-1}$,
the consistent tangent of a behaviour loaded through the border,
instead of the correction. The blocks are not asked for again. The
columns are those of `solve_border_canonical_multirhs`, bit for bit
under the default `rhs_pass_width` (within rounding errors under
another). The storage, y and the provider `block` are those of the
first snippet, set up before the markers.

$$
(A^{-1})_{bb} = \frac{1}{16464}
\begin{pmatrix} 3349 & -474 \\ -598 & 2700 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_streamed_tangent.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Recomputed blocks

The blocks asked for twice, nothing of the matrix stored: the border
block alone, by rows, and a store of refused blocks large enough that
the solve is never deferred. The solution agrees with the stored solve
within rounding errors.

```{literalinclude} ../../../snippets/arrowhead_lupp/static_recomputed.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The runtime solver

`ArrowheadLUppSolverDynamic` has the same entry points, the shape first.
Its block is a view of the workspace, with the accessors of the
compile-time solver: a generic lambda serves both. The block is written
where the stored solve copies it; $A_{bk}$ under the dense coupling and
the copies that restore a refused block come after the areas of the
stored solve: `streamed_workspace_size(shape)` entries, or
`streamed_workspace_size(shape, nrhs)` for the canonical columns and
the tangent.

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_streamed.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

The border block of the inverse with the runtime solver, the workspace
sized for its 2 columns:

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_streamed_border_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

A Newton iteration with the runtime solver: a pass that has not
converged gives the correction in y, the pass that has gives the
tangent in x. That pass writes intermediate values in y, the recovery
data of its right-hand side, so the snippet copies the correction of
the first pass before running the second. `border()` writes $A_{bb}$
and the border rows of y before each pass, and `block` is the provider
of the previous snippets.

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_streamed_tangent.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
