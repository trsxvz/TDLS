# Configuration

The knobs of `ArrowheadLUppConfig`, one snippet each, on the system of
the other pages unless the snippet says otherwise.

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

## Defaults

Dense couplings, the Full storage, the row-major layout, a pivot check
at 0.01, one pass.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_defaults.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Column-major layout

The matrix stored by columns. TiledLUpp factors the blocks as stored,
in its column-major mode, and the pivot check reads the forward stage
of the solve.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_layout.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The threshold of the pivot check

The second block has a pivot of 1 under a coupling of 2: a multiplier
of 2. At the default, 0.01, the block passes; at 1, every pivot must be
the largest candidate of its column, the border rows included, and the
block is refused. Both solutions are right.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 1 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
b = \begin{pmatrix} 21 \\ 25 \\ 20 \\ 23 \\ 21 \\ 37 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/config_condensation_threshold.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The TiledLUpp solvers of the blocks and of the border

Their tile sizes and the two thresholds of the blocks.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_blocks.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Passes

The columns $[A_{kb} \mid y_k]$ of a block solved two at a time.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_pass_width.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Gather

The pivots of a block applied to its columns by a gather.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_gather.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Schur complement in place

The Schur complement accumulated in the border entries of the storage and of the right-hand side.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_schur_in_place.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Unroll policy

No forced unrolling, in the solver and in its TiledLUpp solvers.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_unroll.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Scalar type

The same solve in float. The floating-point knobs are written as float
values.

```{literalinclude} ../../../snippets/arrowhead_lupp/config_scalar_types.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
