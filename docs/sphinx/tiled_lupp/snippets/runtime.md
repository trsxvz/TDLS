# Runtime dimension

`TiledLUppSolverDynamic` on a 5 x 5 system in tiles of 2: two full
tiles and a trailing tile of 1. The dimension `n`, the matrix `A`,
the right-hand sides and the expected solutions are declared above the
shown code, the arrays as `std::vector<double>`.

## Factorize, then substitute

One factorization, two right-hand sides. The dimension is the first
argument of every call, and every operand is a pointer plus a stride.

$$
A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad
b_1 = \begin{pmatrix} 12 \ 16 \ 27 \ 40 \ 50 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \ 8 \ 9 \ 10 \ 11 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \ 2 \ 3 \ 4 \ 5 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \ 1 \ 1 \ 1 \ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute in place

One buffer for the right-hand side and the solution.

$$
A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_substitute_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in one call

Factorization and substitution in one call, two buffers.

$$
A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in place

Factorization and substitution in one call, one buffer, the forward
substitution folded into the factorization.

$$
A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column

The right-hand side is a canonical vector, generated on the fly.

$$
A x = e_2, \quad A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad e_2 = \begin{pmatrix} 0 \ 0 \ 1 \ 0 \ 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A block of canonical columns

Consecutive canonical columns solved together. The column count is a
runtime argument, right after the dimension.

$$
A X = \begin{pmatrix} e_0 & e_1 & e_2 \end{pmatrix}, \quad A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_canonical_block.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute a block

Three right-hand sides in one sweep. A block carries two strides: the
element stride within a column, and the stride between columns.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad B = \begin{pmatrix} 12 & 7 & 6 \\ 16 & 8 & 2 \\ 27 & 9 & 7 \\ 40 & 10 & 2 \\ 50 & 11 & 10 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \\ 5 & 1 & 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute a block in place

The same block, one buffer.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad B = \begin{pmatrix} 12 & 7 & 6 \\ 16 & 8 & 2 \\ 27 & 9 & 7 \\ 40 & 10 & 2 \\ 50 & 11 & 10 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \\ 5 & 1 & 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_multirhs_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block

Factorization and block substitution in one call, two buffers.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad B = \begin{pmatrix} 12 & 7 & 6 \\ 16 & 8 & 2 \\ 27 & 9 & 7 \\ 40 & 10 & 2 \\ 50 & 11 & 10 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \\ 5 & 1 & 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_solve_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block in place

Factorization and block substitution in one call, one buffer.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad B = \begin{pmatrix} 12 & 7 & 6 \\ 16 & 8 & 2 \\ 27 & 9 & 7 \\ 40 & 10 & 2 \\ 50 & 11 & 10 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \\ 5 & 1 & 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_solve_inplace_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Pass cutting

A wide block cut into passes of two columns. The column count stays a
runtime argument, the pass width is a template argument.

$$
A X = B, \quad
B = \left(\begin{array}{cc|cc|c}
12 & 7 & 6 & 1 & 10 \
16 & 8 & 2 & 6 & 2 \
27 & 9 & 7 & 2 & 0 \
40 & 10 & 2 & 8 & 0 \
50 & 11 & 10 & 1 & 2
\end{array}\right), \quad
X = \left(\begin{array}{cc|cc|c}
1 & 1 & 1 & 0 & 2 \
2 & 1 & 0 & 1 & 0 \
3 & 1 & 1 & 0 & 0 \
4 & 1 & 0 & 1 & 0 \
5 & 1 & 1 & 0 & 0
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_pass_cutting.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Out-of-tile counter

The two candidates of column 0 inside the first tile, in red, are below
the threshold. The search below the tile takes row 2, in blue, and the
counter reports the column.

$$
A = \left(\begin{array}{cc|cc|c}
\textcolor{red}{10^{-12}} & 1 & 0 & 0 & 1 \
\textcolor{red}{2 \cdot 10^{-12}} & 6 & 1 & 0 & 0 \ \hline
\textcolor{blue}{3} & 1 & 7 & 1 & 0 \
0 & 0 & 1 & 8 & 1 \ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad
b = \begin{pmatrix} 7 + 10^{-12} \ 15 + 2 \cdot 10^{-12} \ 30 \ 40 \ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_oot_counter.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Tile helpers

The grid of a runtime dimension, read from the solver: the number of
tiles per dimension and the extent of a tile.

$$
A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_tile_helpers.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A tile larger than the dimension

A tile size above the dimension: the grid is a single partial tile.

$$
A = \begin{pmatrix}
3 & 1 & 0 \
1 & 4 & 1 \
0 & 1 & 5
\end{pmatrix}, \quad
b = \begin{pmatrix} 5 \ 12 \ 17 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \ 2 \ 3 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_tile_larger.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Same results as the compile-time solver

The two solvers at equal shape and configuration: the factors, the
pivots and the solution are bitwise identical.

$$
A = \left(\begin{array}{cc|cc|c}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\ \hline
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\ \hline
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/dynamic_vs_static.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
