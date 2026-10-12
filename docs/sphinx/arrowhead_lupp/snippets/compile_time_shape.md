# Compile-time shape

`ArrowheadLUppSolverStatic` on the 6 x 6 system, one snippet per entry
point, on CPU. Its couplings are diagonal, the matrix stored Packed:
the entries of the pattern only, written through the storage map by
`store` (see {doc}`storages`). The calls are the same in a GPU kernel,
see {doc}`getting_started`.

## Solve in place

One buffer for the right-hand side and the solution.

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

```{literalinclude} ../../../snippets/arrowhead_lupp/static_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve with two buffers

`solve` copies b into x and solves in place: b stays as it was.

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

```{literalinclude} ../../../snippets/arrowhead_lupp/static_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block in place

The two columns of Y ride one condensation: each block solves its
coupling columns and both right-hand sides together, the border
system two columns. Each column of X comes out bit for bit as
`solve_inplace` gives it alone. The workspace holds
`multirhs_workspace_size<2>` entries.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
Y = \begin{pmatrix}
21 & 49 \\ 25 & 45 \\ 20 & 29 \\ 23 & 26 \\ 46 & 31 \\ 37 & 12
\end{pmatrix}, \quad
X = \begin{pmatrix}
1 & 6 \\ 2 & 5 \\ 3 & 4 \\ 4 & 3 \\ 5 & 2 \\ 6 & 1
\end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_solve_inplace_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block with two buffers

`solve_multirhs` copies the columns of Y into X and solves in place:
Y stays as it was.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
Y = \begin{pmatrix}
21 & 49 \\ 25 & 45 \\ 20 & 29 \\ 23 & 26 \\ 46 & 31 \\ 37 & 12
\end{pmatrix}, \quad
X = \begin{pmatrix}
1 & 6 \\ 2 & 5 \\ 3 & 4 \\ 4 & 3 \\ 5 & 2 \\ 6 & 1
\end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_solve_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column of the border

The right-hand side is the unit column $e_1$ of the border. It vanishes
on the blocks, which are condensed without right-hand side, and nothing
is recovered. The result is column 1 of the border block of $A^{-1}$,
border rows only.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
(A^{-1})_{bb} \, e_1 = \frac{1}{16464} \begin{pmatrix} -474 \\ 2700 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_border_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A block of canonical columns of the border

Every unit column of the border through one condensation: the whole
border block of $A^{-1}$, the inverse of the Schur complement of the
border. It is the tangent operator of a behaviour loaded through the
border equations.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
(A^{-1})_{bb} = \frac{1}{16464}
\begin{pmatrix} 3349 & -474 \\ -598 & 2700 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_border_canonical_block.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A refused block

The second block is singular. The pivot check refuses it, and the
second pass eliminates it with the border rows, which take the pivots
it lacks. The matrix is invertible: the solve succeeds, and the
trailing `int&` counts the refused block.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 1 & 1 \\
0 & 1 & 0 & 0 & 1 & 1 \\
\end{array}\right), \quad
b = \begin{pmatrix} 21 \\ 25 \\ 20 \\ 23 \\ 15 \\ 13 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/static_refused.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Constant evaluation

The whole solve, second pass included, runs during constant
evaluation, the system of the refused block above. A static_assert
certifies the result, and the compiler rejects any undefined behaviour
on the path.

```{literalinclude} ../../../snippets/arrowhead_lupp/static_constexpr.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```
