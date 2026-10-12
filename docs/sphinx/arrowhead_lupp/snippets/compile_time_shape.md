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
