# Runtime shape

`ArrowheadLUppSolverDynamic` on the same system, the shape a runtime
argument. The storage, the workspace and the pivot array take their
sizes from it. The arithmetic is that of the compile-time solver, bit
for bit.

## Solve in place

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve with two buffers

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block in place

The number of right-hand sides is a runtime argument too, and so is
the workspace: `workspace_size(shape, nrhs)` entries.

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_solve_inplace_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block with two buffers

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_solve_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column of the border

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_border_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A block of canonical columns of the border

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_border_canonical_block.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The same bits as the compile-time solver

Both solvers on the system with a singular block: the same solution,
bit for bit, through the second pass.

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

```{literalinclude} ../../../snippets/arrowhead_lupp/dynamic_vs_static.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
