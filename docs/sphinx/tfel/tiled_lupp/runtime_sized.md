# Runtime-sized objects

`matrix` and `vector`: their extents live in the objects, the
dimension is read at run time and the runtime solver is resolved. The
other entry points are spelled exactly as on fixed-size objects.

## Factorize, then substitute

One factorization, two right-hand sides.

$$
A = \left(\begin{array}{ccc|cc}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\ \hline
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad
b_1 = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 8 \\ 9 \\ 10 \\ 11 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/runtime_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in place

One buffer for the right-hand side and the solution.

$$
A = \left(\begin{array}{ccc|cc}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\ \hline
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/runtime_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column

The right-hand side is a canonical vector, generated on the fly.

$$
A x = e_2, \quad A = \left(\begin{array}{ccc|cc}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\ \hline
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad e_2 = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/runtime_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
