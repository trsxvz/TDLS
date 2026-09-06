# Fixed-size objects

`tmatrix` and `tvector`: the dimension is read from their indexing
policy at compile time, and they resolve the compile-time solver with
the internal residency.

## Factorize, then substitute

One factorization, two right-hand sides.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad
b_1 = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 7 \\ 8 \\ 10 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute in place

One buffer for the right-hand side and the solution.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_substitute_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in one call

Factorization and substitution in one call, two buffers.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in place

Factorization and substitution in one call, one buffer.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column

The right-hand side is a canonical vector, generated on the fly.

$$
A x = e_2, \quad A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad e_2 = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## With a configuration value

A configuration value of the matrix scalar type, as the first template
argument. Every entry point has this form. Tiles of 2 here.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_config.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Counting overloads

`factorize`, `solve` and `solve_inplace` take a trailing `int&`:
the number of columns whose best in-tile pivot fell below the
threshold. The three candidates of column 0 inside the tile, in red,
are tiny; the search below the tile takes row 3, in blue.

$$
A = \left(\begin{array}{ccc|c}
\textcolor{red}{10^{-12}} & 1 & 0 & 2 \\
\textcolor{red}{2 \cdot 10^{-12}} & 5 & 1 & 0 \\
\textcolor{red}{3 \cdot 10^{-12}} & 1 & 6 & 1 \\ \hline
\textcolor{blue}{2} & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 10 \\ 13 \\ 24 \\ 31 \end{pmatrix}, \quad
x = \begin{pmatrix} 0 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/fixed_counting.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Pivot forms

The pivot is any dense `int` object of the dimension, or a raw `int`
array or pointer. The four factorizations are bitwise identical.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tfel/fixed_pivot_forms.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
