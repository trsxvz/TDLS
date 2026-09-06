# Compile-time dimension

`TiledLUppSolverStatic` on a 4 x 4 system in tiles of 2. The matrix,
the right-hand sides and the expected solutions are declared above the
shown code.

## Factorize, then substitute

One factorization, two right-hand sides. The matrix holds the factors
afterwards and the pivot array the row permutation.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad
b_1 = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 7 \\ 8 \\ 10 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute in place

One buffer for the right-hand side and the solution.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_substitute_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in one call

Factorization and substitution in one call, two buffers.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in place

Factorization and substitution in one call, one buffer. The forward
substitution is folded into the factorization, so the result is bitwise
the one of factorize then substitute in place.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column

The right-hand side is a canonical vector, generated on the fly: the
solution is a column of the inverse.

$$
A x = e_2, \quad A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad e_2 = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A block of canonical columns

Consecutive canonical columns solved together: the columns of the
inverse a consistent tangent operator needs.

$$
A X = \begin{pmatrix} e_0 & e_1 & e_2 \end{pmatrix}, \quad A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_canonical_block.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute a block

Three right-hand sides in one sweep, every tile loaded once for the
block.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute a block in place

The same block, one buffer.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_multirhs_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block

Factorization and block substitution in one call, two buffers.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_solve_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block in place

Factorization and block substitution in one call, one buffer. The
forward pass is not folded into the factorization here.

$$
A X = B, \quad A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_solve_inplace_multirhs.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Pass cutting

A wide block cut into passes of two columns: a working-set knob, with
the last pass taking the remainder.

$$
A X = B, \quad
B = \left(\begin{array}{cc|cc|c}
14 & 7 & 8 & 3 & 4 \\
14 & 7 & 3 & 5 & 1 \\
24 & 8 & 6 & 2 & 0 \\
33 & 10 & 5 & 7 & 2
\end{array}\right), \quad
X = \left(\begin{array}{cc|cc|c}
1 & 1 & 2 & 0 & 1 \\
2 & 1 & 0 & 1 & 0 \\
3 & 1 & 1 & 0 & 0 \\
4 & 1 & 0 & 1 & 0
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_pass_cutting.cpp
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
A = \left(\begin{array}{cc|cc}
\textcolor{red}{10^{-12}} & 1 & 0 & 2 \\
\textcolor{red}{2 \cdot 10^{-12}} & 5 & 1 & 0 \\ \hline
\textcolor{blue}{3} & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 10 \\ 13 \\ 24 \\ 31 \end{pmatrix}, \quad
x = \begin{pmatrix} 0 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_oot_counter.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Singular verdict

Column 2, in red, is zero: no pivot reaches the floor and the call
returns `false`.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & \textcolor{red}{0} & 2 \\
1 & 5 & \textcolor{red}{0} & 0 \\ \hline
0 & 1 & \textcolor{red}{0} & 1 \\
2 & 0 & \textcolor{red}{0} & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_singular.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Constant evaluation

The whole solve runs during constant evaluation. The data lives inside
the constexpr function, so the whole program is shown.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_constexpr.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 0
```

## Newton iteration, then tangent columns

The MFront pattern on a small nonlinear system: one solve in place per
iteration on a fresh jacobian, then the tangent columns from one
factorization of the converged jacobian. The residual and jacobian
functions are declared above the shown code.

$$
F(x) = A x + x \circ x - c, \quad J(x) = A + 2\,\mathrm{diag}(x), \quad
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad
c = \begin{pmatrix} 15 \\ 18 \\ 33 \\ 49 \end{pmatrix}, \quad
x^\star = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/static_newton.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
