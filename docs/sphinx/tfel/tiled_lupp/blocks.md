# Blocks of right-hand sides

A matrix-like right-hand side holds one system per column. Every
column is solved against the one factorization.

## Factorize, then substitute a block

One factorization, three right-hand sides in one sweep.

$$
A X = B, \quad A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/block_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute a block in place

The same block, one buffer.

$$
A X = B, \quad A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/block_substitute_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block

Factorization and block substitution in one call, two buffers.

$$
A X = B, \quad A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/block_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve a block in place

Factorization and block substitution in one call, one buffer.

$$
A X = B, \quad A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/block_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A block of canonical columns

A matrix-like x: its columns receive the solutions of consecutive
canonical columns, the tangent-operator pattern.

$$
A X = \begin{pmatrix} e_0 & e_1 & e_2 \end{pmatrix}, \quad A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tfel/block_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Pass cutting

A wide block cut into passes of two columns, through the template
argument of the entry point.

$$
A X = B, \quad
B = \left(\begin{array}{cc|cc|c}
14 & 7 & 8 & 3 & 4 \
14 & 7 & 3 & 5 & 1 \
24 & 8 & 6 & 2 & 0 \
33 & 10 & 5 & 7 & 2
\end{array}\right), \quad
X = \left(\begin{array}{cc|cc|c}
1 & 1 & 2 & 0 & 1 \
2 & 1 & 0 & 1 & 0 \
3 & 1 & 1 & 0 & 0 \
4 & 1 & 0 & 1 & 0
\end{array}\right)
$$

```{literalinclude} ../../../snippets/tfel/block_pass_cutting.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A runtime-sized block

A runtime-sized matrix as right-hand side of a runtime-sized system:
the column count is read from the block.

$$
A X = B, \quad A = \left(\begin{array}{ccc|cc}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\ \hline
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{array}\right), \quad
B = \begin{pmatrix} 12 & 7 & 6 \ 16 & 8 & 2 \ 27 & 9 & 7 \ 40 & 10 & 2 \ 50 & 11 & 10 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 1 \ 2 & 1 & 0 \ 3 & 1 & 1 \ 4 & 1 & 0 \ 5 & 1 & 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/block_runtime.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
