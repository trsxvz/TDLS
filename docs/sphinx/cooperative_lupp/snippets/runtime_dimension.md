# Runtime dimension

`CooperativeLUppSolverDynamic<T, Config>`, one snippet per entry point.
The dimension is the first argument, the rank of the thread the
second. Every operand is a pointer plus a stride. Unless the snippet
says otherwise, n = 5 on threads of 2 rows: a group of 3 threads, the
last one holding a phantom row in its second slot.

## Factorize, then substitute

One factorization, two right-hand sides.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b_1 = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 8 \\ 9 \\ 10 \\ 11 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute in place

One buffer for the right-hand side and the solution.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_substitute_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in one call

The factorization and the substitution in one call.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in place

The forward pass folded into the factorization. y holds b on entry and
x on exit.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column

The right-hand side is a canonical vector, generated on the fly.

$$
A x = e_2, \quad A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The group of a runtime dimension

The number of threads and the size of the workspace, read from the
solver for a runtime n: 3 threads and 15 elements for n = 5.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_threads.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## More rows per thread than the dimension

A `rows_per_thread` of 8: one thread solves every n up to 8, and the
deduced barrier does nothing for it. Beyond 8, the group would need a
second thread, so the snippet checks the number of threads.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_0 \\ t_0 \\ t_0 \\ t_0 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_rows_larger.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Same results as the compile-time solver

The two solvers at equal shape and configuration: the factors, the
row positions and the solution are bitwise identical.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/dynamic_vs_static.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
