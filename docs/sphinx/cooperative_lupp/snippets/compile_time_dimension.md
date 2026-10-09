# Compile-time dimension

`CooperativeLUppSolverStatic<T, N, Config>`, one snippet per entry
point. Unless the snippet says otherwise, the 4 x 4 system is solved
by 2 threads of 2 rows: rows 0 and 2 on thread 0, rows 1 and 3 on
thread 1. The operands are whole objects, external to the threads.

## Factorize, then substitute

One factorization, two right-hand sides. The matrix holds the factors
on exit, the pivot array the position of each row.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b_1 = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 7 \\ 8 \\ 10 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_factorize_substitute.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Factorize, then substitute in place

One buffer for the right-hand side and the solution.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_substitute_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in one call

The factorization and the substitution in one call. The rows stay
with their threads in between.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_solve.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solve in place

The forward pass folded into the factorization, the original MAGMA
kernel. y holds b on entry and x on exit.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_solve_inplace.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One canonical column

The right-hand side is a canonical vector, generated on the fly: the
consistent tangent operator path.

$$
A x = e_2, \quad A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_canonical.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Singular verdict

Column 2 is zero. Every thread runs the whole sequence of barriers and
receives the same verdict.

$$
A = \begin{pmatrix}
4 & 1 & \color{red}{0} & 2 \\
1 & 5 & \color{red}{0} & 0 \\
0 & 1 & \color{red}{0} & 1 \\
2 & 0 & \color{red}{0} & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_singular.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Constant evaluation

The whole solve runs during constant evaluation, on one thread. The
data lives inside the constexpr function.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_0 \\ t_0 \\ t_0 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_constexpr.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```

## Newton iteration, then tangent columns

The MFront pattern on a small nonlinear system, run by the group. Each
thread builds the rows it holds, so the solver reads them without a
barrier. On return, every thread reads the whole Newton step and
updates its copy of x. A barrier keeps the next iteration from
overwriting the step before every thread has read it. At the solution,
one factorization serves the tangent columns.

$$
F(x) = A x + x \odot x - c = 0, \quad
J(x) = A + 2 \operatorname{diag}(x), \quad
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
c = \begin{pmatrix} 15 \\ 18 \\ 33 \\ 49 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_newton.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The deduced barrier

The system of the first snippet, on 2 CPU threads, without a barrier
argument: the solver deduces the barrier of the group, held in the
last two elements of the workspace. `make_sync` returns the same
barrier, for an exchange of the threads between two substitutions:
each thread writes the right-hand side of the rows of the other one.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b_1 = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 7 \\ 8 \\ 10 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/static_deduced_barrier.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
