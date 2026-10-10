# Configuration

The knobs of `CooperativeLUppConfig`, one snippet each, on CPU. Their
roles are listed on the {doc}`family page <../index>`. The knobs of
the groups (the defaults, `rows_per_thread`, the phantom rows) run
groups of CPU threads with `snippets::run_group`, see {doc}`groups`;
next to their matrices, $t_i$ marks the thread that holds the row. The
other knobs solve with one thread.

## Defaults

The default value, and the two solvers named on it. One row per
thread: the 4 x 4 system takes a group of 4 threads.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_3 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_defaults.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Rows per thread

The same 6 x 6 system on three mappings: 6 threads of 1 row, 3 threads
of 2 rows, 2 threads of 3 rows. The solutions are bitwise identical.
On 3 threads:

$$
A = \begin{pmatrix}
8 & 1 & 0 & 0 & 2 & 0 \\
1 & 8 & 1 & 0 & 0 & 2 \\
0 & 1 & 8 & 1 & 0 & 0 \\
0 & 0 & 1 & 8 & 1 & 0 \\
2 & 0 & 0 & 1 & 8 & 1 \\
0 & 2 & 0 & 0 & 1 & 8
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \\ t_2 \end{matrix}, \quad
b = \begin{pmatrix} 20 \\ 32 \\ 30 \\ 40 \\ 52 \\ 57 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_rows_per_thread.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## One thread per system

As many rows per thread as the dimension: one thread solves the
system, with no barrier. Every operand can then be a local array.

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

```{literalinclude} ../../../snippets/cooperative_lupp/config_one_thread.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Phantom rows

A dimension that is not a multiple of `rows_per_thread`. The second
slot holds rows 3 and 4 on threads 0 and 1, and a phantom row on
thread 2.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 0 & 1 \\
1 & 5 & 1 & 0 & 0 \\
0 & 1 & 6 & 1 & 0 \\
0 & 0 & 1 & 7 & 1 \\
1 & 0 & 0 & 1 & 8
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 11 \\ 14 \\ 24 \\ 36 \\ 45 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_phantom_rows.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## More rows per thread than the dimension

A `rows_per_thread` above the dimension acts as the dimension: one
thread per system.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 0 & 1 \\
1 & 5 & 1 & 0 & 0 \\
0 & 1 & 6 & 1 & 0 \\
0 & 0 & 1 & 7 & 1 \\
1 & 0 & 0 & 1 & 8
\end{pmatrix}
\begin{matrix} t_0 \\ t_0 \\ t_0 \\ t_0 \\ t_0 \end{matrix}, \quad
b = \begin{pmatrix} 11 \\ 14 \\ 24 \\ 36 \\ 45 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_rows_larger.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Row interchanges

The same system under logical and physical row interchanges. Row 0
has a zero in column 0, so row 1 holds its pivot, whatever the
threshold. The logical scheme leaves the rows in place and records
their positions; the physical one moves them, as LAPACK does. The
solutions are bitwise identical, and so are the factored rows once
placed.

$$
A = \begin{pmatrix}
0 & 2 & 0 & 1 \\
\color{red}{4} & 1 & 1 & 0 \\
0 & 1 & 3 & 1 \\
2 & 0 & 1 & 5
\end{pmatrix}, \quad
b = \begin{pmatrix} 8 \\ 9 \\ 15 \\ 25 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_row_interchange.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Relative pivot threshold

The row in place holds 3 in column 0, below the 4 of row 1. The
partial pivoting of LAPACK, a threshold of 1, exchanges the two rows.
The default threshold 0.1 keeps the row in place, since 3 reaches a
tenth of 4: no row moves at all.

$$
A = \begin{pmatrix}
\color{red}{3} & 1 & 0 & 0 \\
4 & 6 & 1 & 0 \\
0 & 1 & 5 & 1 \\
0 & 0 & 1 & 4
\end{pmatrix}, \quad
b = \begin{pmatrix} 5 \\ 19 \\ 21 \\ 19 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_pivot_threshold.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Singularity floor

The last pivot is tiny. The default floor accepts it, a raised one
declares the matrix singular.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 0 \\
1 & 5 & 0 & 0 \\
0 & 0 & 6 & 0 \\
0 & 0 & 0 & \color{red}{10^{-8}}
\end{pmatrix}, \quad
b = \begin{pmatrix} 6 \\ 11 \\ 18 \\ 0 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_singular_floor.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## No forced unrolling

The same solve with and without the unroll pragmas: bitwise identical
results.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_unroll.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Column-major layout

The matrix stored column by column, as one system of a batch of 3:
element (r, c) at `batch[(c * 4 + r) * 3 + 1]`. The layout picks the
vectors of each thread: under the column-major layout the threads hold
rows of A and the solver is the kernel of MAGMA; under the default
row-major one they hold columns, and the solve runs through the factors
of A^T (see {doc}`../index`).

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_layout.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Scalar types

The same system in `float` and in `long double`.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/config_scalar_types.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solver constants

The mapping and the knobs, read back from the solver type: 13 rows on
threads of 5 rows take 3 threads, and the last slot holds a real row
on thread 0 only.

```{literalinclude} ../../../snippets/cooperative_lupp/config_constants.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
