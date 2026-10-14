# Through the raw interface

The `tfel::math` objects store their elements contiguously and
row-major. Their `data()` pointer and a stride of 1 are therefore
valid operands of the CooperativeLUpp solvers, as internal slices when
one thread solves the system, or as external objects. $t_j$ marks the
thread that holds each column of A: under the default row-major
layout, the threads hold columns.

## One thread per system

A `rows_per_thread` equal to the dimension: the only thread holds every
column, so the TFEL storage is its slice.

$$
A = \overset{\displaystyle\begin{matrix} \scriptstyle t_0 & \scriptstyle t_0 & \scriptstyle t_0 & \scriptstyle t_0 \end{matrix}}{\begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/cooperative_fixed_one_thread.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A group of threads

Two threads of two columns each share the objects: external operands,
walked with a stride of 1.

$$
A = \overset{\displaystyle\begin{matrix} \scriptstyle t_0 & \scriptstyle t_1 & \scriptstyle t_0 & \scriptstyle t_1 \end{matrix}}{\begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/cooperative_fixed_group.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Runtime-sized objects

`matrix` and `vector` with the runtime solver: the dimension is read
from the object, the group takes 3 threads for n = 5.

$$
A = \overset{\displaystyle\begin{matrix} \scriptstyle t_0 & \scriptstyle t_1 & \scriptstyle t_2 & \scriptstyle t_0 & \scriptstyle t_1 \end{matrix}}{\begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/cooperative_runtime_group.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
