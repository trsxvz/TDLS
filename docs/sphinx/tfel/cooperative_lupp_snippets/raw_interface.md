# Through the raw interface

The `tfel::math` objects store their elements contiguously and
row-major. Their `data()` pointer and a stride of 1 are therefore
valid operands of the CooperativeLUpp solvers, as internal slices or as
external objects. $t_i$ marks the thread that holds each row.

## One thread per system

As many rows per thread as the dimension: the only thread holds every
row, so the TFEL storage is its slice.

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

```{literalinclude} ../../../snippets/tfel/cooperative_fixed_one_thread.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A group of threads

Two threads of two rows share the objects: external operands, walked
with a stride of 1.

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

```{literalinclude} ../../../snippets/tfel/cooperative_runtime_group.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
