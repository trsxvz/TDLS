# Getting started

Three short programs: a first call on CPU, a group of GPU threads, and
the solver of a runtime dimension. Each one solves the 4 x 4 system

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

## On CPU

One thread solves the system: `rows_per_thread` = 4, as many rows as
the dimension. Every entry point takes the rank of the thread first,
here 0, then the operands with their strides, then `work`, the scratch
space of the solver. The template booleans say where each operand
lives: `true` for an array local to the thread.

```{literalinclude} ../../../snippets/cooperative_lupp/start_cpu.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## On GPU

A group of 2 GPU threads, lanes of one warp, solves the system with 2
rows each. Every thread of the group makes the same call with its own
rank `tx`; the workspace and the pivots, shared by the group, live in
shared memory. The solver deduces the barrier of the 2 threads. The
snippet compiles with CUDA or HIP: it is built with the GPU example
options and skipped without a device.

```{literalinclude} ../../../snippets/cooperative_lupp/start_gpu.cu
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```

The same call works on CPU threads too, see {doc}`groups`.

## Runtime dimension

The dimension n is a runtime argument, given first. Every operand is a
pointer plus a stride, and `workspace_size(n)` sizes the workspace.

```{literalinclude} ../../../snippets/cooperative_lupp/start_dynamic.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
