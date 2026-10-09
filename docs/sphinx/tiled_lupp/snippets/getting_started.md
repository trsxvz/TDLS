# Getting started

Three short programs: a first call on CPU, one GPU thread per system,
and the solver of a runtime dimension. Each one solves the 4 x 4
system

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

## On CPU

The matrix is cut into tiles of 2 x 2, the lines of the formula. The
call takes the operands with their strides; the template booleans say
where each one lives: `true` for a local array.

```{literalinclude} ../../../snippets/tiled_lupp/start_cpu.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## On GPU

One GPU thread per system, on a batch of three systems $A + s I$ for
$s = 0, 1, 2$, stored one after the other. Each thread makes the call
of the CPU snippet on its own system, with the pivots in a local
array. The snippet compiles with CUDA or HIP: it is built with the GPU
example options and skipped without a device.

```{literalinclude} ../../../snippets/tiled_lupp/start_gpu.cu
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```

## Runtime dimension

The dimension n is a runtime argument, given first. Every operand is a
pointer plus a stride.

```{literalinclude} ../../../snippets/tiled_lupp/start_dynamic.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
