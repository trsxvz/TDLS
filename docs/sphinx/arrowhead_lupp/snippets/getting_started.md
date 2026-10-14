# Getting started

Three short programs: a first call on CPU, one GPU thread per system,
and the solver of a runtime shape. Each one solves the 6 x 6 system of
a border of 2 unknowns and two blocks of 2

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
b = \begin{pmatrix} 21 \\ 25 \\ 20 \\ 23 \\ 46 \\ 37 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

## On CPU

The shape is a template argument: a border of 2, then 2 blocks of 2. The
default configuration reads the full matrix, row-major, and accepts any
coupling. The call takes the operands with their strides; the template
booleans say where each one lives: `true` for a local array. The
workspace is touched only when a block is refused.

```{literalinclude} ../../../snippets/arrowhead_lupp/start_cpu.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## On GPU

One GPU thread per system, on a batch of three systems $A + s I$ for
$s = 0, 1, 2$, stored one after the other. Each thread makes the call
of the CPU snippet on its own system, with the workspace in a local
array. The snippet compiles with CUDA or HIP: it is built with the GPU
example options and skipped without a device.

```{literalinclude} ../../../snippets/arrowhead_lupp/start_gpu.cu
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```

## Runtime shape

The shape is a runtime argument, given first. The temporaries of the
solve live in the workspace and the pivots in an int array, both sized
from the shape.

```{literalinclude} ../../../snippets/arrowhead_lupp/start_dynamic.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
