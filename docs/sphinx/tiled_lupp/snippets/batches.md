# Batches and layouts

Three systems with the same solution, laid out as each snippet says.
The batch is filled above the shown code.

$$
A^{(s)} = A + s\,I, \quad b^{(s)} = b + s\,x, \quad s = 0, 1, 2, \quad
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

## Residencies

The same solve on local arrays, on a slice of an SoA batch, and with
the matrix in the batch while the pivot and the right-hand side stay
local. Element k of system s sits at:

$$
\text{index}(s, k) = 3\,k + s
$$

```{literalinclude} ../../../snippets/tiled_lupp/batch_residencies.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## AoS batch

Array of structures: the objects of a system are contiguous, the
stride is 1. Element k of system s sits at:

$$
\text{index}(s, k) = 16\,s + k
$$

```{literalinclude} ../../../snippets/tiled_lupp/batch_aos.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## SoA batch

Structure of arrays: consecutive systems touch consecutive addresses,
the stride is the batch size. Element k of system s sits at:

$$
\text{index}(s, k) = 3\,k + s
$$

```{literalinclude} ../../../snippets/tiled_lupp/batch_soa.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## AoSoA batch

Array of structures of arrays: blocks of W = 2 interleaved systems,
the stride is W. The batch is padded to 4 systems so that every block
is full. Element k of system s sits at:

$$
\text{index}(s, k) = 32 \left\lfloor s / 2 \right\rfloor + 2\,k + (s \bmod 2)
$$

```{literalinclude} ../../../snippets/tiled_lupp/batch_aosoa.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## An OpenMP loop over a batch

One iteration per system of an SoA batch, in a function shown whole.
Element k of system s sits at:

$$
\text{index}(s, k) = 3\,k + s
$$

```{literalinclude} ../../../snippets/tiled_lupp/batch_openmp.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 0
```

## Inside a GPU kernel

One thread per system of an SoA batch, in the common CUDA/HIP dialect:
the kernel is shown, the host side allocates, launches and copies
back. Element k of system s sits at:

$$
\text{index}(s, k) = \text{count} \cdot k + s
$$

```{literalinclude} ../../../snippets/tiled_lupp/batch_gpu.cu
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 0
```
