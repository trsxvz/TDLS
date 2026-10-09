# Batches and layouts

Three 4 x 4 systems, $A + s I$ for $s = 0, 1, 2$, all with the same
solution. The batch is filled above the shown code. Each system is
solved by 2 threads of 2 rows, unless the snippet says otherwise.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
A_s = A + s I, \quad b_s = b + s x
$$

## Residencies

The same solve on the slices of the threads, then on a system of an
SoA batch. An internal operand is the slice of the calling thread: its
rows, their pivot entries and their right-hand-side entries, in local
arrays. An external operand is the whole object, reachable by the
group and walked with a stride.

```{literalinclude} ../../../snippets/cooperative_lupp/batch_residencies.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## AoS batch

The systems one after the other: stride 1.

```{literalinclude} ../../../snippets/cooperative_lupp/batch_aos.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## SoA batch

Element k of system s at `base[k * count + s]`: every operand walked
with the batch stride.

```{literalinclude} ../../../snippets/cooperative_lupp/batch_soa.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## AoSoA batch

Blocks of 4 interleaved systems, the last block padded: stride 4.

```{literalinclude} ../../../snippets/cooperative_lupp/batch_aosoa.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## An OpenMP loop over a batch

One thread per system on the CPU cores, in a function shown whole. The
threads of OpenMP do not cooperate: a barrier between them would cost
more than the solve itself.

```{literalinclude} ../../../snippets/cooperative_lupp/batch_openmp.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```

## Inside a GPU kernel

Groups of 2 lanes per system of an SoA batch, as many groups per warp
as fit, in the common CUDA/HIP dialect. Each group has its workspace
in shared memory. No barrier is passed: the solver deduces the barrier
of the 2 lanes, `__syncwarp` on them under CUDA, a wavefront fence
under HIP. Built with the GPU example options, skipped without a
device.

```{literalinclude} ../../../snippets/cooperative_lupp/batch_gpu.cu
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```
