# Batches and layouts

A batch of three systems $A + s I$, $s = 0, 1, 2$, all with the solution
$x = (1, 2, 3, 4, 5, 6)$, Packed and diagonal. The caller offsets the
pointers of each system and passes the stride of its layout: the
solver sees one strided system.

## AoS

Each system contiguous, one after the other: stride 1.

```{literalinclude} ../../../snippets/arrowhead_lupp/batch_aos.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## SoA

Entry $e$ of system $s$ at $e \cdot 3 + s$: the stride is the batch
size. On GPU, the consecutive threads then read consecutive addresses.

```{literalinclude} ../../../snippets/arrowhead_lupp/batch_soa.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## AoSoA

Blocks of 2 interleaved systems, the batch padded to whole blocks: the
stride is 2. On GPU, blocks of 32 systems keep the reads of a warp
coalesced and each system within a few pages; measured on an RTX 5080,
up to 23 % faster than the SoA batch.

```{literalinclude} ../../../snippets/arrowhead_lupp/batch_aosoa.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Mixed residencies

The matrix in the batch, the right-hand side copied into a local array:
external and internal operands in one call.

```{literalinclude} ../../../snippets/arrowhead_lupp/batch_residencies.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## OpenMP

One iteration per system of the SoA batch, the workspace local to the
iteration.

```{literalinclude} ../../../snippets/arrowhead_lupp/batch_openmp.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```

## GPU

One thread per system of the SoA batch, the workspaces in GPU memory
too, in SoA. The snippet compiles with CUDA or HIP.

```{literalinclude} ../../../snippets/arrowhead_lupp/batch_gpu.cu
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
```
