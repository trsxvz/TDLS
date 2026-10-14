# Storages and couplings

The same system in the four storages, then with dense couplings. Under
the diagonal coupling, the first unknown of each block touches the first
border row only, and the second unknown the second row: one coupling
entry per unknown of the block.

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

## Full

The whole $n \times n$ matrix, zeros included. The solver reads the
entries of the pattern only.

```{literalinclude} ../../../snippets/arrowhead_lupp/storage_full.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Packed

The entries of the pattern only: $A_{bb}$, the couplings, the $A_{kb}$,
then the $D_k$. The storage map gives the index of every entry: the
loop below does by hand what `store` does in the other snippets.

```{literalinclude} ../../../snippets/arrowhead_lupp/storage_packed.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Csr

The canonical CSR arrays of the pattern, written by `csr_pattern`: a
CSR matrix assembled elsewhere with these arrays (columns in increasing
order, the zeros of the blocks stored) is already in the storage. The
solver reads the values only; it never reads the index arrays.

```{literalinclude} ../../../snippets/arrowhead_lupp/storage_csr.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Bsr

The canonical BSR arrays, in square blocks of 2: three block rows, the
border then the two blocks. Each block is stored by rows.

```{literalinclude} ../../../snippets/arrowhead_lupp/storage_bsr.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Dense couplings

The border rows touch every unknown of the blocks: the default
coupling, $B \times S$ entries per block.

$$
A = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 1 & 2 & 0 \\
1 & 7 & 0 & 1 & 1 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5 \\
\end{array}\right), \quad
b = \begin{pmatrix} 25 \\ 30 \\ 20 \\ 23 \\ 46 \\ 37 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/arrowhead_lupp/coupling_dense.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
