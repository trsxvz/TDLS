# Views

Views map memory the caller owns as `tfel::math` objects. A contiguous
view resolves the internal residency, a strided view the external one,
and the stride is read from the view.

## A view on a pointer

`map` on a plain pointer, and on a const pointer for a read-only
operand.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/view_pointer.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A strided view over an SoA batch

`map_strided` on system 1 of a batch of three systems stored
structure-of-arrays: element k of system s sits at index 3 k + s.

$$
A^{(1)} = A + I, \quad b^{(1)} = b + x, \quad
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/view_strided.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Elements of a views array

`map_array` maps contiguous objects as an array of views. Each element
is a view, accepted on its own.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad
B = \begin{pmatrix} 14 & 7 & 8 \\ 14 & 7 & 3 \\ 24 & 8 & 6 \\ 33 & 10 & 5 \end{pmatrix}, \quad
X = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 1 & 0 \\ 3 & 1 & 1 \\ 4 & 1 & 0 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/view_array.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Mixed placements

A strided view for the matrix, plain objects for the pivot and the
right-hand side: one residency per argument.

$$
A^{(2)} = A + 2\,I, \quad b^{(2)} = b + 2\,x, \quad
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tfel/view_mixed.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
