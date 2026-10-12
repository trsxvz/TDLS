# Through the raw interface

The `tfel::math` objects store their elements contiguously and
row-major. A `tmatrix` jacobian is therefore the Full storage of an
arrowhead matrix, and its `data()` pointer with a stride of 1 a valid
operand of the ArrowheadLUpp solvers. The system has a border of 2
unknowns and two blocks of 2, with diagonal couplings.

$$
J = \left(\begin{array}{cc|cc|cc}
6 & 1 & 1 & 0 & 2 & 0 \\
1 & 7 & 0 & 1 & 0 & 1 \\ \hline
1 & 0 & 5 & 1 & 0 & 0 \\
0 & 2 & 1 & 4 & 0 & 0 \\ \hline
2 & 1 & 0 & 0 & 6 & 2 \\
0 & 1 & 0 & 0 & 1 & 5
\end{array}\right), \quad
y = \begin{pmatrix} 21 \\ 25 \\ 20 \\ 23 \\ 46 \\ 37 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

## The jacobian as the Full storage

```{literalinclude} ../../../snippets/tfel/arrowhead_fixed_full.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The jacobian written in the Packed storage

The entries of the pattern in a `tvector`, each at the index of the
storage map.

```{literalinclude} ../../../snippets/tfel/arrowhead_fixed_packed.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Runtime-sized objects

`matrix` and `vector` with the runtime solver, the shape given at run
time.

```{literalinclude} ../../../snippets/tfel/arrowhead_runtime.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
