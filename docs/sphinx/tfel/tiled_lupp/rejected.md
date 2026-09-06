# Rejected shapes

Translation units that must not compile. ctest builds each one and
passes only when the diagnostic below appears, so the contracts cannot
be dropped silently.

## Sub-matrix view

The rows of a sub-matrix view, in red, are not one row length apart:
the single-stride addressing cannot express it.

$$
J = \begin{pmatrix}
\textcolor{red}{j_{00}} & \textcolor{red}{j_{01}} & j_{02} & j_{03} \\
\textcolor{red}{j_{10}} & \textcolor{red}{j_{11}} & j_{12} & j_{13} \\
j_{20} & j_{21} & j_{22} & j_{23} \\
j_{30} & j_{31} & j_{32} & j_{33}
\end{pmatrix}, \quad \text{rows of the block 4 elements apart, columns 1}
$$

```{literalinclude} ../../../snippets/tfel/reject_submatrix.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

Diagnostic:

```
tdls adaptors: row-strided matrix views (sub-matrix views) cannot be expressed by the single-stride addressing of the TiledLUpp solvers
```

## Derivative view

A derivative block, in red, mapped inside a larger jacobian: the same
row stride problem.

$$
J = \left(\begin{array}{c|c}
\textcolor{red}{\partial f / \partial x} & \ast \\ \hline
\ast & \ast
\end{array}\right), \quad f, x \in \mathbb{R}^2, \quad J \in \mathbb{R}^{4 \times 4}
$$

```{literalinclude} ../../../snippets/tfel/reject_derivative.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

Diagnostic:

```
tdls adaptors: row-strided matrix views (sub-matrix views) cannot be expressed by the single-stride addressing of the TiledLUpp solvers
```

## Gather view

One pointer per element: no data pointer, no stride.

$$
G = \begin{pmatrix}
p_{00} & p_{01} & p_{02} & p_{03} \\
p_{10} & p_{11} & p_{12} & p_{13} \\
p_{20} & p_{21} & p_{22} & p_{23} \\
p_{30} & p_{31} & p_{32} & p_{33}
\end{pmatrix}, \quad p_{ij} \text{ a pointer to the element}
$$

```{literalinclude} ../../../snippets/tfel/reject_gather.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

Diagnostic:

```
tdls adaptors: A must be a dense object exposing data() and an indexing_policy type (a gather view holding one pointer per element is not one)
```

## Column-major configuration

TFEL stores matrices row-major.

$$
A = \begin{pmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{pmatrix}
\ \text{is stored as}\ (a_{00}, a_{01}, a_{10}, a_{11}),
\ \text{not as}\ (a_{00}, a_{10}, a_{01}, a_{11})
$$

```{literalinclude} ../../../snippets/tfel/reject_colmajor.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

Diagnostic:

```
tdls adaptors: dense objects are addressed row-major, the TFEL convention; a column-major configuration cannot be used through the adaptors
```

## Const matrix

The factorizing entry points write their matrix argument.

```{literalinclude} ../../../snippets/tfel/reject_const.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

Diagnostic:

```
tdls adaptors: A must not be const here (factorize writes it)
```

## Extent mismatch

A right-hand side whose compile-time extent differs from the
dimension.

$$
A \in \mathbb{R}^{4 \times 4}, \quad b \in \mathbb{R}^{3}
$$

```{literalinclude} ../../../snippets/tfel/reject_extent.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

Diagnostic:

```
tdls adaptors: vector extent does not match the system dimension
```
