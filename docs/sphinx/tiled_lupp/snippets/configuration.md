# Configuration

The knobs of `TiledLUppConfig`, one snippet each. Their roles are
listed on the {doc}`family page <../index>`.

## Defaults

The default value, and the two solvers named on it. Tiles of 3 cut
this 4 x 4 matrix into a full tile and a trailing tile of 1.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_defaults.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Tile size

The same 6 x 6 matrix under two tile sizes: one full tile of 4 with a
trailing tile of 2, or a 2 x 2 grid of full tiles of 3.

$$
\left(\begin{array}{cccc|cc}
8 & 1 & 0 & 0 & 2 & 0 \\
1 & 8 & 1 & 0 & 0 & 2 \\
0 & 1 & 8 & 1 & 0 & 0 \\
0 & 0 & 1 & 8 & 1 & 0 \\ \hline
2 & 0 & 0 & 1 & 8 & 1 \\
0 & 2 & 0 & 0 & 1 & 8
\end{array}\right)
\qquad
\left(\begin{array}{ccc|ccc}
8 & 1 & 0 & 0 & 2 & 0 \\
1 & 8 & 1 & 0 & 0 & 2 \\
0 & 1 & 8 & 1 & 0 & 0 \\ \hline
0 & 0 & 1 & 8 & 1 & 0 \\
2 & 0 & 0 & 1 & 8 & 1 \\
0 & 2 & 0 & 0 & 1 & 8
\end{array}\right)
$$

$$
b = \begin{pmatrix} 20 \\ 32 \\ 30 \\ 40 \\ 52 \\ 57 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_tile_size.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A single tile

A tile size equal to the dimension: the whole matrix is one tile.

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

```{literalinclude} ../../../snippets/tiled_lupp/config_single_tile.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A tile larger than the dimension

A tile size above the dimension: the grid is a single partial tile.

$$
A = \begin{pmatrix}
3 & 1 & 0 \\
1 & 4 & 1 \\
0 & 1 & 5
\end{pmatrix}, \quad
b = \begin{pmatrix} 5 \\ 12 \\ 17 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_tile_larger.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Scalar tiles

Tiles of one element: every step handles one pivot.

$$
A = \left(\begin{array}{c|c|c|c}
4 & 1 & 0 & 2 \\ \hline
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_scalar_tiles.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Schedules

The two elimination schedules on the same system.

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

```{literalinclude} ../../../snippets/tiled_lupp/config_schedules.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Pivot thresholds

The last pivot is tiny. The default thresholds accept it, raised ones
declare the matrix singular.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 0 \\
1 & 5 & 0 & 0 \\ \hline
0 & 0 & 6 & 0 \\
0 & 0 & 0 & \textcolor{red}{10^{-8}}
\end{array}\right), \quad
b = \begin{pmatrix} 6 \\ 11 \\ 18 \\ 4 \cdot 10^{-8} \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_thresholds.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Full out-of-tile scan

Column 0 has no acceptable pivot inside its tile. Two rows below can
provide one: the first acceptable, in red, or the largest, in blue.

$$
A = \left(\begin{array}{cc|cc}
10^{-12} & 1 & 0 & 2 \\
2 \cdot 10^{-12} & 5 & 1 & 0 \\ \hline
\textcolor{red}{10^{-6}} & 1 & 6 & 1 \\
\textcolor{blue}{3} & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 10 + 10^{-12} \\ 13 + 2 \cdot 10^{-12} \\ 24 + 10^{-6} \\ 34 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_first_acceptable.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## No forced unrolling

The same solve with and without the unroll pragmas: bitwise identical
factors and solution.

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

```{literalinclude} ../../../snippets/tiled_lupp/config_unroll.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Column-major layout

The matrix stored column by column, as one system of a batch of 3:
the layout knob remaps the flat index, the stride does the rest.

$$
A = \left(\begin{array}{cc|cc}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\ \hline
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{array}\right)
\ \text{stored as}\ 
\begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}^{\mathsf T}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_layout.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Scalar types

The same system in `float` and in `long double`.

$$
A = \left(\begin{array}{ccc|c}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\ \hline
2 & 0 & 1 & 7
\end{array}\right), \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_scalar_types.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## Solver constants

The tile grid and the knobs, read back from the solver type.

$$
A = \left(\begin{array}{cccc|cc}
8 & 1 & 0 & 0 & 2 & 0 \\
1 & 8 & 1 & 0 & 0 & 2 \\
0 & 1 & 8 & 1 & 0 & 0 \\
0 & 0 & 1 & 8 & 1 & 0 \\ \hline
2 & 0 & 0 & 1 & 8 & 1 \\
0 & 2 & 0 & 0 & 1 & 8
\end{array}\right), \quad
b = \begin{pmatrix} 20 \\ 32 \\ 30 \\ 40 \\ 52 \\ 57 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \\ 6 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/tiled_lupp/config_constants.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
