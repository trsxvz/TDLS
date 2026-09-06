# TFEL interoperability

The adaptors of `tdls/tfel/adaptors.hpp` are designed for the
`tfel::math` objects. Their free functions accept them directly and
infer everything the raw interface needs: the scalar type, the
dimension (at compile time or at run time), the strides and, on the
fixed-size path, the residency booleans.

The detection is purely structural: TDLS names and includes nothing
from `TFEL`, so the library stays dependency-free, and any type
following the `TFEL` protocol is accepted. Three shapes are recognized:

- contiguous fixed-size objects and views, exposing `data()` and an
  `indexing_policy` type with constexpr extents (this matches
  `tfel::math::tmatrix`, `tfel::math::tvector` and `tfel::math::View`,
  among others): they resolve the compile-time TiledLUpp solver;
- element-strided views, exposing their runtime stride either through
  a `data()` member returning a (pointer, stride) pair (this matches
  `tfel::math::StridedCoalescedView`) or through a `stride()` /
  `getStride()` member next to a plain `data()`;
- runtime-sized objects, whose indexing policy carries its extents as
  data members and reports a zero extent when default-constructed
  (this matches `tfel::math::matrix` and `tfel::math::vector`). They
  resolve the runtime TiledLUpp solver, the dimension read from the
  object. The extents cannot be checked at compile time on this path:
  the matrix must be square and every vector extent must match its
  dimension, as everywhere in the raw API.

The elements of a `tfel::math::ViewsArray` (the result of `map_array`)
are themselves `View` objects and are accepted individually.

Residencies are inferred per argument (contiguous object: internal;
strided view: external) and can be mixed freely in one call. Two
shapes are rejected at compile time with an explicit message:

- row-strided sub-matrix views, whose row stride cannot be expressed
  by the single-stride addressing: the result of
  `tfel::math::tmatrix::submatrix_view()`, or of
  `tfel::math::map_derivative()`, which maps a derivative block inside
  a larger jacobian;
- gather views holding one pointer per component (this matches
  `tfel::math::CoalescedView`, built by the `map()` overload taking an
  array of pointers), which expose no `data()` at all.

`TFEL` stores matrices row-major and the adaptors follow that
convention: a configuration selecting the column-major layout is
rejected at compile time.

The six operations of the raw interface keep their names: `factorize`,
`solve`, `solve_inplace`, `substitute`, `substitute_inplace` and
`substitute_canonical`. The pivot is an `int` pointer or array, or a
dense `int` object. The factorizing entry points return `false` on a
singular matrix, as in the raw interface. The out-of-tile counter is
available here too: `factorize`, `solve` and `solve_inplace` take an
optional trailing `int&`.

For other types the detection can be overridden by specializing
`tdls::storage_traits`.

Each solver family has its TFEL snippets, one per object kind and per
entry point: {doc}`TiledLUpp snippets <tiled_lupp/index>`.

```{toctree}
:maxdepth: 2
:hidden:

tiled_lupp/index
```
