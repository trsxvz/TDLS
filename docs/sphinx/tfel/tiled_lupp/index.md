# TiledLUpp snippets

The adaptors on real `tfel::math` objects, one snippet per object kind
and per entry point. Each one is a complete program, shown between the
two markers of its source under `docs/snippets/tfel/`, and checked at
the end. They are built with `TDLS_BUILD_TFEL_SNIPPETS`, TFEL found by
`find_package(TFELMath)`, and run under the `snippets` label.

The objects are declared in the shown code, since their types are the
point. The default configuration applies unless the snippet says
otherwise: tiles of 3, so the 4 x 4 systems have one full tile and a
trailing tile of 1.

```{toctree}
:maxdepth: 1

fixed_size
runtime_sized
views
blocks
rejected
```
