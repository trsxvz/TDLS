# Snippets

One snippet per entry point or knob. Each one is a complete program:
the system it solves, then the code between the two markers of its
source under `docs/snippets/tiled_lupp/`. ctest runs them all under
the `snippets` label.

Above the shown code, the program declares what the formula displays:
the matrix as a row-major `double` array, the right-hand sides, and
the expected solution. Below the shown code, it checks its result.

```{toctree}
:maxdepth: 1

configuration
compile_time
runtime
batches
```
