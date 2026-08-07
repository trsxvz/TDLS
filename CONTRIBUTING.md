# Contributing to TDLS

## Reporting bugs

Open an issue on the [issue tracker](https://github.com/trsxvz/TDLS/issues).
Please include the TDLS version or commit identifier, the compiler and
its version, and a minimal reproducer when possible.

## Proposing changes

Contributions are welcome through pull requests. Before submitting,
format the code with the repository `.clang-format` and make sure the
test suite passes:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

By contributing, you agree that your contributions are licensed under
the BSD 3-Clause License of the project (see the LICENSE file).

## Getting help

See [SUPPORT.md](SUPPORT.md).
