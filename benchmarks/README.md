# TDLS benchmark harness

The measurement engine of the TDLS tuning work: it sweeps the TiledLUpp
parameter space (and, later, external comparison solvers) over batches
of independent systems, validates every measurement against the
independent reference LU of the test suites, and emits self-sufficient
CSV rows. GPU-only for now; the CPU side and the applicative
(constitutive-law) case are planned extensions.

## Building

Strictly opt-in, like the GPU examples: nothing is probed unless one
backend is requested, and the requested toolchain becomes mandatory.

```sh
cmake --preset bench-cuda        # or bench-hip; bench-cuda-dev builds
cmake --build build-bench -j     # a single dimension, for iterating
```

The presets are shorthands over two plain options:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTDLS_BUILD_CUDA_BENCHMARKS=ON
cmake --build build -j
```

`TDLS_BUILD_HIP_BENCHMARKS` selects the HIP toolchain instead (the two
are mutually exclusive). `TDLS_BENCH_DIMS` selects the system
dimensions compiled into the binary: `quick` (default:
1 4 8 16 32 64 96 128), `full` (every dimension of [2, 32] plus
48 64 80 96 112 128), or any explicit list (`-DTDLS_BENCH_DIMS="12;13"`,
handy for fast iteration). The grid is sharded into one translation
unit per dimension, so compilation parallelizes and large dimensions
only cost when requested.

## Running

The binary is self-sufficient:

```sh
./build/benchmarks/tdls_bench_purelupp_cuda --list                # the variant tags
./build/benchmarks/tdls_bench_purelupp_cuda --meta                # device/build identity
./build/benchmarks/tdls_bench_purelupp_cuda \
    --filter '^purelupp/static/n12/' --csv results.csv            # measure a slice
```

Protocol: one untimed warmup then 5 timed runs per variant (event
timing, every run on pristine inputs), under two input distributions
(`default`, entries in [-0.5, 0.5]; `stress`, entries in
[-1.4e-10, 1.4e-10], just above the pivot acceptance threshold, so
nearly every system fires the out-of-tile pivoting on about a third
of its columns). Every row
carries the backward error of a solution sample, the verdict parity of
a smaller sample against the reference LU, the out-of-tile statistics,
and the kernel metrics the runtime API self-reports (registers, local
memory - the spill footprint - static shared memory, theoretical
occupancy, wave fill). No external profiler is involved; to profile
one variant in depth, rerun it in isolation:

```sh
ncu ./build/benchmarks/tdls_bench_purelupp_cuda \
    --filter '^purelupp/static/n12/ts3/rl/u1/reg$' --runs 1
```

## Campaigns

`scripts/run_sweep.sh` supervises a full sweep: watchdog timeout per
variant, resume after interruption, campaign index. Bash + coreutils
only.

```sh
benchmarks/scripts/run_sweep.sh ./build/benchmarks/tdls_bench_purelupp_cuda \
    my_results -- --distribution both
```

It produces `_meta.txt`, `_index.tsv` and `results.csv` in the output
directory (default `results/<timestamp>`, git-ignored). Rerunning with
the same directory skips what already succeeded. For stable numbers on
a desktop GPU, lock the clocks first (`nvidia-smi -lgc <MHz>`).

## Layout

```
common/     backend-free host infrastructure: options, registry,
            record/CSV schema, statistics, validation
dispatch/   one file per backend (cuda_or_hip covers both toolchains
            through the single-source dialect of examples/common)
cases/      one directory per benchmark case; pure_lupp sweeps
            standalone batched solves
external/   (planned) opt-in comparison solvers, one directory each
scripts/    campaign supervisor (bash) and, later, offline plotting
            helpers (python, never in the measurement path)
```

Adding a backend = one dispatch file plus one variant file registering
into the same registry; adding a case = one directory under `cases/`;
external solvers will register into the same registry behind
individual CMake options, so downstream handling has no special cases.
