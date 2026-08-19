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
handy for fast iteration). The grid enumeration lives in
`cases/pure_lupp/generate_variants.cmake`, which emits ONE generated
translation unit PER VARIANT into the build tree (never tracked):
compilation parallelizes across every variant, incremental builds
stay exact (a unit is only touched when its content changes, header
edits propagate through depfiles, and units of variants that left the
grid are removed), and the compile time of every variant becomes a
recorded dataset (below). `-G Ninja` is a comfortable option for
large campaign builds, not a requirement.

## Running

The binary is self-sufficient:

```sh
./build/benchmarks/tdls_bench_purelupp_cuda --list                # the variant tags
./build/benchmarks/tdls_bench_purelupp_cuda --meta                # device/build identity
./build/benchmarks/tdls_bench_purelupp_cuda \
    --filter '^purelupp/static/f64/n12/' --csv results.csv        # measure a slice
```

Every variant measures `solve_inplace`, the production entry point
(fused forward substitution, one right-hand-side buffer). The sweep
crosses the solver axes (static/dynamic, float/double, tile size 1 to
6, both schedules, the unroll knob) with the operand placements: the
matrix in `reg` (thread-local), `dram` (the SoA batch in device
memory) or `shm` (staged in shared memory), the rhs and pivot
following the matrix or pinned thread-local through the `rreg` /
`preg` tags. Batched storage is SoA throughout. The threads-per-block of every variant is selected
by the O+W heuristic (occupancy plus wave fill, candidates
32/64/96/128/256, sub-warp candidates when a shared footprint exceeds
the budget of a full warp); `--ntpb` forces a value. Variants that
cannot run are recorded as rows with a non-ok `status`, never as
crashes: `skip_smem` (shared footprint over the per-block budget even
at one thread per block), `skip_dram` (device memory, checked
predictively against the free memory and reactively at every
allocation), `skip_host`, `skip_offset32` (batch beyond the 32-bit
addressing range of the solvers at that dimension), `error_launch`
(the launch probe was rejected by the device).

The input batches are materialized in place on the device by a
counter-based generator (`common/batch.hpp`, SplitMix64): no host
batch, no host-to-device transfer, and the in-place restores between
timed runs are device-side regenerations. The validation regenerates
its sampled systems on the host with the same function, so any
host/device divergence of the generator fails the parity check.

Protocol: one untimed warmup then 5 timed runs per variant (event
timing, every run on pristine inputs), under two input distributions
(`default`, entries in [-0.5, 0.5]; `stress`, entries within 1.4x
the pivot acceptance threshold of the scalar type, so nearly every
system fires the out-of-tile pivoting on about a third of its
columns). Every row
carries the backward error of a solution sample, the verdict parity of
a smaller sample against the reference LU, the out-of-tile statistics,
and the kernel metrics the runtime API self-reports (registers, local
memory - the spill footprint - static shared memory, theoretical
occupancy, wave fill). No external profiler is involved; to profile
one variant in depth, rerun it in isolation:

```sh
ncu ./build/benchmarks/tdls_bench_purelupp_cuda \
    --filter '^purelupp/static/f64/n12/ts3/rl/u1/reg$' --runs 1
```

## Compile times

The build itself is a measurement: a compiler launcher times the
compilation of every variant translation unit and appends
`tag,compile_s,status` to `compile_times.csv` in the build tree (the
log is append-only across rebuilds; the last row of a tag wins). This
dataset joins `results.csv` by tag, so the tuning analysis can weigh
performance against compile cost. Setting
`TDLS_BENCH_COMPILE_TIMEOUT=<seconds>` at build time turns a
too-expensive compilation into a recorded `compile_timeout` verdict:
the unit is replaced by a stub, the build and the link still succeed,
and the variant is simply absent from the binary.

## Campaigns

`scripts/run_sweep.sh` supervises a full sweep: watchdog timeout per
variant, resume after interruption, campaign index. Bash + coreutils
only.

```sh
benchmarks/scripts/run_sweep.sh ./build/benchmarks/tdls_bench_purelupp_cuda \
    my_results -- --distribution both
```

It produces `_meta.txt`, `_index.tsv`, `results.csv` and a copy of
`compile_times.csv` in the output directory (default
`results/<timestamp>`, git-ignored). Rerunning with
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
