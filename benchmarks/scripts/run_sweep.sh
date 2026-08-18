#!/usr/bin/env bash
# Campaign supervisor of the TDLS benchmark harness.
#
# The benchmark binaries are self-sufficient (enumeration, measurement,
# validation, CSV); this script only adds what a process cannot give
# itself: a watchdog timeout per variant, resumability after a crash or
# an interruption, and the campaign index. Pure bash + coreutils, no
# other dependency, by design.
#
# Usage:
#   run_sweep.sh <bench-binary> [output-dir] [-- <extra binary args>]
#
#   output-dir defaults to results/<UTC timestamp>. Extra args after
#   "--" are passed to every binary invocation (e.g. --distribution
#   default, --batch 20000, --filter <regex>).
#
# Environment:
#   TDLS_BENCH_TIMEOUT  watchdog seconds per variant (default 900)
#
# Outputs, in the output directory:
#   _meta.txt    campaign metadata (device, host, date, revision)
#   _index.tsv   one line per finished variant: tag, rc, duration
#   results.csv  the measurement rows, appended by the binary
#
# Rerunning with the same output directory resumes: variants already
# indexed with rc 0 are skipped.

set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <bench-binary> [output-dir] [-- <extra binary args>]"
    exit 2
fi

binary=$1
shift
outdir="results/$(date -u +%Y%m%dT%H%M%SZ)"
if [ $# -ge 1 ] && [ "$1" != "--" ]; then
    outdir=$1
    shift
fi
if [ $# -ge 1 ] && [ "$1" = "--" ]; then
    shift
fi
extra=("$@")
timeout_s=${TDLS_BENCH_TIMEOUT:-900}

if [ ! -x "$binary" ]; then
    echo "error: $binary is not an executable"
    exit 2
fi

mkdir -p "$outdir"
index="$outdir/_index.tsv"
csv="$outdir/results.csv"
touch "$index"

# Campaign metadata: the binary reports the device and build, the
# supervisor adds the campaign context.
{
    echo "# tdls benchmark campaign"
    echo "start_utc : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host      : $(hostname)"
    echo "binary    : $binary"
    echo "timeout_s : $timeout_s"
    echo "extra_args: ${extra[*]:-}"
    "$binary" --meta
} > "$outdir/_meta.txt"

ran=0
skipped=0
failed=0
while IFS= read -r tag; do
    if awk -F'\t' -v t="$tag" '$1 == t && $2 == 0 { found = 1 } END { exit !found }' \
        "$index"; then
        skipped=$((skipped + 1))
        continue
    fi
    start=$(date +%s)
    # The per-tag filter comes last: the option parser keeps the last
    # occurrence, so it overrides any campaign-scoping --filter passed
    # through the extra arguments (which --list, above, does honor).
    timeout "$timeout_s" "$binary" --csv "$csv" "${extra[@]}" --filter "^${tag}\$"
    rc=$?
    duration=$(( $(date +%s) - start ))
    printf '%s\t%s\t%s\n' "$tag" "$rc" "$duration" >> "$index"
    ran=$((ran + 1))
    if [ "$rc" -ne 0 ]; then
        failed=$((failed + 1))
        echo "warning: $tag exited with $rc" >&2
    fi
done < <("$binary" --list "${extra[@]}")

# The compile-time dataset of the build joins the measurements by tag.
compile_log="$(dirname "$binary")/compile_times.csv"
if [ -f "$compile_log" ]; then
    cp "$compile_log" "$outdir/compile_times.csv"
fi

echo "end_utc   : $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$outdir/_meta.txt"
echo "campaign done: $ran run, $skipped already done, $failed failed ($outdir)"
[ "$failed" -eq 0 ]
