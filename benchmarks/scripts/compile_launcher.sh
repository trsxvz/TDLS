#!/usr/bin/env bash
# Compiler launcher of the benchmark harness (CMake
# <LANG>_COMPILER_LAUNCHER). Times the compilation of every generated
# variant translation unit and appends one line per compile to the log
# given as first argument:
#
#     tag,compile_s,status        status: ok | compile_timeout
#
# The tag is derived from the translation-unit stem (underscores map
# one-to-one onto the slashes of the runtime tag), so this log joins
# the measurement CSV directly. The log is append-only: rebuilds add
# rows, the last row of a tag wins. Non-variant sources (main.cu) pass
# through untimed.
#
# When TDLS_BENCH_COMPILE_TIMEOUT (seconds) is set, a compilation
# exceeding it is killed and replaced by a stub object: the build and
# the link still succeed, the variant is simply absent from the binary
# (hence from --list), and the log records compile_timeout. A compile
# cost beyond the threshold is a recorded verdict, not a broken build.

set -u

log=$1
shift

# Locate the source and object of this invocation.
src=""
obj=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "-c" ]; then src=$arg; fi
    if [ "$prev" = "-o" ]; then obj=$arg; fi
    prev=$arg
done

stem=$(basename "$src" .cu)
case $stem in
purelupp_*) ;;
*)
    exec "$@"
    ;;
esac
tag=$(printf '%s' "$stem" | tr '_' '/')

start=$(date +%s%N)
if [ -n "${TDLS_BENCH_COMPILE_TIMEOUT:-}" ]; then
    timeout "$TDLS_BENCH_COMPILE_TIMEOUT" "$@"
    rc=$?
else
    "$@"
    rc=$?
fi
# LC_ALL=C pins the decimal separator: the log is a CSV, a locale
# decimal comma would split the seconds cell in two.
duration=$(LC_ALL=C awk -v ns=$(($(date +%s%N) - start)) 'BEGIN { printf "%.2f", ns / 1e9 }')

status=ok
if [ "$rc" -eq 124 ] && [ -n "${TDLS_BENCH_COMPILE_TIMEOUT:-}" ]; then
    # Timed out: substitute a stub so the link succeeds without the
    # variant, and record the verdict.
    status=compile_timeout
    stub=$(mktemp --suffix=.cu)
    printf '// stub: variant compilation exceeded TDLS_BENCH_COMPILE_TIMEOUT\n' > "$stub"
    "$1" -c "$stub" -o "$obj"
    rc=$?
    rm -f "$stub"
elif [ "$rc" -ne 0 ]; then
    # Real compilation failure: propagate without logging a verdict.
    exit "$rc"
fi

# The header row is written at configure time (CMakeLists.txt): with
# parallel compile jobs, every launcher of the first batch would see
# an empty log and write its own. One short appended line is atomic.
printf '%s,%s,%s\n' "$tag" "$duration" "$status" >> "$log"
exit "$rc"
