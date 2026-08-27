# The test tree is split by target space, each tagging its ctest
# entries with a label so they run independently (ctest -L <label>):
# tests/common validates the shared test infrastructure ("common");
# tests/solvers validates the library itself ("solvers"); the examples
# register themselves as self-checking tests from examples/
# ("examples"); tests/benchmarks will validate the benchmark harness
# once that part exists.

# Optimization flag of the test suites in the Release configuration.
# The suites are correctness tests whose checks do not depend on the
# level, while the fully-unrolled solver templates dominate compile
# time at high levels, so -O1 is the fast default. The CI also builds
# the suites at -O3 (optimizer coverage) and at -O2 under the
# sanitizers.
set(TDLS_TESTS_OPTIMIZATION "-O1" CACHE STRING
    "Optimization flag of the test suites in the Release configuration")
# Clang >=22 flags the harness's __COUNTER__ (unique test-case ids) under
# -Wc2y-extensions, which the tests' -Werror turns fatal. Suppress just
# that diagnostic - but only where the compiler recognises the flag: older
# Clang and GCC reject the unknown option under -Werror. Probing with
# -Werror makes the check fail (rather than silently pass) when it is absent.
include(CheckCXXCompilerFlag)
set(CMAKE_REQUIRED_FLAGS -Werror)
check_cxx_compiler_flag(-Wno-c2y-extensions TDLS_HAVE_WNO_C2Y_EXTENSIONS)
unset(CMAKE_REQUIRED_FLAGS)

add_subdirectory(common)
add_subdirectory(solvers)
