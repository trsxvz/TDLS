# The test tree is split by target space, each tagging its ctest
# entries with a label so they run independently (ctest -L <label>):
# tests/common validates the shared test infrastructure ("common");
# tests/solvers validates the library itself ("solvers"); the examples
# register themselves as self-checking tests from examples/
# ("examples").

# Optimization flag of the test suites in the Release configuration.
# The default -O3 is the level of the Release builds of CMake: the
# suites check the code that the users compile, where the optimizer
# may miscompile it or exploit an undefined behaviour. -O1 builds the
# fully-unrolled solver templates about 3.5 times faster, for quick
# local iterations; the sanitizer job of the CI builds them at -O1.
set(TDLS_TESTS_OPTIMIZATION "-O3" CACHE STRING
    "Optimization flag of the test suites in the Release configuration")
# Clang >=22 flags the harness's __COUNTER__ (unique test-case ids) under
# -Wc2y-extensions, which the tests' -Werror turns fatal. Suppress just
# that diagnostic, but only where the compiler recognises the flag: older
# Clang and GCC reject the unknown option under -Werror. Probing with
# -Werror makes the check fail (rather than silently pass) when it is absent.
include(CheckCXXCompilerFlag)
set(CMAKE_REQUIRED_FLAGS -Werror)
check_cxx_compiler_flag(-Wno-c2y-extensions TDLS_HAVE_WNO_C2Y_EXTENSIONS)
# The streamed ArrowheadLUpp suite compares two code paths bit for bit:
# it turns off the contraction of multiplications and additions where the
# compiler knows the flag (solvers/CMakeLists.txt).
check_cxx_compiler_flag(-ffp-contract=off TDLS_HAVE_FFP_CONTRACT_OFF)
unset(CMAKE_REQUIRED_FLAGS)

add_subdirectory(common)
add_subdirectory(solvers)
