# Vanilla test suites of the solver families: one executable per suite, each
# registered in ctest under the "solvers" label. The suites use the
# homemade harness of tests/common (no external test framework) and may
# use C++20, which is also the floor of the library.

# The suites of the CooperativeLUpp family run the groups of threads of
# the solver on CPU threads.
find_package(Threads REQUIRED)

# tdls_add_solver_suites(FAMILY <family> DIRECTORY <directory>
#                        SUITES <suite>... [LIBRARIES <library>...])
# Builds <directory>/test_<suite>.cpp into tdls_test_<family>_<suite> and
# registers it as the ctest entry solvers.<family>.<suite>.
function(tdls_add_solver_suites)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "FAMILY;DIRECTORY" "SUITES;LIBRARIES")
    foreach(suite IN LISTS ARG_SUITES)
        set(target tdls_test_${ARG_FAMILY}_${suite})
        add_executable(${target} EXCLUDE_FROM_ALL ${ARG_DIRECTORY}/test_${suite}.cpp)
        add_dependencies(buildtests ${target})
        target_link_libraries(${target} PRIVATE tdls::tdls ${ARG_LIBRARIES})
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
        target_compile_features(${target} PRIVATE cxx_std_20)
        # These are correctness tests, not benchmarks. In the Release config
        # the fully-unrolled solver templates expand into large functions
        # whose high-level optimisation dominates compile time
        # (test_cross_full alone: ~20s at -O3 vs ~6s at -O1 under g++),
        # while the bitwise and backward-error checks are unaffected by the
        # level. The suites therefore compile at TDLS_TESTS_OPTIMIZATION
        # (default -O1; it trails the Release -O3 and wins), but only in
        # Release, so a Debug build stays steppable.
        if(NOT MSVC)
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Release>:${TDLS_TESTS_OPTIMIZATION}> -Wall -Wextra -Wpedantic -Werror
                # The harness derives unique test-case ids from __COUNTER__, a
                # ubiquitous extension (GCC/Clang/MSVC; standard in C2y). Clang
                # >=22 flags it under -Wc2y-extensions, which -Werror makes
                # fatal; suppress just that diagnostic where the compiler knows
                # the flag (probed in ../CMakeLists.txt). It trails -Wpedantic
                # so it wins.
                $<$<BOOL:${TDLS_HAVE_WNO_C2Y_EXTENSIONS}>:-Wno-c2y-extensions>)
        endif()
        if(MSVC)
            # Template-heavy suites overflow the default COFF section limit.
            target_compile_options(${target} PRIVATE /bigobj)
        endif()
        add_test(NAME solvers.${ARG_FAMILY}.${suite} COMMAND ${target})
        set_tests_properties(solvers.${ARG_FAMILY}.${suite} PROPERTIES LABELS "solvers")
    endforeach()
endfunction()

# Negative compilation tests of the compile-time contracts: each
# target must fail to build with its contract diagnostic. ctest
# builds them on demand and passes only when the diagnostic appears,
# so a contract silently dropped turns these tests red.
#
# tdls_add_reject_cases(FAMILY <family> DIRECTORY <directory>
#                       CASES <case>... DIAGNOSTICS <diagnostic>...)
# Builds <directory>/<case>.cpp into tdls_test_<family>_<case> and
# registers the ctest entry solvers.<family>.<case>, which expects the
# diagnostic of the same rank.
function(tdls_add_reject_cases)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "FAMILY;DIRECTORY" "CASES;DIAGNOSTICS")
    foreach(case diagnostic IN ZIP_LISTS ARG_CASES ARG_DIAGNOSTICS)
        set(target tdls_test_${ARG_FAMILY}_${case})
        add_executable(${target} EXCLUDE_FROM_ALL ${ARG_DIRECTORY}/${case}.cpp)
        target_link_libraries(${target} PRIVATE tdls::tdls)
        target_compile_features(${target} PRIVATE cxx_std_20)
        add_test(NAME solvers.${ARG_FAMILY}.${case}
                 COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
                         ${target} --config $<CONFIG>)
        # The reject tests build inside the shared binary tree: serialize
        # them so concurrent ctest -j runs cannot collide on the build lock.
        set_tests_properties(
            solvers.${ARG_FAMILY}.${case}
            PROPERTIES LABELS "solvers" PASS_REGULAR_EXPRESSION "${diagnostic}" RESOURCE_LOCK
                       build_tree)
    endforeach()
endfunction()

tdls_add_solver_suites(
    FAMILY tiledlupp
    DIRECTORY tiled_lupp
    SUITES
        oracle_static
        oracle_dynamic
        static_vs_dynamic
        residencies
        layouts
        colmajor
        entry_points
        inplace_paths
        multirhs
        tile_sizes
        singular
        config_knobs
        adaptors
        cross_full
        dynamic_edges
        constexpr)

tdls_add_reject_cases(
    FAMILY tiledlupp
    DIRECTORY tiled_lupp
    CASES
        reject_config_floor_above_threshold
        reject_config_zero_floor
        reject_adaptors_gather_view
        reject_adaptors_const_matrix
        reject_adaptors_const_pivot
        reject_adaptors_config_scalar_mismatch
        reject_adaptors_colmajor_config
        reject_adaptors_rhs_extent_mismatch
        reject_adaptors_multirhs_extent_mismatch
        reject_adaptors_pivot_extent_mismatch
    DIAGNOSTICS
        "singular_floor must not exceed oot_pivot_threshold"
        "singular_floor must be positive"
        "A must be a dense object"
        "A must not be const here"
        "the pivot must be mutable here"
        "the config scalar type must match the matrix element type"
        "dense objects are addressed row-major"
        "b and x extents do not match"
        "B and X must have N rows and the same column count"
        "the pivot must be vector-like")

tdls_add_solver_suites(
    FAMILY cooperativelupp
    DIRECTORY cooperative_lupp
    SUITES
        oracle_static
        oracle_dynamic
        static_vs_dynamic
        rows_per_thread
        entry_points
        residencies
        layouts
        singular
        config_knobs
        row_interchange
        constexpr
    LIBRARIES Threads::Threads)

tdls_add_reject_cases(
    FAMILY cooperativelupp
    DIRECTORY cooperative_lupp
    CASES
        reject_config_zero_floor
        reject_config_rows_per_thread
        reject_config_pivot_threshold_zero
        reject_config_pivot_threshold_above_one
        reject_missing_barrier
    DIAGNOSTICS
        "singular_floor must be positive"
        "rows_per_thread must be >= 1"
        "CooperativeLUppSolverStatic: relative_pivot_threshold must be in"
        "CooperativeLUppSolverDynamic: relative_pivot_threshold must be in"
        "a group of several threads needs a barrier")
