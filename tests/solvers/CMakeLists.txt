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
        # In the Release config the suites compile at TDLS_TESTS_OPTIMIZATION
        # (default -O3, see ../CMakeLists.txt; it trails the Release flags
        # and wins). The fully-unrolled solver templates expand into large
        # functions whose high-level optimisation dominates compile time
        # (test_cross_full alone: ~20s at -O3 vs ~6s at -O1 under g++), so
        # the flag offers -O1 for quick local iterations. Only in Release,
        # so that a Debug build stays steppable.
        if(NOT MSVC)
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Release>:${TDLS_TESTS_OPTIMIZATION}> -Wall -Wextra -Wpedantic -Werror
                # The harness derives unique test-case ids from __COUNTER__, a
                # ubiquitous extension (GCC/Clang/MSVC; standard in C2y). Clang
                # >=22 flags it under -Wc2y-extensions, which -Werror makes
                # fatal; suppress just that diagnostic where the compiler knows
                # the flag (probed in ../CMakeLists.txt). It trails -Wpedantic
                # so it wins.
                $<$<BOOL:${TDLS_HAVE_WNO_C2Y_EXTENSIONS}>:-Wno-c2y-extensions>
                # The suites compare code paths bit for bit (static against
                # dynamic, fused against split, unrolled against rolled...),
                # which a compiler may contract into fused multiply-adds
                # differently (GCC does at -O3 -march=native): no contraction.
                $<$<BOOL:${TDLS_HAVE_FFP_CONTRACT_OFF}>:-ffp-contract=off>)
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

# Run-time contract tests: each program must stop with its message
# (expect_stop.cmake runs it, since a program stopped by a signal fails a
# plain ctest entry whatever its output).
#
# tdls_add_stop_cases(FAMILY <family> DIRECTORY <directory>
#                     CASES <case>... MESSAGES <message>...)
# Builds <directory>/<case>.cpp into tdls_test_<family>_<case> and
# registers it as the ctest entry solvers.<family>.<case>.
function(tdls_add_stop_cases)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "FAMILY;DIRECTORY" "CASES;MESSAGES")
    foreach(case message IN ZIP_LISTS ARG_CASES ARG_MESSAGES)
        set(target tdls_test_${ARG_FAMILY}_${case})
        add_executable(${target} EXCLUDE_FROM_ALL ${ARG_DIRECTORY}/${case}.cpp)
        add_dependencies(buildtests ${target})
        target_link_libraries(${target} PRIVATE tdls::tdls)
        target_compile_features(${target} PRIVATE cxx_std_20)
        if(NOT MSVC)
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME solvers.${ARG_FAMILY}.${case}
                 COMMAND ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:${target}>
                         "-DMESSAGE=${message}" -P
                         ${CMAKE_CURRENT_SOURCE_DIR}/expect_stop.cmake)
        set_tests_properties(solvers.${ARG_FAMILY}.${case} PROPERTIES LABELS "solvers")
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
        wide_offsets
        constexpr)

tdls_add_reject_cases(
    FAMILY tiledlupp
    DIRECTORY tiled_lupp
    CASES
        reject_config_floor_above_threshold
        reject_config_zero_floor
        reject_config_pivot_threshold_zero
        reject_config_pivot_threshold_above_one
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
        "TiledLUppSolverStatic: relative_pivot_threshold must be in"
        "TiledLUppSolverDynamic: relative_pivot_threshold must be in"
        "A must be a dense object"
        "A must not be const here"
        "the pivot must be mutable here"
        "the config scalar type must match the matrix element type"
        "dense objects are addressed row-major"
        "b and x extents do not match"
        "B and X must have N rows and the same column count"
        "the pivot must be vector-like")

tdls_add_solver_suites(
    FAMILY arrowheadlupp
    DIRECTORY arrowhead_lupp
    SUITES
        oracle_static
        oracle_dynamic
        second_pass
        static_vs_dynamic
        storages
        residencies
        layouts
        entry_points
        config_knobs
        wide_offsets
        constexpr
        multirhs
        streamed)

tdls_add_reject_cases(
    FAMILY arrowheadlupp
    DIRECTORY arrowhead_lupp
    CASES
        reject_shape_empty
        reject_shape_dense_coupling_row
        reject_shape_diagonal_rows
        reject_shape_bsr_block
        reject_config_condensation_threshold
        reject_config_condensation_threshold_dynamic
        reject_config_pass_width
        reject_multirhs_none
        reject_canonical_beyond_border
        reject_streamed_canonical_beyond_border
        reject_streamed_tangent_beyond_border
    DIAGNOSTICS
        "border, block and blocks must be >= 1"
        "coupling_row applies to the diagonal coupling only"
        "the diagonal coupling rows exceed the border"
        "the Bsr storage needs a border"
        "ArrowheadLUppSolverStatic: condensation_pivot_threshold must be in"
        "ArrowheadLUppSolverDynamic: condensation_pivot_threshold must be in"
        "rhs_pass_width must be in"
        "ArrowheadLUppSolverStatic: nrhs must be >= 1"
        "ArrowheadLUppSolverStatic: nrhs must be in"
        "ArrowheadLUppSolverStatic: nrhs must be in"
        "ArrowheadLUppSolverStatic: ntan must be in")

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
        auto_sync
        wide_offsets
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
        reject_nosync_group
    DIAGNOSTICS
        "singular_floor must be positive"
        "rows_per_thread must be >= 1"
        "CooperativeLUppSolverStatic: relative_pivot_threshold must be in"
        "CooperativeLUppSolverDynamic: relative_pivot_threshold must be in"
        "a group of several threads cannot run with tdls::NoSync")

tdls_add_stop_cases(
    FAMILY cooperativelupp
    DIRECTORY cooperative_lupp
    CASES
        stop_nosync_runtime
        stop_dirty_workspace
    MESSAGES
        "the runtime solver runs with tdls::NoSync"
        "was not zero before its first use")

# Device suite of TiledLUppSolverDynamic with dimensions known at compile
# time (issue #16), built with the toolchain of the GPU examples
# (TDLS_BUILD_CUDA_EXAMPLES or TDLS_BUILD_HIP_EXAMPLES); it returns 77,
# a skipped test, without a device.
if(TDLS_BUILD_CUDA_EXAMPLES OR TDLS_BUILD_HIP_EXAMPLES)
    set(target tdls_test_tiledlupp_device_constant_dimension)
    if(TDLS_BUILD_CUDA_EXAMPLES)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
    else()
        enable_language(HIP)
        set_source_files_properties(tiled_lupp/test_device_constant_dimension.cu
                                    PROPERTIES LANGUAGE HIP)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL tiled_lupp/test_device_constant_dimension.cu)
    add_dependencies(buildtests ${target})
    target_link_libraries(${target} PRIVATE tdls::tdls)
    target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/examples/common")
    if(TDLS_BUILD_CUDA_EXAMPLES AND CMAKE_VERSION VERSION_LESS 3.26)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
    else()
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                                   HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
    endif()
    add_test(NAME solvers.tiledlupp.device_constant_dimension COMMAND ${target})
    set_tests_properties(solvers.tiledlupp.device_constant_dimension
                         PROPERTIES LABELS "solvers" SKIP_RETURN_CODE 77)
endif()
