# GPU benchmark harness of the TDLS solvers: the measurement engine
# behind the tuning work (issues #7 and #8). Strictly opt-in, exactly
# like the GPU examples: nothing here is reached unless
# TDLS_BUILD_CUDA_BENCHMARKS or TDLS_BUILD_HIP_BENCHMARKS is ON, the
# two are mutually exclusive (single sources compiled as one
# language), and a requested toolchain that cannot be found fails the
# configure loudly.
#
# One executable per (case, backend). The variant grid is sharded into
# one generated translation unit per system dimension, so template
# instantiation parallelizes across compiler jobs and TDLS_BENCH_DIMS
# can build any subset:
#   -DTDLS_BENCH_DIMS=quick   1 4 8 16 32 64 96 128        (default)
#   -DTDLS_BENCH_DIMS=full    2..32 plus 48 64 80 96 112 128
#   -DTDLS_BENCH_DIMS="12;13" any explicit semicolon list

if(TDLS_BUILD_CUDA_BENCHMARKS AND TDLS_BUILD_HIP_BENCHMARKS)
    message(FATAL_ERROR "tdls: TDLS_BUILD_CUDA_BENCHMARKS and TDLS_BUILD_HIP_BENCHMARKS are "
                        "mutually exclusive: the benchmark sources are single sources "
                        "compiled as one language")
endif()
if(TDLS_BUILD_CUDA_BENCHMARKS)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "tdls: TDLS_BUILD_CUDA_BENCHMARKS is ON but no CUDA compiler was "
                            "found; set CMAKE_CUDA_COMPILER or fix the PATH")
    endif()
    enable_language(CUDA)
    set(TDLS_BENCH_GPU_LANGUAGE CUDA)
else()
    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR "tdls: TDLS_BUILD_HIP_BENCHMARKS is ON but no HIP compiler was "
                            "found; set CMAKE_HIP_COMPILER or fix the PATH")
    endif()
    enable_language(HIP)
    set(TDLS_BENCH_GPU_LANGUAGE HIP)
endif()

string(TOLOWER "${TDLS_BENCH_GPU_LANGUAGE}" TDLS_BENCH_BACKEND)

# The dimension list of the build.
set(TDLS_BENCH_DIMS "quick" CACHE STRING
    "Benchmark dimensions: 'quick', 'full', or an explicit semicolon list")
if(TDLS_BENCH_DIMS STREQUAL "quick")
    set(_tdls_bench_dims 1 4 8 16 32 64 96 128)
elseif(TDLS_BENCH_DIMS STREQUAL "full")
    set(_tdls_bench_dims 48 64 80 96 112 128)
    foreach(_dim RANGE 2 32)
        list(APPEND _tdls_bench_dims ${_dim})
    endforeach()
else()
    set(_tdls_bench_dims ${TDLS_BENCH_DIMS})
endif()
list(REMOVE_DUPLICATES _tdls_bench_dims)
list(SORT _tdls_bench_dims COMPARE NATURAL)

# Source revision stamped into every CSV row.
find_package(Git QUIET)
set(TDLS_BENCH_GIT_COMMIT "unknown")
if(Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE _tdls_bench_commit
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_tdls_bench_commit)
        set(TDLS_BENCH_GIT_COMMIT "${_tdls_bench_commit}")
    endif()
endif()

# One generated translation unit per dimension (build tree only).
set(_tdls_bench_dim_sources "")
foreach(TDLS_BENCH_DIM IN LISTS _tdls_bench_dims)
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated/pure_lupp_d${TDLS_BENCH_DIM}.cu")
    configure_file(cases/pure_lupp/dims/dim.cu.in "${_generated}" @ONLY)
    list(APPEND _tdls_bench_dim_sources "${_generated}")
endforeach()

add_executable(tdls_bench_purelupp_${TDLS_BENCH_BACKEND}
               cases/pure_lupp/main.cu ${_tdls_bench_dim_sources})
if(TDLS_BENCH_GPU_LANGUAGE STREQUAL "HIP")
    set_source_files_properties(cases/pure_lupp/main.cu ${_tdls_bench_dim_sources}
                                PROPERTIES LANGUAGE HIP)
endif()
target_link_libraries(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PRIVATE tdls::tdls)
target_include_directories(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cases"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples/common")
set_target_properties(tdls_bench_purelupp_${TDLS_BENCH_BACKEND}
                      PROPERTIES CUDA_STANDARD 20 HIP_STANDARD 20)
target_compile_definitions(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PRIVATE
    TDLS_BENCH_GIT_COMMIT="${TDLS_BENCH_GIT_COMMIT}")

# When the test tree is enabled too, its buildtests/check targets must
# also build this executable, or `make check` would run the smoke test
# below against a missing binary.
if(TARGET buildtests)
    add_dependencies(buildtests tdls_bench_purelupp_${TDLS_BENCH_BACKEND})
endif()

# Smoke test: a tiny batch through one cheap variant per built
# dimension, validating that the harness builds, runs, measures and
# validates. Skipped (77) when no device is present. Full campaigns go
# through scripts/run_sweep.sh, never through ctest.
add_test(NAME benchmarks.pure_lupp.smoke
         COMMAND tdls_bench_purelupp_${TDLS_BENCH_BACKEND} --filter "static/n[0-9]+/ts3/rl/u1/reg"
                 --batch 512 --runs 2 --distribution default --validate 128 --parity 8)
set_tests_properties(benchmarks.pure_lupp.smoke
                     PROPERTIES LABELS "benchmarks" SKIP_RETURN_CODE 77)
