# GPU benchmark harness of the TDLS solvers: the measurement engine
# behind the tuning work (issues #7 and #8). Strictly opt-in, exactly
# like the GPU examples: nothing here is reached unless
# TDLS_BUILD_CUDA_BENCHMARKS or TDLS_BUILD_HIP_BENCHMARKS is ON, the
# two are mutually exclusive (single sources compiled as one
# language), and a requested toolchain that cannot be found fails the
# configure loudly.
#
# One executable per (case, backend). The sweep grid is enumerated in
# cases/pure_lupp/generate_variants.cmake, which emits ONE generated
# translation unit PER VARIANT (git-ignored, build tree only): the
# compile parallelizes across all variants, and the compiler launcher
# records the compile time of every variant into compile_times.csv,
# a first-class dataset of the tuning work (join with the measurement
# CSV by tag). TDLS_BENCH_DIMS selects the dimensions:
#   -DTDLS_BENCH_DIMS=quick   1 4 8 16 32 64 96 128        (default)
#   -DTDLS_BENCH_DIMS=full    2..32 plus 48 64 80 96 112 128
#   -DTDLS_BENCH_DIMS="12;13" any explicit semicolon list
#
# Incremental builds are load-bearing here: the generator only touches
# a translation unit when its content changes (so reconfiguring never
# recompiles), removes units of variants that left the grid, and this
# file registers the generator and its template as configure
# dependencies (so editing them regenerates). Header changes propagate
# through the compiler depfiles as usual.

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

# Grid generation: one translation unit per variant (build tree only).
set(_tdls_bench_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_tdls_bench_manifest "${CMAKE_CURRENT_BINARY_DIR}/variant_sources.txt")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/cases/pure_lupp/variant.cu.in"
            "-DOUTPUT_DIR=${_tdls_bench_generated_dir}"
            "-DMANIFEST=${_tdls_bench_manifest}"
            "-DDIMS=${_tdls_bench_dims}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cases/pure_lupp/generate_variants.cmake"
    RESULT_VARIABLE _tdls_bench_generate_rc)
if(NOT _tdls_bench_generate_rc EQUAL 0)
    message(FATAL_ERROR "tdls: benchmark variant generation failed")
endif()
file(STRINGS "${_tdls_bench_manifest}" _tdls_bench_variant_sources)
# Editing the grid definition or the unit template must rerun the
# generation at the next build.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/cases/pure_lupp/generate_variants.cmake"
             "${CMAKE_CURRENT_SOURCE_DIR}/cases/pure_lupp/variant.cu.in")

# The compile-time log gets its header here, once: the parallel
# launcher instances only append rows (a first-build race would
# otherwise write several headers).
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_times.csv")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compile_times.csv" "tag,compile_s,status\n")
endif()

add_executable(tdls_bench_purelupp_${TDLS_BENCH_BACKEND}
               cases/pure_lupp/main.cu ${_tdls_bench_variant_sources})
if(TDLS_BENCH_GPU_LANGUAGE STREQUAL "HIP")
    set_source_files_properties(cases/pure_lupp/main.cu ${_tdls_bench_variant_sources}
                                PROPERTIES LANGUAGE HIP)
endif()
target_link_libraries(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PRIVATE tdls::tdls)
target_include_directories(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cases"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples/common")
set_target_properties(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PROPERTIES
    CUDA_STANDARD 20
    HIP_STANDARD 20
    # The launcher times every variant compile into compile_times.csv
    # (and, under TDLS_BENCH_COMPILE_TIMEOUT, stubs out the variants
    # whose compilation exceeds the budget).
    ${TDLS_BENCH_GPU_LANGUAGE}_COMPILER_LAUNCHER
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile_launcher.sh;${CMAKE_CURRENT_BINARY_DIR}/compile_times.csv")
target_compile_definitions(tdls_bench_purelupp_${TDLS_BENCH_BACKEND} PRIVATE
    TDLS_BENCH_GIT_COMMIT="${TDLS_BENCH_GIT_COMMIT}")

# When the test tree is enabled too, its buildtests/check targets must
# also build this executable, or `make check` would run the smoke test
# below against a missing binary.
if(TARGET buildtests)
    add_dependencies(buildtests tdls_bench_purelupp_${TDLS_BENCH_BACKEND})
endif()

# Smoke test: a tiny batch through one register and one shared-memory
# variant per built dimension, validating that the harness builds,
# runs, measures, validates, and skips cleanly where the shared
# footprint cannot fit. Skipped (77) when no device is present. Full
# campaigns go through scripts/run_sweep.sh, never through ctest.
add_test(NAME benchmarks.pure_lupp.smoke
         COMMAND tdls_bench_purelupp_${TDLS_BENCH_BACKEND}
                 --filter "static/f64/n[0-9]+/ts3/rl/u1/(reg|shm)$"
                 --batch 512 --runs 2 --distribution default --validate 128 --parity 8)
set_tests_properties(benchmarks.pure_lupp.smoke
                     PROPERTIES LABELS "benchmarks" SKIP_RETURN_CODE 77)
