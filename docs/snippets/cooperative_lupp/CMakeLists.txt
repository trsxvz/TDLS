# Snippets of the CooperativeLUpp pages, one per entry point or knob. The
# groups of threads of the snippets are CPU threads (common/group.hpp).
find_package(Threads REQUIRED)
foreach(snippet
        config_defaults
        config_rows_per_thread
        config_one_thread
        config_phantom_rows
        config_rows_larger
        config_row_interchange
        config_pivot_threshold
        config_singular_floor
        config_unroll
        config_layout
        config_scalar_types
        config_constants
        static_factorize_substitute
        static_substitute_inplace
        static_solve
        static_solve_inplace
        static_canonical
        static_singular
        static_constexpr
        static_newton
        static_deduced_barrier
        dynamic_factorize_substitute
        dynamic_substitute_inplace
        dynamic_solve
        dynamic_solve_inplace
        dynamic_canonical
        dynamic_threads
        dynamic_rows_larger
        dynamic_vs_static
        batch_residencies
        batch_aos
        batch_soa
        batch_aosoa)
    tdls_add_snippet(cooperativelupp ${snippet})
    target_link_libraries(tdls_snippet_cooperativelupp_${snippet} PRIVATE Threads::Threads)
endforeach()

# The OpenMP snippet follows the OpenMP examples: built when a runtime
# is found, skipped with a status message otherwise.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    tdls_add_snippet(cooperativelupp batch_openmp)
    target_link_libraries(tdls_snippet_cooperativelupp_batch_openmp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "tdls: OpenMP not found, OpenMP snippet skipped")
endif()

# The GPU snippet is a single source in the common CUDA/HIP dialect,
# compiled as the language selected by the GPU example options, as the
# GPU snippet of the TiledLUpp pages. The binary returns 77 without a
# device, which ctest reports as a skipped test.
if(TDLS_BUILD_CUDA_EXAMPLES OR TDLS_BUILD_HIP_EXAMPLES)
    set(target tdls_snippet_cooperativelupp_batch_gpu)
    if(TDLS_BUILD_CUDA_EXAMPLES)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
    else()
        enable_language(HIP)
        set_source_files_properties(batch_gpu.cu PROPERTIES LANGUAGE HIP)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL batch_gpu.cu)
    add_dependencies(buildtests ${target})
    target_link_libraries(${target} PRIVATE tdls::tdls)
    target_include_directories(${target} PRIVATE "${TDLS_SNIPPETS_COMMON_DIR}"
                                                 "${CMAKE_SOURCE_DIR}/examples/common")
    if(TDLS_BUILD_CUDA_EXAMPLES AND CMAKE_VERSION VERSION_LESS 3.26)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
    else()
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                                                   HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
    endif()
    add_test(NAME snippets.cooperativelupp.batch_gpu COMMAND ${target})
    set_tests_properties(snippets.cooperativelupp.batch_gpu PROPERTIES LABELS "snippets"
                                                                      SKIP_RETURN_CODE 77)
endif()
