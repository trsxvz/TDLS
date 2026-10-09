# Snippets of the CooperativeLUpp pages, one per entry point or knob, in
# the order of the pages. The groups of threads of the CPU snippets are
# CPU threads (common/group.hpp).
find_package(Threads REQUIRED)
foreach(snippet
        start_cpu
        start_dynamic
        static_factorize_substitute
        static_substitute_inplace
        static_solve
        static_solve_inplace
        static_canonical
        static_singular
        static_constexpr
        static_newton
        dynamic_factorize_substitute
        dynamic_substitute_inplace
        dynamic_solve
        dynamic_solve_inplace
        dynamic_canonical
        dynamic_rows_larger
        dynamic_vs_static
        group_cpu_threads
        group_runtime_dimension
        group_deduced_barrier
        group_explicit_barrier
        group_newton
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

# The GPU snippets, in the common CUDA/HIP dialect (tdls_add_gpu_snippet).
if(TDLS_BUILD_CUDA_EXAMPLES OR TDLS_BUILD_HIP_EXAMPLES)
    if(TDLS_BUILD_CUDA_EXAMPLES)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
    else()
        enable_language(HIP)
    endif()
    foreach(snippet start_gpu batch_gpu)
        tdls_add_gpu_snippet(cooperativelupp ${snippet})
    endforeach()
endif()
