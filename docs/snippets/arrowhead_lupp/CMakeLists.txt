# Snippets of the ArrowheadLUpp pages, one per entry point, storage or knob.
foreach(snippet
        start_cpu
        start_dynamic
        static_solve_inplace
        static_solve
        static_refused
        static_constexpr
        dynamic_solve_inplace
        dynamic_solve
        dynamic_vs_static
        storage_full
        storage_packed
        storage_csr
        storage_bsr
        coupling_dense
        config_defaults
        config_layout
        config_condensation_threshold
        config_blocks
        config_pass_width
        config_gather
        config_schur_in_place
        config_unroll
        config_scalar_types
        batch_aos
        batch_soa
        batch_aosoa
        batch_residencies)
    tdls_add_snippet(arrowheadlupp ${snippet})
endforeach()

# The OpenMP snippet follows the OpenMP examples: built when a runtime
# is found, skipped with a status message otherwise.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    tdls_add_snippet(arrowheadlupp batch_openmp)
    target_link_libraries(tdls_snippet_arrowheadlupp_batch_openmp PRIVATE OpenMP::OpenMP_CXX)
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
        tdls_add_gpu_snippet(arrowheadlupp ${snippet})
    endforeach()
endif()
