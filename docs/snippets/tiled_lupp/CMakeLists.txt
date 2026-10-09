# Snippets of the TiledLUpp pages, one per entry point or knob.
foreach(snippet
        start_cpu
        start_dynamic
        config_defaults
        config_tile_size
        config_single_tile
        config_tile_larger
        config_scalar_tiles
        config_schedules
        config_thresholds
        config_first_acceptable
        config_unroll
        config_layout
        config_scalar_types
        config_constants
        static_factorize_substitute
        static_substitute_inplace
        static_solve
        static_solve_inplace
        static_canonical
        static_canonical_block
        static_multirhs
        static_multirhs_inplace
        static_solve_multirhs
        static_solve_inplace_multirhs
        static_pass_cutting
        static_oot_counter
        static_singular
        static_constexpr
        static_newton
        dynamic_factorize_substitute
        dynamic_substitute_inplace
        dynamic_solve
        dynamic_solve_inplace
        dynamic_canonical
        dynamic_canonical_block
        dynamic_multirhs
        dynamic_multirhs_inplace
        dynamic_solve_multirhs
        dynamic_solve_inplace_multirhs
        dynamic_pass_cutting
        dynamic_oot_counter
        dynamic_tile_helpers
        dynamic_tile_larger
        dynamic_vs_static
        batch_residencies
        batch_aos
        batch_soa
        batch_aosoa)
    tdls_add_snippet(tiledlupp ${snippet})
endforeach()

# The OpenMP snippet follows the OpenMP examples: built when a runtime
# is found, skipped with a status message otherwise.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    tdls_add_snippet(tiledlupp batch_openmp)
    target_link_libraries(tdls_snippet_tiledlupp_batch_openmp PRIVATE OpenMP::OpenMP_CXX)
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
        tdls_add_gpu_snippet(tiledlupp ${snippet})
    endforeach()
endif()
