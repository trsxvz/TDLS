# OpenMP examples: a moderate batch of integration points, one thread
# iteration per point. Built only when an OpenMP runtime is available.
# The detection is deliberately implicit, unlike the GPU examples'
# opt-in toolchains: OpenMP is a flag of the already-chosen host
# compiler, and a failed detection cannot break the configure: it
# only skips these examples, with the status message below.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    foreach(example aggregate_batch_omp iwan_batch_omp polycrystal_batch_omp)
        tdls_add_example(arrowheadlupp ${example})
        target_link_libraries(tdls_example_arrowheadlupp_${example} PRIVATE OpenMP::OpenMP_CXX)
    endforeach()
else()
    message(STATUS "tdls: OpenMP not found, OpenMP examples skipped")
endif()
