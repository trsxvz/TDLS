# GPU examples: a large batch of independent systems, one thread per
# system. Single sources in the common CUDA/HIP dialect (see
# examples/common/gpu_runtime.hpp), compiled as the language selected
# by the opt-in options TDLS_BUILD_CUDA_EXAMPLES /
# TDLS_BUILD_HIP_EXAMPLES: nothing is probed while both are OFF, and a
# requested toolchain that cannot be found is a hard configure error.
# Each binary returns 77 when no device is present, which ctest
# reports as a skipped test.
if(TDLS_BUILD_CUDA_EXAMPLES AND TDLS_BUILD_HIP_EXAMPLES)
    message(FATAL_ERROR "tdls: TDLS_BUILD_CUDA_EXAMPLES and TDLS_BUILD_HIP_EXAMPLES are "
                        "mutually exclusive: the example sources are compiled as a single "
                        "language")
endif()
if(TDLS_BUILD_CUDA_EXAMPLES)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "tdls: TDLS_BUILD_CUDA_EXAMPLES is ON but no CUDA compiler was "
                            "found; set CMAKE_CUDA_COMPILER or fix the PATH")
    endif()
    enable_language(CUDA)
    set(TDLS_GPU_LANGUAGE CUDA)
elseif(TDLS_BUILD_HIP_EXAMPLES)
    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR "tdls: TDLS_BUILD_HIP_EXAMPLES is ON but no HIP compiler was "
                            "found; set CMAKE_HIP_COMPILER or fix the PATH")
    endif()
    enable_language(HIP)
    set(TDLS_GPU_LANGUAGE HIP)
endif()
if(TDLS_GPU_LANGUAGE)
    foreach(example implicit_ode_batch_gpu implicit_ode_batch_gpu_soa
                    integral_equation_batch_gpu integral_equation_batch_gpu_soa)
        if(TDLS_GPU_LANGUAGE STREQUAL "HIP")
            set_source_files_properties(${example}.cu PROPERTIES LANGUAGE HIP)
        endif()
        add_executable(tdls_example_tiledlupp_${example} ${example}.cu)
        target_link_libraries(tdls_example_tiledlupp_${example} PRIVATE tdls::tdls)
        target_include_directories(tdls_example_tiledlupp_${example}
                                   PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../common")
        set_target_properties(tdls_example_tiledlupp_${example} PROPERTIES CUDA_STANDARD 20
                                                                           HIP_STANDARD 20)
        add_test(NAME examples.tiledlupp.${example} COMMAND tdls_example_tiledlupp_${example})
        set_tests_properties(examples.tiledlupp.${example} PROPERTIES LABELS "examples"
                                                                      SKIP_RETURN_CODE 77)
    endforeach()
else()
    message(STATUS "tdls: GPU examples not built (opt in with TDLS_BUILD_CUDA_EXAMPLES "
                   "or TDLS_BUILD_HIP_EXAMPLES)")
endif()
