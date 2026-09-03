# SYCL examples: the three problems as single-source parallel_for
# kernels, one work-item per system, the solver operands in the private
# memory of each work-item. Opt-in through TDLS_BUILD_SYCL_EXAMPLES:
# the C++ compiler must then accept the SYCL flags (icpx with the
# default -fsycl), and a compiler that does not fails the configure
# loudly, as for the CUDA and HIP toolchains. The default spir64 target
# is JIT-compiled SPIR-V, so no device is needed at build time; at run
# time the binaries report themselves skipped when no SYCL device is
# present (exit code 77).
if(TDLS_BUILD_SYCL_EXAMPLES)
    set(TDLS_SYCL_FLAGS "-fsycl" CACHE STRING
        "Compile and link flags enabling SYCL (add -fsycl-targets=... for AOT)")
    separate_arguments(TDLS_SYCL_FLAGS_LIST NATIVE_COMMAND "${TDLS_SYCL_FLAGS}")

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "${TDLS_SYCL_FLAGS}")
    set(CMAKE_REQUIRED_LINK_OPTIONS ${TDLS_SYCL_FLAGS_LIST})
    check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        int main() {
            return sycl::range<1>(1).size() == 1 ? 0 : 1;
        }" TDLS_HAVE_SYCL)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    if(NOT TDLS_HAVE_SYCL)
        message(FATAL_ERROR "tdls: TDLS_BUILD_SYCL_EXAMPLES is ON but the C++ compiler does "
                            "not accept the SYCL flags '${TDLS_SYCL_FLAGS}'; use a SYCL "
                            "compiler such as icpx, or fix TDLS_SYCL_FLAGS")
    endif()

    foreach(example implicit_ode_batch_sycl integral_equation_batch_sycl norton_law_batch_sycl)
        add_executable(tdls_example_tiledlupp_${example} EXCLUDE_FROM_ALL ${example}.cpp)
        add_dependencies(buildtests tdls_example_tiledlupp_${example})
        target_link_libraries(tdls_example_tiledlupp_${example} PRIVATE tdls::tdls)
        target_include_directories(tdls_example_tiledlupp_${example}
                                   PRIVATE "${TDLS_EXAMPLES_COMMON_DIR}")
        target_compile_features(tdls_example_tiledlupp_${example} PRIVATE cxx_std_20)
        target_compile_options(tdls_example_tiledlupp_${example}
                               PRIVATE ${TDLS_SYCL_FLAGS_LIST} -Wall -Wextra -Wpedantic -Werror)
        target_link_options(tdls_example_tiledlupp_${example} PRIVATE ${TDLS_SYCL_FLAGS_LIST})
        add_test(NAME examples.tiledlupp.${example} COMMAND tdls_example_tiledlupp_${example})
        set_tests_properties(examples.tiledlupp.${example} PROPERTIES LABELS "examples"
                                                                      SKIP_RETURN_CODE 77)
    endforeach()
else()
    message(STATUS "tdls: SYCL examples not built (opt in with TDLS_BUILD_SYCL_EXAMPLES)")
endif()
