# Parallel STL examples: the three problems on std::for_each with the
# par_unseq policy, one index per system. A single source serves both
# scales. The CPU targets build whenever the host compiler offers a
# parallel STL backend (nvc++ with -stdpar=multicore, or libstdc++ on
# oneTBB) and are skipped with the status message below otherwise: as
# for the OpenMP examples, the detection is implicit and cannot break
# the configure. The device targets are opt-in through
# TDLS_BUILD_STDPAR_DEVICE_EXAMPLES and need a device at run time. The
# offload is a matter of compiler flags, which the user states in
# TDLS_STDPAR_DEVICE_FLAGS: --acpp-stdpar with AdaptiveCpp, -stdpar=gpu
# with nvc++, --hipstdpar with an --offload-arch with the ROCm clang.
set(TDLS_STDPAR_EXAMPLES implicit_ode_batch_stdpar integral_equation_batch_stdpar
                         norton_law_batch_stdpar)

set(TDLS_STDPAR_CPU_FOUND FALSE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(TDLS_STDPAR_CPU_FOUND TRUE)
    set(TDLS_STDPAR_CPU_OPTIONS -stdpar=multicore)
    set(TDLS_STDPAR_CPU_LIBRARIES)
else()
    find_package(TBB QUIET)
    if(TBB_FOUND)
        include(CheckCXXSourceCompiles)
        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        set(CMAKE_REQUIRED_LIBRARIES TBB::tbb)
        check_cxx_source_compiles("
            #include <algorithm>
            #include <execution>
            #include <vector>
            int main() {
                std::vector<int> v(8, 1);
                std::for_each(std::execution::par_unseq, v.begin(), v.end(),
                              [](int& x) { x += 1; });
                return v[7] == 2 ? 0 : 1;
            }" TDLS_HAVE_PARALLEL_STL)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(TDLS_HAVE_PARALLEL_STL)
            set(TDLS_STDPAR_CPU_FOUND TRUE)
            set(TDLS_STDPAR_CPU_OPTIONS)
            set(TDLS_STDPAR_CPU_LIBRARIES TBB::tbb)
        endif()
    endif()
endif()

if(TDLS_STDPAR_CPU_FOUND)
    foreach(example IN LISTS TDLS_STDPAR_EXAMPLES)
        tdls_add_example(tiledlupp ${example})
        target_compile_options(tdls_example_tiledlupp_${example} PRIVATE ${TDLS_STDPAR_CPU_OPTIONS})
        target_link_options(tdls_example_tiledlupp_${example} PRIVATE ${TDLS_STDPAR_CPU_OPTIONS})
        target_link_libraries(tdls_example_tiledlupp_${example} PRIVATE ${TDLS_STDPAR_CPU_LIBRARIES})
    endforeach()
else()
    message(STATUS "tdls: no parallel STL backend found, parallel STL examples skipped")
endif()

if(TDLS_BUILD_STDPAR_DEVICE_EXAMPLES)
    set(TDLS_STDPAR_DEVICE_FLAGS "" CACHE STRING
        "Compile and link flags offloading the parallel STL to a device")
    if(NOT TDLS_STDPAR_DEVICE_FLAGS)
        message(FATAL_ERROR "tdls: TDLS_BUILD_STDPAR_DEVICE_EXAMPLES is ON but "
                            "TDLS_STDPAR_DEVICE_FLAGS is empty; set it to the offload flags of "
                            "the C++ compiler (--acpp-stdpar, -stdpar=gpu, --hipstdpar ...)")
    endif()
    separate_arguments(TDLS_STDPAR_DEVICE_FLAGS_LIST NATIVE_COMMAND "${TDLS_STDPAR_DEVICE_FLAGS}")
    foreach(example IN LISTS TDLS_STDPAR_EXAMPLES)
        set(target tdls_example_tiledlupp_${example}_gpu)
        add_executable(${target} EXCLUDE_FROM_ALL ${example}.cpp)
        add_dependencies(buildtests ${target})
        target_link_libraries(${target} PRIVATE tdls::tdls)
        target_include_directories(${target} PRIVATE "${TDLS_EXAMPLES_COMMON_DIR}")
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_compile_options(${target} PRIVATE ${TDLS_STDPAR_DEVICE_FLAGS_LIST}
                                                 -Wall -Wextra -Wpedantic -Werror)
        target_link_options(${target} PRIVATE ${TDLS_STDPAR_DEVICE_FLAGS_LIST})
        add_test(NAME examples.tiledlupp.${example}_gpu COMMAND ${target})
        set_tests_properties(examples.tiledlupp.${example}_gpu PROPERTIES LABELS "examples")
    endforeach()
endif()
