# Documentation snippets: one self-checking program per snippet, shown
# in the Sphinx pages by literalinclude between its two markers. Each
# is compiled at the C++20 floor of the library and executed by ctest
# under the "snippets" label, so the documentation cannot drift from
# the code. The tree is split by solver family; docs/snippets/common
# holds the result check shared by every snippet.
set(TDLS_SNIPPETS_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")
function(tdls_add_snippet family snippet)
    set(target tdls_snippet_${family}_${snippet})
    add_executable(${target} EXCLUDE_FROM_ALL ${snippet}.cpp)
    add_dependencies(buildtests ${target})
    target_link_libraries(${target} PRIVATE tdls::tdls)
    target_include_directories(${target} PRIVATE "${TDLS_SNIPPETS_COMMON_DIR}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME snippets.${family}.${snippet} COMMAND ${target})
    set_tests_properties(snippets.${family}.${snippet} PROPERTIES LABELS "snippets")
endfunction()

# tdls_add_gpu_snippet(<family> <snippet>)
# Builds the GPU snippet <snippet>.cu, a single source in the common
# CUDA/HIP dialect, as the language the GPU example options select: the
# calling directory enables it, since enable_language needs a directory
# scope. The binary returns 77 without a device, which ctest reports as a
# skipped test.
function(tdls_add_gpu_snippet family snippet)
    set(target tdls_snippet_${family}_${snippet})
    if(NOT TDLS_BUILD_CUDA_EXAMPLES)
        set_source_files_properties(${snippet}.cu PROPERTIES LANGUAGE HIP)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL ${snippet}.cu)
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
    add_test(NAME snippets.${family}.${snippet} COMMAND ${target})
    set_tests_properties(snippets.${family}.${snippet} PROPERTIES LABELS "snippets"
                                                                  SKIP_RETURN_CODE 77)
endfunction()

add_subdirectory(cooperative_lupp)
add_subdirectory(tiled_lupp)
if(TDLS_BUILD_TFEL_SNIPPETS)
    add_subdirectory(tfel)
endif()
