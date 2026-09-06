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

add_subdirectory(tiled_lupp)
if(TDLS_BUILD_TFEL_SNIPPETS)
    add_subdirectory(tfel)
endif()
