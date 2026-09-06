# TFEL snippets: the adaptors on real tfel::math objects. Opt-in
# through TDLS_BUILD_TFEL_SNIPPETS, TFEL found by find_package(TFELMath);
# the library itself stays TFEL-free.
find_package(TFELMath REQUIRED)

foreach(snippet
        fixed_factorize_substitute
        fixed_substitute_inplace
        fixed_solve
        fixed_solve_inplace
        fixed_canonical
        fixed_config
        fixed_counting
        fixed_pivot_forms
        runtime_factorize_substitute
        runtime_solve_inplace
        runtime_canonical
        view_pointer
        view_strided
        view_array
        view_mixed
        block_factorize_substitute
        block_substitute_inplace
        block_solve
        block_solve_inplace
        block_canonical
        block_pass_cutting
        block_runtime)
    tdls_add_snippet(tfel ${snippet})
    target_link_libraries(tdls_snippet_tfel_${snippet} PRIVATE tfel::TFELMath)
endforeach()

# Rejected shapes: translation units that must not compile, built on
# demand by ctest and passing only when the diagnostic of the contract
# appears, as the reject tests of tests/solvers.
set(TDLS_TFEL_REJECT_CASES
    reject_submatrix
    reject_derivative
    reject_gather
    reject_colmajor
    reject_const
    reject_extent)
set(TDLS_TFEL_REJECT_DIAGNOSTICS
    "row-strided matrix views"
    "row-strided matrix views"
    "A must be a dense object"
    "dense objects are addressed row-major"
    "A must not be const here"
    "vector extent does not match the system dimension")

foreach(case diagnostic IN ZIP_LISTS TDLS_TFEL_REJECT_CASES TDLS_TFEL_REJECT_DIAGNOSTICS)
    set(target tdls_snippet_tfel_${case})
    add_executable(${target} EXCLUDE_FROM_ALL ${case}.cpp)
    target_link_libraries(${target} PRIVATE tdls::tdls tfel::TFELMath)
    target_compile_features(${target} PRIVATE cxx_std_20)
    add_test(NAME snippets.tfel.${case}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${target}
                     --config $<CONFIG>)
    set_tests_properties(snippets.tfel.${case} PROPERTIES LABELS "snippets"
                         PASS_REGULAR_EXPRESSION "${diagnostic}" RESOURCE_LOCK build_tree)
endforeach()
