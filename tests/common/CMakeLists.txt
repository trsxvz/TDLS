# Tests of the shared test infrastructure itself (this directory): the
# reference solver and the backward-error metric that every solver suite
# anchors its accuracy claim on. Tagged "common". These translation
# units use only the homemade harness - they do not include the library
# - so they stay cheap to compile.
add_executable(tdls_test_common_backward_error backward_error.cpp)
target_include_directories(tdls_test_common_backward_error PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(tdls_test_common_backward_error PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(tdls_test_common_backward_error PRIVATE
    $<$<CONFIG:Release>:-O1> -Wall -Wextra -Wpedantic -Werror
    # Clang >=22 flags __COUNTER__ (harness ids) under -Wc2y-extensions;
    # -Werror makes it fatal. Exempt it on Clang (no-op elsewhere).
    $<$<CXX_COMPILER_ID:Clang>:-Wno-c2y-extensions>)
endif()
add_test(NAME common.backward_error COMMAND tdls_test_common_backward_error)
set_tests_properties(common.backward_error PROPERTIES LABELS "common")
