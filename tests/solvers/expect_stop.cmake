# Runs a program that must stop with a message: the test passes when the
# program fails and its output holds the message. A program stopped by a
# signal fails a plain ctest entry whatever its output, hence this script.
#   cmake -DPROGRAM=<executable> -DMESSAGE=<text> -P expect_stop.cmake
execute_process(COMMAND "${PROGRAM}" RESULT_VARIABLE result OUTPUT_VARIABLE output
                ERROR_VARIABLE output)
if("${result}" STREQUAL "0")
    message(FATAL_ERROR "The program finished instead of stopping.")
endif()
string(FIND "${output}" "${MESSAGE}" position)
if(position EQUAL -1)
    message(FATAL_ERROR "The program stopped (${result}) without the message \"${MESSAGE}\":\n"
                        "${output}")
endif()
message(STATUS "The program stopped (${result}) with the expected message.")
