# Grid generator of the pure-LUpp benchmark: emits one translation
# unit per variant into OUTPUT_DIR (from variant.cu.in), removes the
# units of variants that left the grid, and writes the source list to
# MANIFEST. This script is the single definition of the sweep grid;
# the translation units it emits are build-tree artifacts, never
# tracked.
#
# Incremental-build correctness: every unit is written through
# configure_file, which only touches the file when its content
# changes, so reconfiguring never causes spurious recompilations; the
# CMakeLists registers this script and the template as configure
# dependencies, so editing either regenerates everything.
#
# Expected -D inputs: TEMPLATE (variant.cu.in), OUTPUT_DIR, MANIFEST,
# DIMS (semicolon-separated dimension list).

foreach(_required TEMPLATE OUTPUT_DIR MANIFEST DIMS)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "generate_variants.cmake: missing -D${_required}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(_sources "")

# Axes. No token may contain an underscore: the underscore-separated
# file stem must map one-to-one onto the slash-separated variant tag
# (the compile-time log relies on that mapping).
set(_scalars f64 f32)
set(_schedules rl ll)
set(_kinds static-u1 static-u0 dynamic)
set(_placements reg dram dram-rreg dram-preg dram-rreg-preg shm shm-rreg shm-preg shm-rreg-preg)

foreach(V_N IN LISTS DIMS)
    foreach(_scalar IN LISTS _scalars)
        if(_scalar STREQUAL "f64")
            set(V_SCALAR double)
        else()
            set(V_SCALAR float)
        endif()
        foreach(V_TS RANGE 1 6)
            foreach(_sched IN LISTS _schedules)
                if(_sched STREQUAL "rl")
                    set(V_SCHED RightLooking)
                else()
                    set(V_SCHED LeftLooking)
                endif()
                foreach(_kind IN LISTS _kinds)
                    if(_kind STREQUAL "static-u1")
                        set(V_KIND static_)
                        set(V_UNROLL true)
                        set(_kind_stem "static_${_scalar}")
                        set(_u_stem "_u1")
                    elseif(_kind STREQUAL "static-u0")
                        set(V_KIND static_)
                        set(V_UNROLL false)
                        set(_kind_stem "static_${_scalar}")
                        set(_u_stem "_u0")
                    else()
                        set(V_KIND dynamic_)
                        set(V_UNROLL true)
                        set(_kind_stem "dynamic_${_scalar}")
                        set(_u_stem "")
                    endif()
                    foreach(_placement IN LISTS _placements)
                        string(REPLACE "-" "_" V_PLACEMENT "${_placement}")
                        set(_stem
                            "purelupp_${_kind_stem}_n${V_N}_ts${V_TS}_${_sched}${_u_stem}_${_placement}")
                        string(REPLACE "-" "_" V_ID "${_stem}")
                        string(REPLACE "_" "/" V_TAG "${_stem}")
                        configure_file("${TEMPLATE}" "${OUTPUT_DIR}/${_stem}.cu" @ONLY)
                        list(APPEND _sources "${OUTPUT_DIR}/${_stem}.cu")
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endforeach()

# Remove the units of variants that left the grid, so a shrunk grid
# never links (or measures) stale variants.
file(GLOB _existing "${OUTPUT_DIR}/*.cu")
foreach(_file IN LISTS _existing)
    list(FIND _sources "${_file}" _index)
    if(_index EQUAL -1)
        file(REMOVE "${_file}")
    endif()
endforeach()

# The manifest is rewritten through the same only-if-changed door.
string(REPLACE ";" "\n" _manifest_content "${_sources}")
file(WRITE "${MANIFEST}.tmp" "${_manifest_content}\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${MANIFEST}.tmp" "${MANIFEST}")
file(REMOVE "${MANIFEST}.tmp")

list(LENGTH _sources _count)
message(STATUS "tdls benchmarks: ${_count} variant translation units generated")
