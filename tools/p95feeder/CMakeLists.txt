# tools/p95feeder/CMakeLists.txt -- ecm_p95feeder, the standalone Prime95 handoff feeder.
#
# ecm.exe writes its stage-1 results locally only (<tmp_dir>/e{n:07d}_c{k}.tmp); the feeder
# watches that directory and delivers each finished curve into a running Prime95 instance
# (a save file e{n:07d} plus an ECM= line under a free [Worker #N] section of
# worktodo.add).  See docs/architecture/STAGE1.md 11 and tools/test/test_feeder.ps1.
#
# GMP-only: no OpenCL/OpenSSL dependency.  Moved out of the root CMakeLists.txt; the
# executable stays in the build root (build_vs18\Release\ecm_p95feeder.exe) because
# tools/test/test_feeder.ps1 references that path.

add_executable(ecm_p95feeder
    ${CMAKE_SOURCE_DIR}/src/core/ecm_p95feeder.cpp
    ${CMAKE_SOURCE_DIR}/src/core/p95_worktodo.cpp
    ${CMAKE_SOURCE_DIR}/src/core/ecm_worktodo.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/ecm_edwards_save.cpp
)

target_include_directories(ecm_p95feeder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(ecm_p95feeder)
ecm_tool_output_root(ecm_p95feeder)

if(MSVC)
    target_compile_options(ecm_p95feeder PRIVATE /utf-8)
endif()

# Copy the GMP runtime DLL next to the executable (same treatment as `ecm`/`ecm_cuda`).
foreach(_gmp_dll_cand "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll"
                      "${OPENSSL_ROOT_DIR}/bin/gmp-10.dll")
    if(MSVC AND EXISTS "${_gmp_dll_cand}")
        add_custom_command(TARGET ecm_p95feeder POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_gmp_dll_cand}"
                "$<TARGET_FILE_DIR:ecm_p95feeder>"
            COMMENT "Copying GMP DLL next to ecm_p95feeder")
        break()
    endif()
endforeach()
