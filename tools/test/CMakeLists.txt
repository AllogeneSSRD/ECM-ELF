# tools/test/CMakeLists.txt -- test / verification executables.
#
# Moved out of the root CMakeLists.txt (see tools/bench/CMakeLists.txt for the rationale
# and for ecm_tool_output_root(), which keeps the executables in the build root so the
# documented paths and the tools/test/*.ps1 scripts keep working).

# ---------------------------------------------------------------------------
# main: the older OpenCL low-risk test harness (src/main_test.cpp +
# src/opencl_low_risk_tests.cpp, sharing src/cl_probe.cpp with the other OpenCL tools).
# ---------------------------------------------------------------------------
add_executable(stage1_checkpoint_rotation_test
    stage1_checkpoint_rotation_test.cpp
    ${CMAKE_SOURCE_DIR}/src/core/ecm_checkpoint.cpp
    ${CMAKE_SOURCE_DIR}/src/core/ecm_mont_ckpt.cpp
    ${CMAKE_SOURCE_DIR}/src/core/ecm_stage1_checkpoint_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_log.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_runtime_config.cpp)
target_include_directories(stage1_checkpoint_rotation_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
ecm_target_link_gmp(stage1_checkpoint_rotation_test)
ecm_tool_output_root(stage1_checkpoint_rotation_test)

add_executable(worktodo_sections_test
    worktodo_sections_test.cpp
    ${CMAKE_SOURCE_DIR}/src/core/ecm_worktodo.cpp)
target_include_directories(worktodo_sections_test PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
ecm_target_link_gmp(worktodo_sections_test)
if(MSVC)
    target_compile_options(worktodo_sections_test PRIVATE /utf-8)
endif()
ecm_tool_output_root(worktodo_sections_test)

add_executable(main
    ${CMAKE_SOURCE_DIR}/src/cl_probe.cpp
    ${CMAKE_SOURCE_DIR}/src/main_test.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_low_risk_tests.cpp
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(main
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
)

ecm_target_link_gmp(main)
ecm_tool_output_root(main)

# ---------------------------------------------------------------------------
# Sliced CIOS standalone verification targets (OpenCL kernels; GMP reference values).
# ---------------------------------------------------------------------------
add_executable(sliced_cios_test
    ${CMAKE_SOURCE_DIR}/src/sliced_cios_test.cpp
)

target_include_directories(sliced_cios_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sliced_cios_test
    OpenCL::OpenCL
)

ecm_target_link_gmp(sliced_cios_test)
ecm_tool_output_root(sliced_cios_test)

add_executable(sliced_cios_8192_test
    ${CMAKE_SOURCE_DIR}/src/sliced_cios_8192_test.cpp
)

target_include_directories(sliced_cios_8192_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sliced_cios_8192_test
    OpenCL::OpenCL
)

ecm_target_link_gmp(sliced_cios_8192_test)
ecm_tool_output_root(sliced_cios_8192_test)
