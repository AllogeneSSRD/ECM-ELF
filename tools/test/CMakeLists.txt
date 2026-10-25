# tools/test/CMakeLists.txt -- test / verification executables.
#
# Moved out of the root CMakeLists.txt (see tools/bench/CMakeLists.txt for the rationale
# and for ecm_tool_output_root(), which keeps the executables in the build root so the
# documented paths and the tools/test/*.ps1 scripts keep working).

# ---------------------------------------------------------------------------
# main: the older OpenCL low-risk test harness (tools/test/main_test.cpp +
# tools/test/opencl_low_risk_tests.cpp, sharing src/ecm/backend/opencl/runtime/device.cpp with the other OpenCL tools).
# ---------------------------------------------------------------------------
add_executable(stage1_selection_test stage1_selection_test.cpp
    ${CMAKE_SOURCE_DIR}/src/ecm/core/stage1/selection.cpp)
target_include_directories(stage1_selection_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
ecm_tool_output_root(stage1_selection_test)

add_executable(stage1_checkpoint_rotation_test
    stage1_checkpoint_rotation_test.cpp
    ${CMAKE_SOURCE_DIR}/src/ecm/backend/opencl/stage1/montgomery/checkpoint.cpp
    ${CMAKE_SOURCE_DIR}/src/ecm/backend/cpu/stage1/montgomery/checkpoint.cpp
    ${CMAKE_SOURCE_DIR}/src/io/stage1_checkpoint.cpp
    ${CMAKE_SOURCE_DIR}/src/ecm/platform/stage1_stop.cpp
    ${CMAKE_SOURCE_DIR}/src/io/files.cpp
    ${CMAKE_SOURCE_DIR}/src/ecm/platform/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/ecm/application/configuration/runtime.cpp)
target_include_directories(stage1_checkpoint_rotation_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
ecm_target_link_gmp(stage1_checkpoint_rotation_test)
ecm_tool_output_root(stage1_checkpoint_rotation_test)

add_executable(worktodo_sections_test
    worktodo_sections_test.cpp
    ${CMAKE_SOURCE_DIR}/src/io/files.cpp
    ${CMAKE_SOURCE_DIR}/src/io/worktodo.cpp)
target_include_directories(worktodo_sections_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
ecm_target_link_gmp(worktodo_sections_test)
if(MSVC)
    target_compile_options(worktodo_sections_test PRIVATE /utf-8)
endif()
ecm_tool_output_root(worktodo_sections_test)

add_executable(stage2_transfer_test
    stage2_transfer_test.cpp
    ${CMAKE_SOURCE_DIR}/src/sync/stage2.cpp
    ${CMAKE_SOURCE_DIR}/src/io/files.cpp
    ${CMAKE_SOURCE_DIR}/src/io/worktodo.cpp)
target_include_directories(stage2_transfer_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
ecm_target_link_gmp(stage2_transfer_test)
if(MSVC)
    target_compile_options(stage2_transfer_test PRIVATE /utf-8)
endif()
ecm_tool_output_root(stage2_transfer_test)

add_executable(main
    ${CMAKE_SOURCE_DIR}/src/ecm/backend/opencl/runtime/device.cpp
    ${CMAKE_SOURCE_DIR}/tools/test/main_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/test/opencl_low_risk_tests.cpp
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
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
    ${CMAKE_SOURCE_DIR}/tools/test/sliced_cios_test.cpp
)

target_include_directories(sliced_cios_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(sliced_cios_test
    OpenCL::OpenCL
)

ecm_target_link_gmp(sliced_cios_test)
ecm_tool_output_root(sliced_cios_test)

add_executable(sliced_cios_8192_test
    ${CMAKE_SOURCE_DIR}/tools/test/sliced_cios_8192_test.cpp
)

target_include_directories(sliced_cios_8192_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(sliced_cios_8192_test
    OpenCL::OpenCL
)

ecm_target_link_gmp(sliced_cios_8192_test)
ecm_tool_output_root(sliced_cios_8192_test)
