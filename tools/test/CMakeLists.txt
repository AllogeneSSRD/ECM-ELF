# tools/test/CMakeLists.txt -- test / verification executables.
#
# Moved out of the root CMakeLists.txt (see tools/bench/CMakeLists.txt for the rationale
# and for ecm_tool_output_root(), which keeps the executables in the build root so the
# documented paths and the tools/test/*.ps1 scripts keep working).

# ---------------------------------------------------------------------------
# main: the older OpenCL low-risk test harness (src/main_test.cpp +
# src/opencl_low_risk_tests.cpp, sharing src/cl_probe.cpp with the other OpenCL tools).
# ---------------------------------------------------------------------------
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
