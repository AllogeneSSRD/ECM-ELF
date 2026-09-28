# tools/disasm/CMakeLists.txt -- OpenCL assembly self-test and ISA-export tools.
#
# Moved out of the root CMakeLists.txt (see tools/bench/CMakeLists.txt for the rationale
# and for ecm_tool_output_root(), which keeps the executables in the build root).
# All three share src/cl_probe.cpp and the cgbn_opencl library.

add_executable(opencl_asm_selftest
    ${CMAKE_SOURCE_DIR}/src/cl_probe.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_asm_selftest.cpp
)

target_include_directories(opencl_asm_selftest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_asm_selftest
    OpenCL::OpenCL
    cgbn_opencl
)

ecm_tool_output_root(opencl_asm_selftest)

add_executable(opencl_mont_isa_export
    ${CMAKE_SOURCE_DIR}/src/cl_probe.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_mont_isa_export.cpp
)

target_include_directories(opencl_mont_isa_export PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_mont_isa_export
    OpenCL::OpenCL
    cgbn_opencl
)

ecm_tool_output_root(opencl_mont_isa_export)

add_executable(opencl_addsub_isa_export
    ${CMAKE_SOURCE_DIR}/src/cl_probe.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_addsub_isa_export.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_addsub_manifest.cpp
)

target_include_directories(opencl_addsub_isa_export PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_addsub_isa_export
    OpenCL::OpenCL
    cgbn_opencl
)

ecm_tool_output_root(opencl_addsub_isa_export)
