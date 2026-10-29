# Native Stage2 libraries and the existing production queue/process driver.
# This file can be included by the root project or the lightweight test build.
get_filename_component(_ocl_repo "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT TARGET ecm_opencl_ntt)
    add_library(ecm_opencl_ntt STATIC "${_ocl_repo}/src/ecm/opencl/ntt/runtime.cpp")
    add_library(ecm_opencl_stage2_ops STATIC
        "${_ocl_repo}/src/ecm/stage2/opencl/polynomial.cpp"
        "${_ocl_repo}/src/ecm/stage2/opencl/arithmetic.cpp"
        "${_ocl_repo}/src/ecm/stage2/opencl/tree.cpp"
        "${_ocl_repo}/src/ecm/stage2/opencl/points.cpp")
    target_link_libraries(ecm_opencl_stage2_ops PUBLIC ecm_opencl_ntt)
    foreach(_target ecm_opencl_ntt ecm_opencl_stage2_ops)
        target_compile_features(${_target} PUBLIC cxx_std_17)
        target_include_directories(${_target} PUBLIC "${_ocl_repo}/src")
        target_link_libraries(${_target} PRIVATE OpenCL::OpenCL)
        target_compile_definitions(${_target} PRIVATE CL_TARGET_OPENCL_VERSION=120)
        if(MSVC)
            target_compile_options(${_target} PRIVATE /O2 /utf-8)
        endif()
    endforeach()
endif()
include("${_ocl_repo}/tools/build/internal/cuda_sources.cmake")
set(_ocl_host ${ECM_STAGE2_HOST_SOURCES})
list(TRANSFORM _ocl_host PREPEND "${_ocl_repo}/")
add_executable(ecm_opencl_stage2 ${_ocl_host} "${_ocl_repo}/src/ecm/stage2/opencl/engine.cpp")
target_compile_features(ecm_opencl_stage2 PRIVATE cxx_std_17)
target_compile_definitions(ecm_opencl_stage2 PRIVATE ECM_STAGE2_OPENCL=1 NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_include_directories(ecm_opencl_stage2 PRIVATE "${_ocl_repo}/include" "${_ocl_repo}/src" "${GMP_INCLUDE_DIR}")
target_link_libraries(ecm_opencl_stage2 PRIVATE ecm_opencl_stage2_ops "${GMP_LIBRARY}")
if(MSVC)
    target_compile_options(ecm_opencl_stage2 PRIVATE /O2 /utf-8 /bigobj)
endif()
add_custom_target(ecm_opencl_stage2_resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${_ocl_repo}/src/ecm/opencl/resources/ntt" "$<TARGET_FILE_DIR:ecm_opencl_stage2>/opencl_resources/ntt"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${_ocl_repo}/src/ecm/opencl/resources/stage2" "$<TARGET_FILE_DIR:ecm_opencl_stage2>/opencl_resources/stage2")
add_dependencies(ecm_opencl_stage2 ecm_opencl_stage2_resources)
if(EXISTS "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll")
    add_custom_command(TARGET ecm_opencl_stage2 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll" "$<TARGET_FILE_DIR:ecm_opencl_stage2>")
endif()
