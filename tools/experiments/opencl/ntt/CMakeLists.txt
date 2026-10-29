cmake_minimum_required(VERSION 3.21)
project(ecm_opencl_stage2_probes LANGUAGES CXX)
get_filename_component(ECM_REPO "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
find_package(OpenCL REQUIRED)
set(ECM_GMP_ROOT "" CACHE PATH "Coherent GMP headers/library/runtime prefix")
find_path(ECM_GMP_INCLUDE gmp.h HINTS "${ECM_GMP_ROOT}/include" REQUIRED)
find_library(ECM_GMP_LIBRARY NAMES gmp libgmp HINTS "${ECM_GMP_ROOT}/lib" REQUIRED)
set(ECM_WINDOWS_GMP_ROOT "${ECM_GMP_ROOT}")
set(GMP_INCLUDE_DIR "${ECM_GMP_INCLUDE}")
set(GMP_LIBRARY "${ECM_GMP_LIBRARY}")
# Separate field runtime, Stage2 algorithms and GMP-only probes.
add_library(ecm_opencl_ntt STATIC "${ECM_REPO}/src/ecm/opencl/ntt/runtime.cpp")
add_library(ecm_opencl_stage2_ops STATIC
    "${ECM_REPO}/src/ecm/stage2/opencl/polynomial.cpp"
    "${ECM_REPO}/src/ecm/stage2/opencl/arithmetic.cpp"
    "${ECM_REPO}/src/ecm/stage2/opencl/tree.cpp"
    "${ECM_REPO}/src/ecm/stage2/opencl/points.cpp")
target_link_libraries(ecm_opencl_stage2_ops PUBLIC ecm_opencl_ntt)
foreach(library IN ITEMS ecm_opencl_ntt ecm_opencl_stage2_ops)
    target_compile_features(${library} PUBLIC cxx_std_17)
    target_include_directories(${library} PUBLIC "${ECM_REPO}/src")
    target_link_libraries(${library} PRIVATE OpenCL::OpenCL)
    target_compile_definitions(${library} PRIVATE CL_TARGET_OPENCL_VERSION=120
        "ECM_OPENCL_NTT_RESOURCE_ROOT=\"${CMAKE_CURRENT_BINARY_DIR}/opencl_resources/ntt\"")
    if(MSVC)
        target_compile_options(${library} PRIVATE /W4 /utf-8)
    endif()
endforeach()
include("${ECM_REPO}/tools/build/internal/opencl_stage2.cmake")
add_custom_target(opencl_ntt_resources ALL
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/opencl_resources/ntt"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${ECM_REPO}/src/ecm/opencl/resources/ntt" "${CMAKE_CURRENT_BINARY_DIR}/opencl_resources/ntt"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${ECM_REPO}/src/ecm/opencl/resources/stage2" "${CMAKE_CURRENT_BINARY_DIR}/opencl_resources/stage2")
foreach(probe IN ITEMS opencl_ntt_probe opencl_s4_probe opencl_stage2_probe opencl_points_probe)
    add_executable(${probe} "${ECM_REPO}/tools/test/${probe}.cpp")
    target_include_directories(${probe} PRIVATE "${ECM_REPO}/include")
    target_include_directories(${probe} SYSTEM PRIVATE "${ECM_GMP_INCLUDE}")
    if(probe STREQUAL "opencl_ntt_probe")
        target_link_libraries(${probe} PRIVATE ecm_opencl_ntt)
    else()
        target_link_libraries(${probe} PRIVATE ecm_opencl_stage2_ops)
    endif()
    target_link_libraries(${probe} PRIVATE OpenCL::OpenCL "${ECM_GMP_LIBRARY}")
    target_compile_definitions(${probe} PRIVATE CL_TARGET_OPENCL_VERSION=120 NOMINMAX
        "ECM_OPENCL_NTT_RESOURCE_ROOT=\"${CMAKE_CURRENT_BINARY_DIR}/opencl_resources/ntt\"")
    if(MSVC)
        target_compile_options(${probe} PRIVATE /W4 /utf-8)
    endif()
    add_dependencies(${probe} opencl_ntt_resources)
    if(WIN32 AND EXISTS "${ECM_GMP_ROOT}/bin/gmp-10.dll")
        add_custom_command(TARGET ${probe} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${ECM_GMP_ROOT}/bin/gmp-10.dll" "$<TARGET_FILE_DIR:${probe}>")
    endif()
endforeach()
