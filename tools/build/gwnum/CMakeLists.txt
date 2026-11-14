cmake_minimum_required(VERSION 3.20)
project(ecm_gwnum LANGUAGES CXX)

if(NOT (WIN32 AND MSVC AND MSVC_CXX_ARCHITECTURE_ID STREQUAL "x64"))
    message(FATAL_ERROR "The GWNUM Stage1 DLL requires Windows x64 MSVC")
endif()

get_filename_component(ECM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(ECM_GWNUM_ROOT "${ECM_ROOT}/.refactor/p95v3106b01.source/gwnum"
    CACHE PATH "GWNUM 31.6 headers and Windows x64 library")
foreach(_file gwnum.h gwnum64.lib)
    if(NOT EXISTS "${ECM_GWNUM_ROOT}/${_file}")
        message(FATAL_ERROR "Missing GWNUM 31.6 dependency: ${ECM_GWNUM_ROOT}/${_file}")
    endif()
endforeach()

add_library(ecm_gwnum MODULE "${ECM_ROOT}/src/ecm/stage1/montgomery/cpu/gwnum_plugin.cpp")
target_compile_features(ecm_gwnum PRIVATE cxx_std_17)
target_include_directories(ecm_gwnum PRIVATE "${ECM_ROOT}/include" "${ECM_GWNUM_ROOT}")
target_link_libraries(ecm_gwnum PRIVATE "${ECM_GWNUM_ROOT}/gwnum64.lib" advapi32)
# Match the CRT used by the upstream Windows static library.
set_target_properties(ecm_gwnum PROPERTIES
    OUTPUT_NAME gwnum
    MSVC_RUNTIME_LIBRARY MultiThreaded)
