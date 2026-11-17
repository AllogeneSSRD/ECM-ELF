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
option(ECM_GW_STAGE2 "Include the Edwards Stage2 polynomial engine" OFF)
if(ECM_GW_STAGE2)
    set(_cpu "${ECM_ROOT}/src/ecm/stage2/cpu")
    set(_gmp "${ECM_ROOT}/third_party/gmp-zen3/dist")
    target_sources(ecm_gwnum PRIVATE "${_cpu}/gwpoly.cpp" "${_cpu}/gwpoly_plugin.cpp")
    target_include_directories(ecm_gwnum PRIVATE "${_gmp}/include")
    target_link_libraries(ecm_gwnum PRIVATE "${_gmp}/lib/gmp.lib" "${ECM_GWNUM_ROOT}/polymult64.lib")
    set(_identity "")
    foreach(_file "${_cpu}/gwpoly.cpp" "${_cpu}/gwpoly_plugin.cpp" "${_cpu}/gwpoly.h"
                  "${ECM_ROOT}/include/ecm/stage2_gw_abi.h" "${ECM_GWNUM_ROOT}/gwnum64.lib"
                  "${ECM_GWNUM_ROOT}/polymult64.lib")
        file(SHA256 "${_file}" _hash)
        string(APPEND _identity "${_hash}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
    endforeach()
    string(SHA256 _hash "${_identity}")
    string(SUBSTRING "${_hash}" 0 16 _hash)
    target_compile_definitions(ecm_gwnum PRIVATE "ECM_GW_BUILD=\"gwnum.dll ABI=1 build=${_hash}\"")
endif()
# Match the CRT used by the upstream Windows static library.
set_target_properties(ecm_gwnum PROPERTIES
    OUTPUT_NAME gwnum
    MSVC_RUNTIME_LIBRARY MultiThreaded)
