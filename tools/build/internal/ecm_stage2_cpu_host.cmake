# CPU-specific objects only. The GPU executable already owns the shared Stage2
# queue, result publishing and configuration code.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/ecm_libecm.cmake")
get_filename_component(_stage2_repo "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(_stage2_ecm_header "${ECM_LIBECM_SOURCE_DIR}/ecm-ecm.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_stage2_ecm_header}")
file(STRINGS "${_stage2_ecm_header}" _stage2_ntt_definition REGEX "^#define NTT_SIZE_THRESHOLD +[0-9]+")
if(NOT _stage2_ntt_definition MATCHES "NTT_SIZE_THRESHOLD +([0-9]+)")
    message(FATAL_ERROR "Cannot read GMP-ECM NTT limb threshold")
endif()
add_library(ecm_stage2_cpu_host STATIC
    "${_stage2_repo}/src/ecm/stage2/launch.cpp"
    "${_stage2_repo}/src/ecm/stage2/edwards/cpu/main.cpp"
    "${_stage2_repo}/src/ecm/stage2/cpu/gwpoly_loader.cpp"
    "${_stage2_repo}/src/ecm/stage2/cpu/runtime_tune.cpp"
    "${_stage2_repo}/src/ecm/platform/stop.cpp")
target_compile_features(ecm_stage2_cpu_host PRIVATE cxx_std_17)
target_include_directories(ecm_stage2_cpu_host PRIVATE "${_stage2_repo}/include" "${_stage2_repo}/src")
target_compile_definitions(ecm_stage2_cpu_host PRIVATE ECM_STAGE2_COMBINED NOMINMAX _CRT_SECURE_NO_WARNINGS
    "ECM_LIBECM_NTT_THRESHOLD=${CMAKE_MATCH_1}")
target_compile_options(ecm_stage2_cpu_host PRIVATE /O2 /utf-8)
target_link_libraries(ecm_stage2_cpu_host PUBLIC ecm_cpu_library OpenMP::OpenMP_CXX)
