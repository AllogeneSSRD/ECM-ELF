cmake_minimum_required(VERSION 3.22.1)
project("ecm")

set(MPA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../..")

set(ECM_ANDROID_GMP_ROOT "${ECM_ANDROID_GMP_ROOT}" CACHE PATH
    "Prefix with include/gmp.h and lib/libgmp.a (e.g. vcpkg arm64-android)")

message(STATUS "ECM Android: ECM_ANDROID_GMP_ROOT='${ECM_ANDROID_GMP_ROOT}'")

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        jni_utf8.cpp
        opencl_loader.cpp
        opencl_program_cache.cpp
        opencl_probe.cpp
        opencl_bench.cpp
        opencl_addsub_bench.cpp
        opencl_montsqr_bench.cpp
        kernel_assets.cpp
        sys_gpu_stats.cpp
        ecm_android_run.cpp
        opencl_android_shim.cpp
        opencl_ecm_log_android.cpp
        "${MPA_ROOT}/src/opencl_ecm_addsub_manifest.cpp"
        "${MPA_ROOT}/src/opencl_ecm_montsqr_manifest.cpp")

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE __ANDROID__ CL_TARGET_OPENCL_VERSION=120)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/CL"
        "${MPA_ROOT}/include"
        "${MPA_ROOT}/src"
        "${MPA_ROOT}/src/ecm/backend/opencl/runtime")

set(_ECM_GMP_HEADER "${ECM_ANDROID_GMP_ROOT}/include/gmp.h")
set(_ECM_GMP_LIB "${ECM_ANDROID_GMP_ROOT}/lib/libgmp.a")

if(ECM_ANDROID_GMP_ROOT AND EXISTS "${_ECM_GMP_HEADER}" AND EXISTS "${_ECM_GMP_LIB}")
    message(STATUS "ECM Android: GMP found at ${ECM_ANDROID_GMP_ROOT}")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ECM_HAVE_GMP=1)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ECM_ANDROID_GMP_ROOT}/include")
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
            "${MPA_ROOT}/src/ecm/application/configuration/legacy/ecm_expr.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/stage1/montgomery/entry.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/stage1/montgomery/engine.cpp"
            "${MPA_ROOT}/src/ecm/service/stage1/ecm_gpu_common.cpp"
            "${MPA_ROOT}/src/ecm/service/execution/legacy/ecm_params_init.cpp"
            "${MPA_ROOT}/src/ecm/storage/checkpoint/ecm_checkpoint.cpp"
            "${MPA_ROOT}/src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp"
            "${MPA_ROOT}/src/ecm/storage/save/ecm_save.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/stage1/montgomery/debug_utils.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/stage1/montgomery/montgomery.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/arithmetic/limb24.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/arithmetic/path_registry.cpp"
            "${MPA_ROOT}/src/ecm/application/configuration/runtime.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/diagnostics/selftest.cpp"
            "${MPA_ROOT}/src/ecm/backend/opencl/runtime/runtime.cpp")
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${_ECM_GMP_LIB}")
else()
    message(STATUS "ECM Android: GMP not found (header=${_ECM_GMP_HEADER})")
    message(STATUS "ECM Android: pass -DECM_ANDROID_GMP_ROOT=... from Gradle; see docs/DEV_GMP_SETUP.md")
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        android
        log
        dl)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")
