cmake_minimum_required(VERSION 3.20)
project(main)

# Config source, generated bindings, templates and reference must agree.
# This native hash guard avoids a Python dependency for ordinary builds.
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/build/internal/check_ecm_config.cmake)
add_custom_target(ecm_config_check ALL
    COMMAND ${CMAKE_COMMAND} "-DECM_CONFIG_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/build/internal/check_ecm_config.cmake"
    COMMENT "Checking generated ECM configuration")

# MSVC Debug CRT (MDd) has an incompatibility with GMP's custom allocator
# and our timestamped-log mutex contention pattern. Use Release CRT (/MD)
# for all configurations to avoid crashes. Debug symbols still available.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
    add_compile_definitions($<$<CONFIG:Debug>:_HAS_ITERATOR_DEBUGGING=0>)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)

# Desktop GMP (vcpkg x64-windows). Android uses ECM_ANDROID_GMP_ROOT in app/CMakeLists.txt.
#
# A locally rebuilt GMP with x86_64/zen3 (BMI2 + ADX) mpn kernels is preferred when
# present: it is 1.17-1.26x faster on stage-1 for >=1000-bit operands (see
# docs/architecture/STAGE1.md 搂10.2). Build it with third_party/build_gmp_zen3.sh
# (requires BMI2/ADX: Intel Haswell 2013+, AMD Excavator/Zen+).
# Override with -DECM_WINDOWS_GMP_ROOT=<prefix> to use a different GMP.
set(ECM_WINDOWS_GMP_ROOT "" CACHE PATH
    "Prefix with include/gmp.h and lib/gmp.lib (e.g. vcpkg installed/x64-windows)")
if(NOT ECM_WINDOWS_GMP_ROOT)
    set(_ecm_gmp_zen3 "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gmp-zen3/dist")
    if(EXISTS "${_ecm_gmp_zen3}/include/gmp.h" AND EXISTS "${_ecm_gmp_zen3}/lib/gmp.lib")
        set(ECM_WINDOWS_GMP_ROOT "${_ecm_gmp_zen3}")
        message(STATUS "GMP: using local zen3/BMI2 build at ${_ecm_gmp_zen3}")
    elseif(DEFINED ENV{VCPKG_ROOT})
        set(ECM_WINDOWS_GMP_ROOT "$ENV{VCPKG_ROOT}/installed/x64-windows")
    elseif(EXISTS "D:/code/vcpkg/installed/x64-windows/include/gmp.h")
        set(ECM_WINDOWS_GMP_ROOT "D:/code/vcpkg/installed/x64-windows")
    endif()
endif()
find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS "${ECM_WINDOWS_GMP_ROOT}/include")
find_library(GMP_LIBRARY NAMES gmp libgmp HINTS "${ECM_WINDOWS_GMP_ROOT}/lib")
# The vcpkg toolchain sets CMAKE_FIND_ROOT_PATH to its own installed/ triplet, so
# HINTS outside that root are ignored and find_path/find_library would silently
# return the vcpkg GMP even when ECM_WINDOWS_GMP_ROOT points elsewhere. Pin both
# to the configured prefix explicitly (an explicit -DGMP_INCLUDE_DIR/-DGMP_LIBRARY
# on the command line still wins for its own cache entry only if it is inside the
# prefix; use ECM_WINDOWS_GMP_ROOT to switch GMP builds).
if(ECM_WINDOWS_GMP_ROOT)
    if(EXISTS "${ECM_WINDOWS_GMP_ROOT}/include/gmp.h")
        set(GMP_INCLUDE_DIR "${ECM_WINDOWS_GMP_ROOT}/include" CACHE PATH "GMP include directory" FORCE)
    endif()
    if(EXISTS "${ECM_WINDOWS_GMP_ROOT}/lib/gmp.lib")
        set(GMP_LIBRARY "${ECM_WINDOWS_GMP_ROOT}/lib/gmp.lib" CACHE FILEPATH "GMP import library" FORCE)
    endif()
endif()
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR
        "GMP not found (include=${GMP_INCLUDE_DIR}, lib=${GMP_LIBRARY}). "
        "Install: vcpkg install gmp:x64-windows "
        "or pass -DECM_WINDOWS_GMP_ROOT=<prefix>")
endif()
message(STATUS "GMP: include=${GMP_INCLUDE_DIR} lib=${GMP_LIBRARY}")

option(ECM_BUILD_CPU_STAGE2 "Build standalone CPU Edwards Stage2" OFF)
if(ECM_BUILD_CPU_STAGE2)
    include(${CMAKE_CURRENT_SOURCE_DIR}/tools/build/internal/ecm_cpu_stage2.cmake)
endif()

function(ecm_target_link_gmp target)
    target_include_directories(${target} PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(${target} ${GMP_LIBRARY})
endfunction()

# OpenCL backend library (static) for tests and later integration.
# Includes the runtime-config TU so every linker of this lib gets ecm_runtime_config().
add_library(cgbn_opencl STATIC
    src/ecm/backend/opencl/runtime/runtime.cpp
    src/ecm/application/configuration/runtime.cpp
)

target_include_directories(cgbn_opencl PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cgbn_opencl PRIVATE
    OpenCL::OpenCL
)

target_compile_definitions(cgbn_opencl PRIVATE
    "ECM_KERNEL_ROOT_DEFAULT=\"${CMAKE_BINARY_DIR}/opencl_resources\""
)

# Production resources accompany the local worker and are independent of CXX
# objects. Keep the diagnostic Montgomery self-test in the same payload.
add_custom_target(ecm_opencl_resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/src/ecm/backend/opencl/resources"
        "${CMAKE_BINARY_DIR}/opencl_resources"
    COMMENT "Syncing OpenCL production resources")
add_dependencies(cgbn_opencl ecm_opencl_resources)

add_library(opencl_ecm_entry STATIC
    src/ecm/backend/opencl/stage1/montgomery/entry.cpp
)

target_include_directories(opencl_ecm_entry PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(opencl_ecm_entry ${GMP_LIBRARY})

# The Stage1 driver and CPU paths are identical source files in the OpenCL and
# CUDA executables. Keep one source inventory while compiling them per target:
# each executable still supplies its own GPU backend and compile definitions.
include(${CMAKE_SOURCE_DIR}/tools/build/internal/cuda_sources.cmake)
ecm_cuda_write_inventory("${CMAKE_BINARY_DIR}/cuda_sources.json")
# Source identity is host-only: changing result/reporting code must not force
# all CUDA kernel translation units to rebuild.
execute_process(COMMAND git rev-parse --short=12 HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _stage1_revision OUTPUT_STRIP_TRAILING_WHITESPACE)
set(_stage1_identity "${ECM_TIERS}|${CMAKE_BUILD_TYPE}")
foreach(_source IN LISTS ECM_STAGE1_COMMON_SOURCES)
    file(SHA256 "${CMAKE_SOURCE_DIR}/${_source}" _stage1_hash)
    string(APPEND _stage1_identity "${_stage1_hash}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/${_source}")
endforeach()
foreach(_source include/ecm/storage/results/publication.h src/ecm/storage/results/ecm_result_json.h
                src/ecm/core/group_order/group_order.h src/ecm/service/diagnostics/order_analysis.h
                include/ecm/core/stage1/selection.h include/ecm/core/stage1/scalar.h include/ecm/service/stage1/task.h
                src/ecm/service/stage1/execution_contract.h src/ecm/platform/generated/configuration_defaults.h
                src/ecm/backend/opencl/stage1/montgomery/workflow.cpp
                src/ecm/backend/cuda/stage1/montgomery/workflow.cpp
                src/ecm/backend/cuda/stage1/edwards/workflow.cpp
                src/ecm/backend/cuda/stage1/edwards/checkpoint.cpp
                src/ecm/backend/cuda/stage1/edwards/host_math.cpp
                src/ecm/backend/cuda/stage1/montgomery/host_math.cpp src/ecm/backend/cuda/stage1/montgomery/checkpoint.cpp src/ecm/backend/cuda/stage1/montgomery/output.cpp
                src/ecm/backend/cuda/stage1/montgomery/host_policy.cpp
                src/ecm/storage/save/ecm_edwards_io.h include/ecm/platform/program_version.h
                include/ecm/platform/stage1_progress.h)
    file(SHA256 "${CMAKE_SOURCE_DIR}/${_source}" _stage1_hash)
    string(APPEND _stage1_identity "${_stage1_hash}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/${_source}")
endforeach()
string(SHA256 _stage1_identity "${_stage1_identity}")
string(SUBSTRING "${_stage1_identity}" 0 12 _stage1_identity)
set(_stage1_identity_file "${CMAKE_CURRENT_BINARY_DIR}/ecm_stage1_source_identity.txt")
file(GENERATE OUTPUT "${_stage1_identity_file}" CONTENT "${_stage1_revision}.s1.${_stage1_identity}\n")
# A post-link sidecar leaves flags.make and CUDA response files unchanged.
function(ecm_stage1_identity_sidecar target)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_stage1_identity_file}" "$<TARGET_FILE_DIR:${target}>/ecm_stage1_build_identity.txt")
endfunction()

add_executable(ecm
    ${ECM_STAGE1_COMMON_SOURCES}
    src/ecm/backend/opencl/stage1/montgomery/workflow.cpp
    src/ecm/backend/opencl/stage1/montgomery/engine.cpp
    src/ecm/backend/opencl/factory.cpp
    src/ecm/backend/opencl/runtime/device.cpp
    src/ecm/backend/opencl/stage1/montgomery/debug_utils.cpp
    src/ecm/backend/opencl/stage1/montgomery/montgomery.cpp
    src/ecm/backend/opencl/arithmetic/path_registry.cpp
    src/ecm/backend/opencl/diagnostics/selftest.cpp
)
ecm_stage1_identity_sidecar(ecm)

# 鍙湁杩欎袱涓?SIMD TU 鎷?AVX512 codegen; 璋冪敤鐐瑰繀椤诲湪 edwards_simd_available()
# (鍩虹嚎 TU 閲岀殑 CPUID 鎺㈡祴) 杩斿洖 1 涔嬪悗, 瑙?docs/architecture/STAGE1.md 搂13.8銆?
if(MSVC)
    set_source_files_properties(src/ecm/backend/cpu/stage1/edwards/simd_edwards.cpp src/ecm/backend/cpu/arithmetic/ifma/simd_mont_ifma.cpp
        src/ecm/backend/cpu/stage1/montgomery/simd_mont_curve.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/ecm/backend/cpu/stage1/edwards/simd_edwards.cpp src/ecm/backend/cpu/arithmetic/ifma/simd_mont_ifma.cpp
        src/ecm/backend/cpu/stage1/montgomery/simd_mont_curve.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512ifma;-mavx512dq")
endif()

target_include_directories(ecm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(ecm
    OpenCL::OpenCL
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
    opencl_ecm_entry
)

ecm_target_link_gmp(ecm)

# indicators.hpp contains Unicode block characters for progress
# bars. MSVC's GBK code page misparses them; /utf-8 tells the compiler to
# treat source files as UTF-8 regardless of system locale.
if(MSVC)
    target_compile_options(ecm PRIVATE /utf-8)
endif()

# Copy GMP runtime DLL next to the output binary. The configured GMP prefix
# (ECM_WINDOWS_GMP_ROOT) wins over the vcpkg default so a custom GMP build --
# e.g. the x86_64/zen3 (BMI2) rebuild under third_party/gmp-zen3 -- is picked up
# automatically instead of the generic-kernel vcpkg DLL.
foreach(_gmp_dll_cand "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll"
                      "${OPENSSL_ROOT_DIR}/bin/gmp-10.dll")
    if(MSVC AND EXISTS "${_gmp_dll_cand}")
        add_custom_command(TARGET ecm POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_gmp_dll_cand}"
                "$<TARGET_FILE_DIR:ecm>"
            COMMENT "Copying GMP DLL (${_gmp_dll_cand}) to output dir")
        break()
    endif()
endforeach()

# Copy OpenSSL + GMP runtime DLLs from vcpkg to output dir
if(MSVC AND EXISTS "${OPENSSL_ROOT_DIR}/bin")
    set(_vcpkg_bin "${OPENSSL_ROOT_DIR}/bin")
    # add_custom_command(TARGET ecm POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #         "${_vcpkg_bin}/libcrypto-3-x64.dll"
    #         "${_vcpkg_bin}/libssl-3-x64.dll"
    #         "$<TARGET_FILE_DIR:ecm>"
    #     COMMENT "Copying OpenSSL DLLs to output dir")
endif()
#
#  ecm_cuda native CUDA/CGBN backend (mirror of the OpenCL `ecm` executable)
#
#  Reuses the exact same driver / argument parsing / checkpoint / save / logging
#  translation units as `ecm`, swapping only the GPU backend at link time via the
#  ecm_backend_* seam (include/ecm/service/execution/legacy_gpu.h):
#     - OpenCL: src/ecm/backend/opencl/factory.cpp  (in `ecm`)
#     - CUDA:   src/ecm/backend/cuda/factory.cpp  (here)
#
#  Auto-enabled when a CUDA toolkit is present; force off with -DECM_ENABLE_CUDA=OFF.
#
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(ECM_ENABLE_CUDA "Build the ecm_cuda (CUDA/CGBN) executable" ON)
else()
    message(STATUS "CUDA compiler not found; ecm_cuda target disabled")
    set(ECM_ENABLE_CUDA OFF)
endif()

set(ECM_CUDA_ARCHITECTURES "80" CACHE STRING
    "CUDA compute capabilities for ecm_cuda (e.g. 89 for RTX 40-series)")

if(ECM_ENABLE_CUDA)
    enable_language(CUDA)

    option(ECM_BUILD_CUDA_STAGE2 "Build standalone Stage1-save CUDA Stage2 driver" OFF)
    if(ECM_BUILD_CUDA_STAGE2)
        if(NOT WIN32)
            message(FATAL_ERROR "The minimal ecm_cuda_stage2 process/queue driver currently requires Windows")
        endif()
        add_executable(ecm_cuda_stage2
            ${ECM_STAGE2_PRODUCTION_SOURCES}
            ${ECM_STAGE2_HOST_SOURCES})
        target_include_directories(ecm_cuda_stage2 PRIVATE
            ${GMP_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/include
            "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>")
        target_compile_definitions(ecm_cuda_stage2 PRIVATE NTT_GL_FIXED_MODE=3 NTT_OUTER_UNROLL_U=0 NTT_GL_ADD_SUB_MASK=1)
        target_link_libraries(ecm_cuda_stage2 PRIVATE ${GMP_LIBRARY})
        set_target_properties(ecm_cuda_stage2 PROPERTIES
            CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${ECM_CUDA_ARCHITECTURES}")
        target_compile_options(ecm_cuda_stage2 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
        if(EXISTS "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll")
            add_custom_command(TARGET ecm_cuda_stage2 POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll"
                    "$<TARGET_FILE_DIR:ecm_cuda_stage2>")
        endif()
    endif()

    set(CGBN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/cgbn/include" CACHE PATH "CGBN include directory")
    if(NOT EXISTS "${CGBN_INCLUDE_DIR}/cgbn/cgbn.h")
        message(FATAL_ERROR "CGBN header not found at ${CGBN_INCLUDE_DIR}/cgbn/cgbn.h")
    endif()
    option(ECM_CUDA_ENABLE_PRAC "Build experimental resident/PRAC Stage1 kernels (slow template compilation)" OFF)

    include("${CMAKE_SOURCE_DIR}/tools/build/internal/cuda_timing.cmake")
    option(ECM_CUDA_ISOLATE_TIERS "One device module per curve/param/container" OFF)
    if(ECM_CUDA_ISOLATE_TIERS)
        include("${CMAKE_SOURCE_DIR}/tools/build/internal/cuda_modules.cmake")
        ecm_stage1_isolate_modules()
        add_compile_definitions(ECM_CUDA_ISOLATE_TIERS)
    endif()
    add_executable(ecm_cuda
        ${ECM_STAGE1_COMMON_SOURCES}
        ${ECM_STAGE1_CUDA_SOURCES})
    ecm_stage1_identity_sidecar(ecm_cuda)

    if(ECM_CUDA_ENABLE_PRAC)
        add_library(ecm_cuda_prac OBJECT ${ECM_STAGE1_PRAC_SOURCES})
        target_sources(ecm_cuda PRIVATE $<TARGET_OBJECTS:ecm_cuda_prac>)
    endif()
    target_compile_definitions(ecm_cuda PRIVATE ECM_HAS_CUDA_STAGE1=1 ECM_CUDA_ENABLE_PRAC=$<BOOL:${ECM_CUDA_ENABLE_PRAC}>)
    # CGBN uses uint32_t without including stdint.h. CUDA 12.6 does not provide
    # that transitive include; keep the compatibility fix in our tracked build.
    target_compile_options(ecm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=stdint.h>)
    message(STATUS "ecm_cuda: experimental resident/PRAC kernels ${ECM_CUDA_ENABLE_PRAC}")

    set_target_properties(ecm_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                              CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(ecm_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/runtime
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>"
        ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda
        ${CGBN_INCLUDE_DIR}
        ${CGBN_INCLUDE_DIR}/cgbn   # the .cu uses #include <cgbn.h>
        ${GMP_INCLUDE_DIR}
    )

    # ---------------------------------------------------------------------------------
    # Device-code size.  The kernel set is huge (the stage-1 exponent is 375 Mbit at
    # B1 = 260e6, and every tier is instantiated for three parametrizations), so the
    # embedded cubins dominate the executable: a full build was 70.4 MB before these
    # two options.
    #
    # Measured on one tier-restricted TU (suyama_tpi16, tier 5120):
    #     PTX + SASS, uncompressed : 853,000 B      <- CMake's default (arch=89)
    #     SASS only ("89-real")    : 690,160 B      (-19%: drops the compute_89 PTX)
    #     PTX + SASS, --compress   : 322,168 B      (-62%)
    #     SASS only,   --compress  : 159,328 B      (-81%)   <- both options below
    #
    # ECM_CUDA_EMBED_PTX=ON keeps the compute_XX PTX so the binary can JIT onto a *newer*
    # architecture.  OFF (default) therefore drops forward compatibility: an sm_89-only
    # build refuses to run on sm_90+ with "no kernel image is available".  Re-enable it if
    # you build on one machine and run on another with a newer GPU.
    set(ECM_CUDA_EMBED_PTX "OFF" CACHE STRING "1 = also embed PTX (JIT to newer archs; bigger binaries)")
    set(ECM_CUDA_COMPRESS  "ON"  CACHE STRING "1 = compress the embedded fatbin (-Xfatbin -compress-all)")
    set(ECM_CUDA_PTX_ARCH "" CACHE STRING "one independent virtual PTX target")

    if(ECM_CUDA_EMBED_PTX)
        set(_ecm_cuda_arches "${ECM_CUDA_ARCHITECTURES}")
    else()
        # CMake's "-real" suffix means "SASS for this arch, no PTX".
        set(_ecm_cuda_arches "")
        foreach(_a IN LISTS ECM_CUDA_ARCHITECTURES)
            if(_a MATCHES "-real$")
                list(APPEND _ecm_cuda_arches "${_a}")
            else()
                list(APPEND _ecm_cuda_arches "${_a}-real")
            endif()
        endforeach()
    endif()

    # Independent virtual and real targets: e.g. real 60 + virtual 50 for cu11.8,
    # or real 80 + virtual 75 for cu13.3. A PTX target need not be a SASS member.
    if(ECM_CUDA_PTX_ARCH)
        set(_ecm_cuda_arches "")
        foreach(_a IN LISTS ECM_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-real$" "" _real "${_a}")
            list(APPEND _ecm_cuda_arches "${_real}-real")
        endforeach()
        list(APPEND _ecm_cuda_arches "${ECM_CUDA_PTX_ARCH}-virtual")
        message(STATUS "ecm_cuda: PTX embedded for ${ECM_CUDA_PTX_ARCH} only; "
                       "device code: ${_ecm_cuda_arches}")
    endif()

    set_target_properties(ecm_cuda PROPERTIES
        CUDA_ARCHITECTURES "${_ecm_cuda_arches}"
    )

    if(ECM_CUDA_COMPRESS)
        target_compile_options(ecm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin;-compress-all>)
        message(STATUS "ecm_cuda: fatbin compression ON, device code for ${_ecm_cuda_arches}")
    else()
        message(STATUS "ecm_cuda: device code for ${_ecm_cuda_arches} (uncompressed)")
    endif()
    if(NOT ECM_CUDA_EMBED_PTX AND NOT ECM_CUDA_PTX_ARCH)
        message(STATUS "ecm_cuda: no PTX embedded -- needs exactly the arch above "
                       "(set -DECM_CUDA_EMBED_PTX=ON to JIT onto newer GPUs)")
    endif()

    # Dev build (kernels up to 1024 bits) is the default; enable the full CGBN
    # kernel set (larger N, much longer compile) with -DECM_CUDA_FULL_BUILD=ON.
    # Without it ecm_cuda refuses N >= 1024 bits ("Warning: Dev build, only support
    # N<1024"), which makes it unusable for the sizes this project targets -- the
    # dedicated NMake build in the README sets it ON.
    option(ECM_CUDA_FULL_BUILD "Compile the full CGBN kernel set instead of the dev build (N<=1024)" OFF)
    if(ECM_CUDA_FULL_BUILD)
        target_compile_definitions(ecm_cuda PRIVATE ECM_CUDA_FULL_BUILD)
    endif()

    # PROBE ONLY (docs/performance/STAGE1.md): make the fused double-and-add run its
    # addition half only every k-th bit, to measure the timing ceiling of an add-chain
    # (PRAC / NAF + dictionary) schedule.  k > 1 IS MATHEMATICALLY WRONG -- results will be
    # garbage; it exists to answer "how much would a chain with 1/k add density save?".
    # Default 1 = the correct kernel.
    set(ECM_PROBE_ADD_DENSITY "1" CACHE STRING
        "PROBE ONLY: run the fused add every k-th bit (k>1 = wrong results, for timing)")
    target_compile_definitions(ecm_cuda PRIVATE ECM_PROBE_ADD_DENSITY=${ECM_PROBE_ADD_DENSITY})

    # PROBE ONLY (docs/performance/STAGE1.md): replace the ladder step with a
    # windowed-chain op mix -- one xz doubling per bit plus one REAL differential addition
    # with a PROJECTIVE difference point (4M+2S) every M-th bit.  M=1 is the PRAC-like mix,
    # M=w+1 the ideal width-w NAF density.  RESULTS ARE WRONG (timing only).
    # Default 0 = probe off, correct kernel.
    set(ECM_PROBE_CHAIN_W "0" CACHE STRING
        "PROBE ONLY: chain op mix with one projective-difference add every M bits (wrong results)")
    target_compile_definitions(ecm_cuda PRIVATE ECM_PROBE_CHAIN_W=${ECM_PROBE_CHAIN_W})

    # Kernel launch/occupancy tunables (docs/performance/STAGE1.md).  TPB must
    # match the launch config in src/ecm/backend/cuda/stage1/montgomery/runtime.cu, hence compile-time.  Defaults
    # are the MEASURED best from the 2026-09-25 sweep (M511/M761, 8192 curves, B1=1e5,
    # param3, GPU 1): the historical 256/4 measured 78.18 M curve-bits/s at M511, 128/1
    # measures 79.15 M, and blocks = curves/(TPB/TPI) is what actually drives throughput
    # (TPB=512 halves the block count and measures -15%).
    set(ECM_TPB "128" CACHE STRING "CUDA block size for the CGBN stage-1 kernel")
    set(ECM_MAX_ROTATION "1" CACHE STRING "CGBN MAX_ROTATION (limb rotation in the multiply)")
    target_compile_definitions(ecm_cuda PRIVATE ECM_TPB=${ECM_TPB} ECM_MAX_ROTATION=${ECM_MAX_ROTATION})

    # Register cap for the SMALL-tier kernels only (<=2048 bits: the tpi4/tpi8 sources).
    # Measured: capping 72 -> 56 registers raises occupancy from 7 to 9 blocks/SM and buys
    # +4.7% at M511 and M761 (48 overflows and gives it back).  The >=2560-bit instantiations
    # live in other sources and keep the compiler default, because their allocation is
    # unmeasured and a 56-register cap there could spill badly.  Set to 0 to disable.
    # DEFAULT 0: the per-tier __maxnreg__(cgbn_params_t::REG_TARGET) in the kernel header now
    # encodes the register budget per bit width, which is strictly better than a per-file
    # --maxrregcount.  Set this to a number only to override it as a blunt instrument.
    set(ECM_MAXRREG_SMALL "0" CACHE STRING "maxrregcount for the <=2048-bit kernel sources (0 = off)")
    if(NOT ECM_MAXRREG_SMALL STREQUAL "0")
        set_source_files_properties(
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g0.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param3_g1.cu
            PROPERTIES COMPILE_OPTIONS "--maxrregcount=${ECM_MAXRREG_SMALL}")
    endif()

    # Same treatment for the param0 (Suyama) sources -- the *recommended* parametrization --
    # and for the param2 family, which is the same kernel body instantiated with
    # CONST_DIFF = true (it MUST be capped too: without the cap the param2 path measured
    # 74.12 M curve-bits/s instead of the 79+ M its op count predicts).
    # Measured 2026-09-25 at M511/8192 curves on the param0 path: cap 0 -> 71.07 M, 64 ->
    # 71.68 M (+0.9%), 56 -> 71.25 M.  So param0 gains far less than param3 (+4.7%) because
    # it executes 10 mont ops per bit instead of 8.2 and spills sooner -- its remaining gap
    # to param3 (71.7 vs 82.0 M) is the op count, not occupancy.  64 is the measured best.
    # NOTE: a per-source-file cap applies to EVERY tier in that file, including the
    # >=2560-bit instantiations whose allocation is unmeasured -- flagged in
    # docs/performance/STAGE1.md
    # DEFAULT 0: a per-FILE cap also hits that file's >=2560-bit instantiations, which the
    # measurement above shows should stay unconstrained (they want ~141 registers).  The
    # <=2048-bit tiers inside these files are handled by the tier-aware MIN_BLOCKS instead.
    set(ECM_MAXRREG_SUYAMA "0" CACHE STRING "maxrregcount for the param0/Suyama + param2 kernel sources (0 = off)")
    if(NOT ECM_MAXRREG_SUYAMA STREQUAL "0")
        set_source_files_properties(
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g0.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g1.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g2.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g3.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g4.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g5.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param0_g6.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g0.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g1.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g2.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g3.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g4.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g5.cu
            ${CMAKE_SOURCE_DIR}/src/ecm/backend/cuda/stage1/montgomery/instances/instances_param2_g6.cu
            PROPERTIES COMPILE_OPTIONS "--maxrregcount=${ECM_MAXRREG_SUYAMA}")
    endif()

    # Tier filter for fast iteration (docs/performance/STAGE1.md).  On this machine
    # every parallel generator fails (Ninja hangs in compiler ABI detection, MSBuild's CUDA
    # 13.3 target cancels the build, jom breaks CMake's try_compile), so kernel builds stay
    # serial and a full build costs ~20 min.  -DECM_TIERS=4096,8192 compiles ONLY those
    # instantiations, turning the 12-tier tpi16 TU into a couple of seconds.  Empty = all.
    set(ECM_TIERS "" CACHE STRING "comma-separated BITS list to compile (empty = all tiers)")
    if(NOT ECM_TIERS STREQUAL "")
        string(REPLACE "," ";" tier_list "${ECM_TIERS}")
        target_compile_definitions(ecm_cuda PRIVATE ECM_TIERS_RESTRICTED)
        foreach(tier IN LISTS tier_list)
            string(STRIP "${tier}" tier)
            target_compile_definitions(ecm_cuda PRIVATE ECM_TIER_${tier})
            # The 512- and 1024-bit tiers are guarded by their CGBN TYPEDEF NAME, not by
            # their number (cgbn_params_small = 512, cgbn_params_medium = 1024), so
            # -DECM_TIERS=1024 used to define ECM_TIER_1024 while every guard asked for
            # ECM_TIER_medium: the tier list silently compiled NO 1024-bit kernel and the
            # dispatch fell through to the next tier (observed: a 991-bit N ran on the
            # 4608-bit suyama kernel, 5x slower, with t = 3617).  Define both spellings.
            if(tier STREQUAL "512")
                target_compile_definitions(ecm_cuda PRIVATE ECM_TIER_small)
            elseif(tier STREQUAL "1024")
                target_compile_definitions(ecm_cuda PRIVATE ECM_TIER_medium)
            endif()
        endforeach()
        message(STATUS "ecm_cuda: kernel tiers restricted to ${tier_list}")
    endif()

    # A/B override for the per-tier register budget.  0 = use the table in
    # cgbn_stage1_kernel.h (56 regs for <=2048-bit tiers, uncapped above); any other
    # value forces one __maxnreg__ target on EVERY tier (255 = explicitly uncapped).
    # NOTE: this is a *kernel* attribute, not a per-file --maxrregcount, which is why
    # it can be set per bit width; see docs/performance/STAGE1.md 5.7.
    set(ECM_REG_TARGET_FORCE "0" CACHE STRING "force one __maxnreg__ register target for all tiers (0 = per-tier table)")
    if(NOT ECM_REG_TARGET_FORCE STREQUAL "0")
        target_compile_definitions(ecm_cuda PRIVATE ECM_REG_TARGET_FORCE=${ECM_REG_TARGET_FORCE})
        message(STATUS "ecm_cuda: forcing __maxnreg__(${ECM_REG_TARGET_FORCE}) on every kernel tier")
    endif()

    # A/B the two orderings of the fused ladder step (same arithmetic; see the header).
    set(ECM_STEP_VARIANT "1" CACHE STRING "1 = double_add_v2 (historical), 2 = explicit-prep variant")
    target_compile_definitions(ecm_cuda PRIVATE ECM_STEP_VARIANT=${ECM_STEP_VARIANT})

    # param2 is a SECOND FULL COPY of every tier (2560..16384 included), i.e. a large slice of
    # the kernel compile time, and it is still opt-in at runtime (--gpu-param 2).  Set this to 1
    # to skip those instantiations entirely while working on the param0/param3 paths;
    # --gpu-param 2 then fails with an explicit "not compiled into this binary" error.
    set(ECM_NO_PARAM2 "0" CACHE STRING "1 = do not compile the param2 kernel instantiations")
    target_compile_definitions(ecm_cuda PRIVATE ECM_NO_PARAM2=${ECM_NO_PARAM2})
    if(ECM_NO_PARAM2)
        message(STATUS "ecm_cuda: param2 kernel instantiations DISABLED (-DECM_NO_PARAM2=1)")
    endif()

    # PROBE ONLY: price the param2-shaped kernel (5M+4S) on the param0 data path.  Wrong
    # results, timing only -- see docs/performance/STAGE1.md
    set(ECM_PARAM2SHAPE "0" CACHE STRING "PROBE ONLY: run the param2-shaped step on the param0 path")
    target_compile_definitions(ecm_cuda PRIVATE ECM_PARAM2SHAPE=${ECM_PARAM2SHAPE})

    # Mersenne fold domain for the suyama family (param0 + param2).  For N = 2^k - 1 the
    # Montgomery reduction of every modular multiply is replaced by a fold (2^k == 1), which
    # removes CGBN's Q*N chains -- the GPU analogue of the CPU IFMA fold domain.  Measured
    # -21%..-31% per modular multiply on the 4060 Laptop at 4608/5120/8192 bits
    # (tools/bench/cgbn_mers_fold_probe.cu, docs/performance/STAGE1.md).
    #
    # This is a WHOLE-BUILD switch: a binary built with -DECM_MERS_FOLD=1 only carries fold
    # kernels, so it REFUSES any N that is not 2^k - 1 and refuses --gpu-param 3
    # (src/ecm/backend/cuda/stage1/montgomery/runtime.cu).  Default 0 keeps the historical Montgomery kernels.
    set(ECM_MERS_FOLD "0" CACHE STRING "1 = suyama family runs the Mersenne fold domain (N = 2^k-1 only)")
    target_compile_definitions(ecm_cuda PRIVATE ECM_MERS_FOLD=${ECM_MERS_FOLD})
    if(ECM_MERS_FOLD)
        message(STATUS "ecm_cuda: MERSENNE FOLD domain enabled for param0/param2 (-DECM_MERS_FOLD=1)")
    endif()

    # Pascal sm_60 / Maxwell workaround: with NDEBUG (Release), CUDA 12.x fails to
    # unroll one of CGBN's carry chains on the compute_60 PTX path, so CGBN emits
    # its intentional poison ("ASM_ERROR chain destructor: ... loop not unrolled")
    # and ptxas aborts with a parse error near 'destructor'. Undefining NDEBUG for
    # the CUDA TUs restores unrolling. compute_61/70/80/89 are unaffected, so we
    # only apply it when an architecture <= 60 is targeted.
    set(_ecm_cuda_low_arch OFF)
    foreach(_a IN LISTS ECM_CUDA_ARCHITECTURES)
        string(REGEX MATCH "^[0-9]+" _an "${_a}")
        if(_an AND (_an LESS 61))
            set(_ecm_cuda_low_arch ON)
        endif()
    endforeach()
    if(_ecm_cuda_low_arch)
        target_compile_options(ecm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
        message(STATUS "ecm_cuda: undefining NDEBUG for CUDA (sm<=60 CGBN chain-unroll workaround)")
    endif()

    # Register cap for the CGBN stage-1 kernel (docs/performance/STAGE1.md).  The
    # occupancy sweep showed throughput tracks the number of resident blocks, so trading a
    # few registers (spills) for more warps per SM can pay off.  0 = compiler default.
    set(ECM_MAXRREG "0" CACHE STRING "nvcc --maxrregcount for ecm_cuda (0 = default)")
    if(NOT ECM_MAXRREG STREQUAL "0")
        target_compile_options(ecm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${ECM_MAXRREG}>)
    endif()

    target_link_libraries(ecm_cuda ${GMP_LIBRARY})

    if(MSVC)
        # indicators.hpp uses Unicode block glyphs; force UTF-8 for host C++ TUs.
        target_compile_options(ecm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
        target_compile_options(ecm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    endif()

    # Keep the runtime from the same prefix as the selected GMP import library.
    # OPENSSL_ROOT_DIR may point at a different (generic vs Zen3) GMP build.
    get_filename_component(_ecm_cuda_gmp_lib_dir "${GMP_LIBRARY}" DIRECTORY)
    get_filename_component(_ecm_cuda_gmp_dll "${_ecm_cuda_gmp_lib_dir}/../bin/gmp-10.dll" ABSOLUTE)
    if(MSVC AND EXISTS "${_ecm_cuda_gmp_dll}")
        add_custom_command(TARGET ecm_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ecm_cuda_gmp_dll}"
                "$<TARGET_FILE_DIR:ecm_cuda>"
            COMMENT "Copying GMP DLL to ecm_cuda output dir")
    endif()

    message(STATUS "ecm_cuda enabled (CUDA arch ${ECM_CUDA_ARCHITECTURES}, CGBN ${CGBN_INCLUDE_DIR})")
endif()

# ===========================================================================
# Optional helper targets: bench / test / disasm / p95feeder.
#
# These used to be defined in this file.  They now live in their own
# tools/<group>/CMakeLists.txt so that this file stays about the two shipped
# executables (`ecm` and `ecm_cuda`) plus the libraries they need.
#
#   -DECM_BUILD_TOOLS=OFF   configure without any helper target (faster configure and a
#                           smaller build tree).  `ecm`/`ecm_cuda` are unaffected: none of
#                           the helpers is a dependency of either.
# ===========================================================================
option(ECM_BUILD_TOOLS "Build the bench/test/disasm/p95feeder helper executables under tools/" ON)

# Helper executables keep landing in the build ROOT (not in tools/<group>/), because the
# docs and the tools/*.ps1 scripts reference them by path:
#   build_vs18\Release\simd_mont_gate.exe, build_vs18\Release\ecm_p95feeder.exe,
#   build_cuda_cmake\cpu_mont_bench.exe ...
# (VS appends the configuration directory to this, i.e. exactly today's layout.)
function(ecm_tool_output_root target)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

if(ECM_BUILD_TOOLS)
    add_subdirectory(tools/bench)
    add_subdirectory(tools/test)
    add_subdirectory(tools/disasm)
    add_subdirectory(tools/p95feeder)
    add_custom_target(ecm_opencl_tool_resources
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/kernels/opencl/bench"
            "${CMAKE_BINARY_DIR}/opencl_resources/bench"
        COMMENT "Syncing OpenCL tool resources")
    foreach(_tool opencl_ecm_addsub opencl_ecm_montsqr main
                  opencl_asm_selftest opencl_mont_isa_export opencl_addsub_isa_export)
        add_dependencies(${_tool} ecm_opencl_tool_resources)
    endforeach()
    message(STATUS "tools: bench/test/disasm/p95feeder helper targets enabled (ECM_BUILD_TOOLS=ON)")
else()
    message(STATUS "tools: helper targets disabled (ECM_BUILD_TOOLS=OFF)")
endif()
# ===========================================================================
# ecm_gui -- the graphical front-end (docs/usage/GUI.md).
#
# Deliberately independent of ECM_BUILD_TOOLS: it is a deliverable, not a helper,
# and it is Windows-only (Dear ImGui + Win32 + Direct3D 11 + NVML).
# It must NEVER break a configure on a machine that cannot build it: the sources
# live in an optional vendored tree (third_party/imgui) and a missing tree only
# turns this target off, with a status message.
# ===========================================================================
option(ECM_BUILD_GUI "Build the ecm_gui front-end (Windows only, needs third_party/imgui)" ON)
set(ECM_IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui" CACHE PATH
    "Dear ImGui source directory used by ecm_gui (docking branch)")

if(NOT ECM_BUILD_GUI)
    message(STATUS "gui: disabled (ECM_BUILD_GUI=OFF)")
elseif(NOT WIN32)
    message(STATUS "gui: skipped (ecm_gui is Windows-only for now; see docs/usage/GUI.md)")
elseif(NOT EXISTS "${ECM_IMGUI_DIR}/imgui.h")
    message(STATUS "gui: skipped -- ${ECM_IMGUI_DIR}/imgui.h not found. Vendor Dear ImGui "
                   "(docking branch) there; see third_party/imgui/README.md")
else()
    add_subdirectory(src/ecm/application/gui)
    message(STATUS "gui: ecm_gui enabled (ImGui from ${ECM_IMGUI_DIR})")
endif()

# Shared optional order/factor diagnostics; math library stays in its own host.
if(WIN32)
    set(ECM_DIAGNOSTICS_GMP_ROOT "${ECM_WINDOWS_GMP_ROOT}" CACHE PATH "Diagnostic host GMP prefix" FORCE)
    add_subdirectory(src/ecm/service/diagnostics)
    foreach(_worker ecm ecm_cuda ecm_cuda_stage2)
        if(TARGET ${_worker})
            add_dependencies(${_worker} ecm_diagnostics_runtime)
        endif()
    endforeach()
endif()

# GUI v2 is opt-in and independent of the legacy GUI target and its renderer.
option(ECM_BUILD_GUI_V2 "Build the independent Win32/Vulkan GUI v2" OFF)
if(ECM_BUILD_GUI_V2)
    add_subdirectory(src/ecm/application/gui_v2)
endif()

# Explicit targets must run the guard too (not only the default ALL build).
foreach(_ecm_config_target ecm ecm_cuda ecm_cuda_stage2 ecm_gui ecm_gui_v2)
    if(TARGET ${_ecm_config_target})
        add_dependencies(${_ecm_config_target} ecm_config_check)
    endif()
endforeach()

# Export object owners for build/dependency identity without changing CUDA flags.
ecm_cuda_write_target_graph()

if(TARGET ecm_cuda_prac)
    add_dependencies(ecm_cuda_prac ecm_config_check)
    foreach(_property INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS
            CXX_STANDARD CXX_STANDARD_REQUIRED CUDA_STANDARD CUDA_STANDARD_REQUIRED CUDA_ARCHITECTURES)
        get_target_property(_value ecm_cuda ${_property})
        if(NOT _value STREQUAL "_value-NOTFOUND")
            set_property(TARGET ecm_cuda_prac PROPERTY ${_property} "${_value}")
        endif()
    endforeach()
endif()
