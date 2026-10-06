cmake_minimum_required(VERSION 3.20)
project(main)

# MSVC Debug CRT (MDd) has an incompatibility with GMP's custom allocator
# and our timestamped-log mutex contention pattern. Use Release CRT (/MD)
# for all configurations to avoid crashes. Debug symbols still available.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
    add_compile_definitions($<$<CONFIG:Debug>:_HAS_ITERATOR_DEBUGGING=0>)
endif()

# set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)

# Desktop GMP (vcpkg x64-windows). Android uses ECM_ANDROID_GMP_ROOT in app/CMakeLists.txt.
#
# A locally rebuilt GMP with x86_64/zen3 (BMI2 + ADX) mpn kernels is preferred when
# present: it is 1.17-1.26x faster on stage-1 for >=1000-bit operands (see
# docs/ECM_EDWARDS_STAGE1.md 搂10.2). Build it with third_party/build_gmp_zen3.sh
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

function(ecm_target_link_gmp target)
    target_include_directories(${target} PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(${target} ${GMP_LIBRARY})
endfunction()

# OpenCL backend library (static) for tests and later integration.
# Includes the runtime-config TU so every linker of this lib gets ecm_runtime_config().
add_library(cgbn_opencl STATIC
    kernels/opencl/impl_opencl.cpp
    src/opencl_ecm_runtime_config.cpp
)

target_include_directories(cgbn_opencl PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/kernels/opencl
)

target_link_libraries(cgbn_opencl PRIVATE
    OpenCL::OpenCL
)

target_compile_definitions(cgbn_opencl PRIVATE
    "ECM_KERNEL_ROOT_DEFAULT=\"${CMAKE_SOURCE_DIR}/kernels/opencl\""
)

add_library(opencl_ecm_entry STATIC
    src/opencl_ecm_entry.cpp
)

target_include_directories(opencl_ecm_entry PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(opencl_ecm_entry ${GMP_LIBRARY})

add_executable(ecm
    src/opencl_ecm_stage1.cpp
    src/opencl_backend_glue.cpp
    src/cl_probe.cpp
    src/core/ecm_driver.cpp
    src/core/ecm_params_init.cpp
    src/core/ecm_checkpoint.cpp
    src/core/ecm_save.cpp
    src/core/ecm_gpu_common.cpp
    src/core/ecm_queue_config.cpp
    src/core/ecm_worktodo.cpp
    # Prime95 worktodo.add handoff for finished tasks (docs/DEV_ECM_GUI.md 18) plus the
    # p95_worktodo helpers it shares with tools/p95feeder.
    src/core/p95_transfer.cpp
    src/core/p95_worktodo.cpp
    src/opencl_ecm_debug_utils.cpp
    src/opencl_ecm_log.cpp
    src/opencl_ecm_mont.cpp
    src/opencl_ecm_path_registry.cpp
    src/opencl_ecm_selftest.cpp
    src/cpu/ecm_edwards_cpu.cpp
    src/cpu/ecm_edwards_save.cpp
    src/cpu/simd_edwards.cpp
    src/cpu/simd_mont_ifma.cpp
    # Suyama-sigma Montgomery stage 1 (Prime95 sigma_type=1 / gmp-ecm -param 0):
    # scalar MPN path + the 8-lane AVX512-IFMA batch.  See
    # docs/ECM_Montgomery_STAGE1.md.
    src/cpu/ecm_mont_cpu.cpp
    src/cpu/simd_mont_curve.cpp
    # Mid-stage-1 checkpoints for both of those paths (internal format).
    src/core/ecm_mont_ckpt.cpp
    # s = torsion * lcm(1..B1): one product-tree builder shared by the GPU batch,
    # the CPU Edwards path and the CPU Montgomery path.
    src/core/ecm_stage1_exp.cpp
    # persistent, self-validating cache for that value: at B1 = 260e6 the product is 375 Mbit
    # and costs ~10 s to build, which every queue task would otherwise repeat.
    src/core/ecm_stage1_exp_cache.cpp
)

# 鍙湁杩欎袱涓?SIMD TU 鎷?AVX512 codegen; 璋冪敤鐐瑰繀椤诲湪 edwards_simd_available()
# (鍩虹嚎 TU 閲岀殑 CPUID 鎺㈡祴) 杩斿洖 1 涔嬪悗, 瑙?docs/ECM_EDWARDS_STAGE1.md 搂13.8銆?
if(MSVC)
    set_source_files_properties(src/cpu/simd_edwards.cpp src/cpu/simd_mont_ifma.cpp
        src/cpu/simd_mont_curve.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/cpu/simd_edwards.cpp src/cpu/simd_mont_ifma.cpp
        src/cpu/simd_mont_curve.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512ifma;-mavx512dq")
endif()

target_include_directories(ecm PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/opencl
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
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
#  ecm_backend_* seam (include/ecm_backend.h):
#     - OpenCL: src/opencl_backend_glue.cpp  (in `ecm`)
#     - CUDA:   src/cuda/ecm_cuda_backend.cu  (here)
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
            src/cuda/ecm_cuda_stage2.cu
            src/core/ecm_cuda_stage2_main.cpp
            src/core/ecm_expr.cpp
            src/core/ecm_worktodo.cpp
            src/core/ecm_queue_config.cpp)
        target_include_directories(ecm_cuda_stage2 PRIVATE ${GMP_INCLUDE_DIR})
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

    set(CGBN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cgbn/include")

    add_executable(ecm_cuda
        # Shared, backend-neutral host TUs (same files `ecm` uses)
        src/core/ecm_driver.cpp
        src/core/ecm_params_init.cpp
        src/core/ecm_checkpoint.cpp
        src/core/ecm_save.cpp
        src/core/ecm_gpu_common.cpp
        src/core/ecm_queue_config.cpp
        src/core/ecm_worktodo.cpp
        # Prime95 worktodo.add handoff for finished tasks (docs/DEV_ECM_GUI.md 18).
        src/core/p95_transfer.cpp
        src/core/p95_worktodo.cpp
        src/opencl_ecm_log.cpp
        src/opencl_ecm_runtime_config.cpp
        src/cpu/ecm_edwards_cpu.cpp
        src/cpu/ecm_edwards_save.cpp
        # ecm_driver.cpp (shared) calls the CPU Edwards SIMD batch unconditionally,
        # so this target needs the two AVX512 TUs as well without them the link
        # fails on ed_soa_* / ifma_to_mpz_lane.  The AVX512 COMPILE_OPTIONS set
        # above are per-source-file (directory scoped) and so apply here too.
        src/cpu/simd_edwards.cpp
        src/cpu/simd_mont_ifma.cpp
        # ecm_driver.cpp also dispatches the Suyama-sigma Montgomery stage-1 path
        # (--mont / ini mont=1), so both of its TUs are needed here as well.
        src/cpu/ecm_mont_cpu.cpp
        src/cpu/simd_mont_curve.cpp
        src/core/ecm_mont_ckpt.cpp
        src/core/ecm_stage1_exp.cpp
        src/core/ecm_stage1_exp_cache.cpp
        # CUDA backend + CGBN stage 1 (host TU + per-TPI kernel TUs)
        src/cuda/ecm_cuda_backend.cu
        kernels/cuda/cgbn_stage1.cu
        kernels/cuda/cgbn_stage1_prac_kernels.cu
        kernels/cuda/cgbn_stage1_prac_small.cu
        kernels/cuda/cgbn_stage1_prac_tpi16.cu
        kernels/cuda/cgbn_stage1_prac_tpi32.cu
        kernels/cuda/cgbn_stage1_prac_alternate32.cu
        src/core/ecm_prac_plan.cpp
        kernels/cuda/cgbn_stage1_kernels_tpi4.cu
        kernels/cuda/cgbn_stage1_kernels_tpi8.cu
        kernels/cuda/cgbn_stage1_kernels_tpi16.cu
        kernels/cuda/cgbn_stage1_kernels_tpi32.cu
        # Suyama param0 kernel variants (gpu_param = 0), kept in a separate TU so
        # the param3 builds do not pay for the extra instantiations.
        kernels/cuda/cgbn_stage1_kernels_suyama.cu
        # param2 ("batch 2", 6-torsion) variants (gpu_param = 2): same kernel body as the
        # Suyama family, instantiated with CONST_DIFF = true (no xdiff multiply).  Separate
        # TU on purpose -- a runtime flag cost 34% (docs/ECM_CGBN_OPTIMIZATION.md §5.6).
        kernels/cuda/cgbn_stage1_kernels_param2.cu
        # The TPI=16 (2560..8192) and TPI=32 (9216..16384) lookups of the two families above,
        # each in its own TU.  They dominate the kernel compile time: with everything in one
        # file the suyama TU alone took 745 s and param2 711 s, which made them the critical
        # path of the parallel build even with 6 concurrent nvcc jobs.  Split 2026-09-25 --
        # see docs/ECM_CGBN_OPTIMIZATION.md 8.8.
        kernels/cuda/cgbn_stage1_kernels_suyama_tpi16.cu
        kernels/cuda/cgbn_stage1_kernels_suyama_tpi32.cu
        kernels/cuda/cgbn_stage1_kernels_param2_tpi16.cu
        kernels/cuda/cgbn_stage1_kernels_param2_tpi32.cu
    )

    set_target_properties(ecm_cuda PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                              CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(ecm_cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core
        ${CMAKE_SOURCE_DIR}/src/cpu
        ${CMAKE_SOURCE_DIR}/kernels/cuda
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
set(ECM_CUDA_PTX_ARCH   ""    CACHE STRING "embed PTX for this arch only (release recipe: the oldest supported arch)")

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

    # ECM_CUDA_PTX_ARCH=<arch>: embed PTX for ONE architecture only (normally the oldest one
    # you support) while every other architecture gets SASS only.  This is the release
    # recipe: the PTX fallback lets the driver JIT onto GPUs newer than anything you built
    # SASS for, and it costs only ~24% of one SASS set (measured on our kernels).
    # Set it together with ECM_CUDA_EMBED_PTX=OFF, e.g.
    #   -DECM_CUDA_ARCHITECTURES=75;86;89;90;100;120 -DECM_CUDA_PTX_ARCH=75
    if(ECM_CUDA_PTX_ARCH)
        set(_ecm_cuda_arches "")
        foreach(_a IN LISTS ECM_CUDA_ARCHITECTURES)
            if(_a MATCHES "^${ECM_CUDA_PTX_ARCH}(-|$)")
                list(APPEND _ecm_cuda_arches "${_a}")          # SASS + PTX
            else()
                list(APPEND _ecm_cuda_arches "${_a}-real")     # SASS only
            endif()
        endforeach()
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

    # PROBE ONLY (docs/ECM_CGBN_OPTIMIZATION.md §5): make the fused double-and-add run its
    # addition half only every k-th bit, to measure the timing ceiling of an add-chain
    # (PRAC / NAF + dictionary) schedule.  k > 1 IS MATHEMATICALLY WRONG -- results will be
    # garbage; it exists to answer "how much would a chain with 1/k add density save?".
    # Default 1 = the correct kernel.
    set(ECM_PROBE_ADD_DENSITY "1" CACHE STRING
        "PROBE ONLY: run the fused add every k-th bit (k>1 = wrong results, for timing)")
    target_compile_definitions(ecm_cuda PRIVATE ECM_PROBE_ADD_DENSITY=${ECM_PROBE_ADD_DENSITY})

    # PROBE ONLY (docs/ECM_CGBN_OPTIMIZATION.md §5.4): replace the ladder step with a
    # windowed-chain op mix -- one xz doubling per bit plus one REAL differential addition
    # with a PROJECTIVE difference point (4M+2S) every M-th bit.  M=1 is the PRAC-like mix,
    # M=w+1 the ideal width-w NAF density.  RESULTS ARE WRONG (timing only).
    # Default 0 = probe off, correct kernel.
    set(ECM_PROBE_CHAIN_W "0" CACHE STRING
        "PROBE ONLY: chain op mix with one projective-difference add every M bits (wrong results)")
    target_compile_definitions(ecm_cuda PRIVATE ECM_PROBE_CHAIN_W=${ECM_PROBE_CHAIN_W})

    # Kernel launch/occupancy tunables (docs/ECM_CGBN_OPTIMIZATION.md §5.5/§8).  TPB must
    # match the launch config in kernels/cuda/cgbn_stage1.cu, hence compile-time.  Defaults
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
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_tpi4.cu
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_tpi8.cu
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
    # docs/ECM_CGBN_OPTIMIZATION.md §8.
    # DEFAULT 0: a per-FILE cap also hits that file's >=2560-bit instantiations, which the
    # measurement above shows should stay unconstrained (they want ~141 registers).  The
    # <=2048-bit tiers inside these files are handled by the tier-aware MIN_BLOCKS instead.
    set(ECM_MAXRREG_SUYAMA "0" CACHE STRING "maxrregcount for the param0/Suyama + param2 kernel sources (0 = off)")
    if(NOT ECM_MAXRREG_SUYAMA STREQUAL "0")
        set_source_files_properties(
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_suyama.cu
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_suyama_tpi16.cu
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_suyama_tpi32.cu
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_param2.cu
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_param2_tpi16.cu
            ${CMAKE_SOURCE_DIR}/kernels/cuda/cgbn_stage1_kernels_param2_tpi32.cu
            PROPERTIES COMPILE_OPTIONS "--maxrregcount=${ECM_MAXRREG_SUYAMA}")
    endif()

    # Tier filter for fast iteration (docs/ECM_CGBN_OPTIMIZATION.md §8).  On this machine
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
    # it can be set per bit width; see docs/ECM_CGBN_OPTIMIZATION.md 5.7.
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
    # results, timing only -- see docs/ECM_CGBN_OPTIMIZATION.md §5.6.
    set(ECM_PARAM2SHAPE "0" CACHE STRING "PROBE ONLY: run the param2-shaped step on the param0 path")
    target_compile_definitions(ecm_cuda PRIVATE ECM_PARAM2SHAPE=${ECM_PARAM2SHAPE})

    # Mersenne fold domain for the suyama family (param0 + param2).  For N = 2^k - 1 the
    # Montgomery reduction of every modular multiply is replaced by a fold (2^k == 1), which
    # removes CGBN's Q*N chains -- the GPU analogue of the CPU IFMA fold domain.  Measured
    # -21%..-31% per modular multiply on the 4060 Laptop at 4608/5120/8192 bits
    # (tools/bench/cgbn_mers_fold_probe.cu, docs/ECM_CGBN_OPTIMIZATION.md §9).
    #
    # This is a WHOLE-BUILD switch: a binary built with -DECM_MERS_FOLD=1 only carries fold
    # kernels, so it REFUSES any N that is not 2^k - 1 and refuses --gpu-param 3
    # (kernels/cuda/cgbn_stage1.cu).  Default 0 keeps the historical Montgomery kernels.
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

    # Register cap for the CGBN stage-1 kernel (docs/ECM_CGBN_OPTIMIZATION.md §8).  The
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

    # Copy GMP runtime DLL next to ecm_cuda.exe (same as `ecm`).
    if(MSVC AND EXISTS "${OPENSSL_ROOT_DIR}/bin/gmp-10.dll")
        add_custom_command(TARGET ecm_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENSSL_ROOT_DIR}/bin/gmp-10.dll"
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
    message(STATUS "tools: bench/test/disasm/p95feeder helper targets enabled (ECM_BUILD_TOOLS=ON)")
else()
    message(STATUS "tools: helper targets disabled (ECM_BUILD_TOOLS=OFF)")
endif()
# ===========================================================================
# ecm_gui -- the graphical front-end (docs/DEV_ECM_GUI.md).
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
    message(STATUS "gui: skipped (ecm_gui is Windows-only for now; see docs/DEV_ECM_GUI.md M7)")
elseif(NOT EXISTS "${ECM_IMGUI_DIR}/imgui.h")
    message(STATUS "gui: skipped -- ${ECM_IMGUI_DIR}/imgui.h not found. Vendor Dear ImGui "
                   "(docking branch) there; see third_party/imgui/README.md")
else()
    add_subdirectory(src/gui)
    message(STATUS "gui: ecm_gui enabled (ImGui from ${ECM_IMGUI_DIR})")
endif()
