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

# ---------------------------------------------------------------------------
# ecm_p95feeder 鈥?standalone Prime95 handoff feeder.
#
# ecm.exe writes its stage-1 results locally only (<tmp_dir>/e{n:07d}_c{k}.tmp);
# the feeder watches that directory and delivers each finished curve into a
# running Prime95 instance (save file e{n:07d} + an ECM= line under a free
# [Worker #N] section of worktodo.add). See docs/ECM_EDWARDS_STAGE1.md 搂11.
# GMP-only: no OpenCL/OpenSSL dependency.
# ---------------------------------------------------------------------------
add_executable(ecm_p95feeder
    src/core/ecm_p95feeder.cpp
    src/core/p95_worktodo.cpp
    src/core/ecm_worktodo.cpp
    src/cpu/ecm_edwards_save.cpp
)

target_include_directories(ecm_p95feeder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(ecm_p95feeder)

if(MSVC)
    target_compile_options(ecm_p95feeder PRIVATE /utf-8)
endif()

foreach(_gmp_dll_cand "${ECM_WINDOWS_GMP_ROOT}/bin/gmp-10.dll"
                      "${OPENSSL_ROOT_DIR}/bin/gmp-10.dll")
    if(MSVC AND EXISTS "${_gmp_dll_cand}")
        add_custom_command(TARGET ecm_p95feeder POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_gmp_dll_cand}"
                "$<TARGET_FILE_DIR:ecm_p95feeder>"
            COMMENT "Copying GMP DLL next to ecm_p95feeder")
        break()
    endif()
endforeach()

# indicators.hpp contains Unicode block characters (鈻屸枑鈻堚杽鈥? for progress
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

add_executable(main
    src/cl_probe.cpp
    src/main_test.cpp
    src/opencl_low_risk_tests.cpp
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

add_executable(opencl_ecm_addsub
    src/opencl_ecm_addsub_bench.cpp
    src/opencl_ecm_addsub_manifest.cpp
)

target_include_directories(opencl_ecm_addsub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(opencl_ecm_addsub PRIVATE BUILD_OPENCL_ECM_ADDSUB_MAIN=1)

target_link_libraries(opencl_ecm_addsub
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
)

ecm_target_link_gmp(opencl_ecm_addsub)

target_link_libraries(opencl_ecm_addsub
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
)

# CPU addsub micro-benchmark (no OpenCL dependency, GMP only)
#
# ISA strategy: the main TU + scalar/AVX2 kernels are compiled with an AVX2
# baseline so the binary runs on any AVX2 CPU.  The AVX512 kernels live in a
# separate TU compiled with /arch:AVX512 and are dispatched only after runtime
# CPUID detection.  Compiling the whole program with /arch:AVX512 would let the
# compiler emit AVX512 instructions in scalar code, crashing AVX2-only CPUs.
add_executable(cpu_addsub_bench
    tools/bench/cpu_addsub_bench.cpp
    tools/bench/cpu_addsub_avx512.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_runtime_config.cpp
)

target_include_directories(cpu_addsub_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/tools/bench
)

target_compile_definitions(cpu_addsub_bench PRIVATE BUILD_CPU_ADDSUB_MAIN=1)

ecm_target_link_gmp(cpu_addsub_bench)

# Baseline for the whole target: AVX2 (safe on all target CPUs).
if(MSVC)
    target_compile_options(cpu_addsub_bench PRIVATE /arch:AVX2)
else()
    target_compile_options(cpu_addsub_bench PRIVATE -mavx2 -mfma)
endif()

# AVX512 kernels: isolated TU, only this file gets AVX512 codegen.
if(MSVC)
    set_source_files_properties(tools/bench/cpu_addsub_avx512.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(tools/bench/cpu_addsub_avx512.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq")
endif()

# ---------------------------------------------------------------------------
# simd_mont_gate 鈥?M1 gate bench for the AVX512-IFMA batched Montgomery kernel
# (docs/ECM_EDWARDS_STAGE1.md 搂13).  src/cpu/simd_mont_ifma.cpp is the ONLY file
# compiled with /arch:AVX512; the bench TU itself stays at the default baseline
# and the eventual driver must gate every call on runtime CPUID detection.
# ---------------------------------------------------------------------------
add_executable(simd_mont_gate
    tools/bench/simd_mont_gate.cpp
    src/cpu/simd_mont_ifma.cpp
)

target_include_directories(simd_mont_gate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(simd_mont_gate)

# ecm_edwards_mont.h carries Chinese comments in UTF-8; without /utf-8 MSVC
# decodes it as CP936 and the preprocessor trips over the comment bytes.
if(MSVC)
    target_compile_options(simd_mont_gate PRIVATE /utf-8)
endif()

if(MSVC)
    set_source_files_properties(src/cpu/simd_mont_ifma.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/cpu/simd_mont_ifma.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512ifma;-mavx512dq")
endif()

# ---------------------------------------------------------------------------
# ecm_edwards_standalone 鈥?one-curve stage-1 dump for cross-checking against
# Prime95 or an independent reference ladder.  Uses only the public API of
# src/cpu/ecm_edwards_cpu.h; it used to be a main() inside that production TU
# behind BUILD_ECM_EDWARDS_STANDALONE.
# ---------------------------------------------------------------------------
add_executable(ecm_edwards_standalone
    tools/bench/ecm_edwards_standalone.cpp
    src/cpu/ecm_edwards_cpu.cpp
)

target_include_directories(ecm_edwards_standalone PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(ecm_edwards_standalone)

if(MSVC)
    target_compile_options(ecm_edwards_standalone PRIVATE /utf-8)
endif()

# ---------------------------------------------------------------------------
# simd_edwards_bench 鈥?M2 verification + timing: batched Edwards stage-1
# (src/cpu/simd_edwards.cpp) against the production scalar ladder
# (src/cpu/ecm_edwards_cpu.cpp, without its standalone main).
# ---------------------------------------------------------------------------
add_executable(simd_edwards_bench
    tools/bench/simd_edwards_bench.cpp
    src/cpu/simd_edwards.cpp
    src/cpu/simd_mont_ifma.cpp
    src/cpu/ecm_edwards_cpu.cpp
)

target_include_directories(simd_edwards_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(simd_edwards_bench)

if(MSVC)
    target_compile_options(simd_edwards_bench PRIVATE /utf-8)
endif()

# Only the two SIMD TUs get AVX512 codegen; the bench TU and the scalar
# reference stay at the baseline so the binary reports a clean error on a
# CPU without AVX512-IFMA instead of executing illegal instructions.
if(MSVC)
    set_source_files_properties(src/cpu/simd_edwards.cpp src/cpu/simd_mont_ifma.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/cpu/simd_edwards.cpp src/cpu/simd_mont_ifma.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512ifma;-mavx512dq")
endif()

add_executable(opencl_ecm_montsqr
    src/opencl_ecm_montsqr_bench.cpp
)

target_include_directories(opencl_ecm_montsqr PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(opencl_ecm_montsqr PRIVATE BUILD_OPENCL_ECM_MONTSQR_MAIN=1)

target_link_libraries(opencl_ecm_montsqr
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
)

ecm_target_link_gmp(opencl_ecm_montsqr)

if(MSVC)
    # NVIDIA OpenCL JIT of mont_priv_opt.cl can overflow the default 1MB thread stack.
    target_link_options(opencl_ecm_montsqr PRIVATE "/STACK:16777216")
endif()

# 鈹€鈹€ Sliced CIOS standalone verification target 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
add_executable(sliced_cios_test
    src/sliced_cios_test.cpp
)
target_include_directories(sliced_cios_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sliced_cios_test
    OpenCL::OpenCL
)
ecm_target_link_gmp(sliced_cios_test)

add_executable(sliced_cios_8192_test
    src/sliced_cios_8192_test.cpp
)
target_include_directories(sliced_cios_8192_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sliced_cios_8192_test OpenCL::OpenCL)
ecm_target_link_gmp(sliced_cios_8192_test)

add_executable(opencl_asm_selftest
    src/cl_probe.cpp
    src/opencl_asm_selftest.cpp
)

target_include_directories(opencl_asm_selftest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_asm_selftest
    OpenCL::OpenCL
    cgbn_opencl
)

add_executable(opencl_mont_isa_export
    src/cl_probe.cpp
    src/opencl_mont_isa_export.cpp
)

target_include_directories(opencl_mont_isa_export PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_mont_isa_export
    OpenCL::OpenCL
    cgbn_opencl
)

add_executable(opencl_addsub_isa_export
    src/cl_probe.cpp
    src/opencl_addsub_isa_export.cpp
    src/opencl_ecm_addsub_manifest.cpp
)

target_include_directories(opencl_addsub_isa_export PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_addsub_isa_export
    OpenCL::OpenCL
    cgbn_opencl
)

# 鈹€鈹€ CPU Montgomery Benchmark (AVX512/AVX2 batched CIOS) 鈹€鈹€
add_executable(cpu_mont_bench
    tools/bench/cpu_mont_bench.cpp
    tools/bench/cpu_mont_avx.cpp
    tools/bench/cpu_mont_scalar.cpp
)

target_include_directories(cpu_mont_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/tools/bench
)

target_compile_definitions(cpu_mont_bench PRIVATE BUILD_CPU_MONT_MAIN=1)

ecm_target_link_gmp(cpu_mont_bench)

if(MSVC)
    # AVX512 intrinsics require /arch:AVX512 on MSVC (avx512f, avx512dq used)
    set_source_files_properties(tools/bench/cpu_mont_avx.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
endif()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  ecm_cuda 鈥?native CUDA/CGBN backend (mirror of the OpenCL `ecm` executable)
#
#  Reuses the exact same driver / argument parsing / checkpoint / save / logging
#  translation units as `ecm`, swapping only the GPU backend at link time via the
#  ecm_backend_* seam (include/ecm_backend.h):
#     - OpenCL: src/opencl_backend_glue.cpp  (in `ecm`)
#     - CUDA:   src/cuda/ecm_cuda_backend.cu  (here)
#
#  Auto-enabled when a CUDA toolkit is present; force off with -DECM_ENABLE_CUDA=OFF.
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
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
        src/opencl_ecm_log.cpp
        src/opencl_ecm_runtime_config.cpp
        src/cpu/ecm_edwards_cpu.cpp
        src/cpu/ecm_edwards_save.cpp
        # ecm_driver.cpp (shared) calls the CPU Edwards SIMD batch unconditionally,
        # so this target needs the two AVX512 TUs as well 鈥?without them the link
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

    set_target_properties(ecm_cuda PROPERTIES
        CUDA_ARCHITECTURES "${ECM_CUDA_ARCHITECTURES}"
    )

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
