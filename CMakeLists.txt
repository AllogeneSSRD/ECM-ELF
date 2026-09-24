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
# docs/ECM_EDWARDS_STAGE1.md §10.2). Build it with third_party/build_gmp_zen3.sh
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
)

# 只有这两个 SIMD TU 拿 AVX512 codegen; 调用点必须在 edwards_simd_available()
# (基线 TU 里的 CPUID 探测) 返回 1 之后, 见 docs/ECM_EDWARDS_STAGE1.md §13.8。
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
# ecm_p95feeder — standalone Prime95 handoff feeder.
#
# ecm.exe writes its stage-1 results locally only (<tmp_dir>/e{n:07d}_c{k}.tmp);
# the feeder watches that directory and delivers each finished curve into a
# running Prime95 instance (save file e{n:07d} + an ECM= line under a free
# [Worker #N] section of worktodo.add). See docs/ECM_EDWARDS_STAGE1.md §11.
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

# indicators.hpp contains Unicode block characters (▌▐█▄…) for progress
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
# simd_mont_gate — M1 gate bench for the AVX512-IFMA batched Montgomery kernel
# (docs/ECM_EDWARDS_STAGE1.md §13).  src/cpu/simd_mont_ifma.cpp is the ONLY file
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
# ecm_edwards_standalone — one-curve stage-1 dump for cross-checking against
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
# simd_edwards_bench — M2 verification + timing: batched Edwards stage-1
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

# ── Sliced CIOS standalone verification target ────────────────────
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

# ── CPU Montgomery Benchmark (AVX512/AVX2 batched CIOS) ──
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

# ══════════════════════════════════════════════════════════════════════════
#  ecm_cuda — native CUDA/CGBN backend (mirror of the OpenCL `ecm` executable)
#
#  Reuses the exact same driver / argument parsing / checkpoint / save / logging
#  translation units as `ecm`, swapping only the GPU backend at link time via the
#  ecm_backend_* seam (include/ecm_backend.h):
#     - OpenCL: src/opencl_backend_glue.cpp  (in `ecm`)
#     - CUDA:   src/cuda/ecm_cuda_backend.cu  (here)
#
#  Auto-enabled when a CUDA toolkit is present; force off with -DECM_ENABLE_CUDA=OFF.
# ══════════════════════════════════════════════════════════════════════════
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
        # so this target needs the two AVX512 TUs as well — without them the link
        # fails on ed_soa_* / ifma_to_mpz_lane.  The AVX512 COMPILE_OPTIONS set
        # above are per-source-file (directory scoped) and so apply here too.
        src/cpu/simd_edwards.cpp
        src/cpu/simd_mont_ifma.cpp
        # ecm_driver.cpp also dispatches the Suyama-sigma Montgomery stage-1 path
        # (--mont / ini mont=1), so both of its TUs are needed here as well.
        src/cpu/ecm_mont_cpu.cpp
        src/cpu/simd_mont_curve.cpp
        # CUDA backend + CGBN stage 1 (host TU + per-TPI kernel TUs)
        src/cuda/ecm_cuda_backend.cu
        kernels/cuda/cgbn_stage1.cu
        kernels/cuda/cgbn_stage1_kernels_tpi4.cu
        kernels/cuda/cgbn_stage1_kernels_tpi8.cu
        kernels/cuda/cgbn_stage1_kernels_tpi16.cu
        kernels/cuda/cgbn_stage1_kernels_tpi32.cu
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
    option(ECM_CUDA_FULL_BUILD "Compile the full CGBN kernel set instead of the dev build (N<=1024)" OFF)
    if(ECM_CUDA_FULL_BUILD)
        target_compile_definitions(ecm_cuda PRIVATE ECM_CUDA_FULL_BUILD)
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
