# tools/bench/CMakeLists.txt -- benchmark / A-B / standalone-dump executables.
#
# These targets used to be defined in the root CMakeLists.txt; they live here now because
# they are tools, not part of the shipped `ecm`/`ecm_cuda` pair.  The root adds this
# directory with add_subdirectory(); -DECM_BUILD_TOOLS=OFF skips all of them.
#
# Two things stay exactly as they were in the root file:
#   * OUTPUT LOCATION: ecm_tool_output_root() keeps every tool executable in the build
#     root (build_vs18\Release\<tool>.exe, build_cuda_cmake\<tool>.exe), which the docs and
#     the tools/*.ps1 scripts reference by path.
#   * AVX512 codegen: the SIMD translation units under src/cpu/ are shared with `ecm`, and
#     their /arch:AVX512 flags were set as *source file properties* in the root directory.
#     Source-file properties are directory-scoped, so each target that compiles those TUs
#     re-applies them here via set_property(TARGET_DIRECTORY ...) (CMake >= 3.18).
#
# The measurements quoted in the comments are from docs/ECM_EDWARDS_STAGE1.md and
# docs/ECM_Montgomery_STAGE1.md.

# ---------------------------------------------------------------------------
# opencl_ecm_addsub: OpenCL add/sub micro-benchmark (main() lives behind
# BUILD_OPENCL_ECM_ADDSUB_MAIN=1 in src/opencl_ecm_addsub_bench.cpp).
# ---------------------------------------------------------------------------
add_executable(opencl_ecm_addsub
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_addsub_bench.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_addsub_manifest.cpp
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
ecm_tool_output_root(opencl_ecm_addsub)

# ---------------------------------------------------------------------------
# cpu_addsub_bench: CPU add/sub micro-benchmark (no OpenCL, GMP only).
#
# ISA strategy: the main TU + scalar/AVX2 kernels are compiled with an AVX2 baseline so
# the binary runs on any AVX2 CPU.  The AVX512 kernels live in a separate TU compiled
# with /arch:AVX512 and are dispatched only after a runtime CPUID check -- compiling the
# whole program with /arch:AVX512 would let the compiler emit AVX512 instructions in
# scalar code and crash AVX2-only CPUs.
# ---------------------------------------------------------------------------
add_executable(cpu_addsub_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_addsub_bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_addsub_avx512.cpp
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_runtime_config.cpp
)

target_include_directories(cpu_addsub_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cpu_addsub_bench PRIVATE BUILD_CPU_ADDSUB_MAIN=1)

ecm_target_link_gmp(cpu_addsub_bench)
ecm_tool_output_root(cpu_addsub_bench)

if(MSVC)
    target_compile_options(cpu_addsub_bench PRIVATE /arch:AVX2)
else()
    target_compile_options(cpu_addsub_bench PRIVATE -mavx2 -mfma)
endif()

if(MSVC)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_addsub_avx512.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_addsub_avx512.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq")
endif()

# ---------------------------------------------------------------------------
# simd_mont_gate: M1 gate bench for the AVX512-IFMA batched Montgomery kernel
# (docs/ECM_EDWARDS_STAGE1.md 13).  src/cpu/simd_mont_ifma.cpp is the ONLY file compiled
# with /arch:AVX512; the bench TU itself stays at the default baseline and the driver
# must gate every call on runtime CPUID detection.
# ---------------------------------------------------------------------------
add_executable(simd_mont_gate
    ${CMAKE_CURRENT_SOURCE_DIR}/simd_mont_gate.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/simd_mont_ifma.cpp
)

target_include_directories(simd_mont_gate PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(simd_mont_gate)
ecm_tool_output_root(simd_mont_gate)

# ecm_edwards_mont.h carries UTF-8 comments; without /utf-8 MSVC decodes it as CP936 and
# the preprocessor trips over the comment bytes.
if(MSVC)
    target_compile_options(simd_mont_gate PRIVATE /utf-8)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/cpu/simd_mont_ifma.cpp
        TARGET_DIRECTORY simd_mont_gate PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/cpu/simd_mont_ifma.cpp
        TARGET_DIRECTORY simd_mont_gate PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512ifma;-mavx512dq")
endif()

# ---------------------------------------------------------------------------
# ecm_edwards_standalone: one-curve stage-1 dump for cross-checking against Prime95 or an
# independent reference ladder.  Uses only the public API of src/cpu/ecm_edwards_cpu.h;
# it used to be a main() inside that production TU behind BUILD_ECM_EDWARDS_STANDALONE.
# ---------------------------------------------------------------------------
add_executable(ecm_edwards_standalone
    ${CMAKE_CURRENT_SOURCE_DIR}/ecm_edwards_standalone.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/ecm_edwards_cpu.cpp
)

target_include_directories(ecm_edwards_standalone PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(ecm_edwards_standalone)
ecm_tool_output_root(ecm_edwards_standalone)

if(MSVC)
    target_compile_options(ecm_edwards_standalone PRIVATE /utf-8)
endif()

# ---------------------------------------------------------------------------
# simd_edwards_bench: M2 verification + timing -- the batched Edwards stage-1
# (src/cpu/simd_edwards.cpp) against the production scalar ladder
# (src/cpu/ecm_edwards_cpu.cpp, without its standalone main).
# ---------------------------------------------------------------------------
add_executable(simd_edwards_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/simd_edwards_bench.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/simd_edwards.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/simd_mont_ifma.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/ecm_edwards_cpu.cpp
)

target_include_directories(simd_edwards_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/cpu
)

ecm_target_link_gmp(simd_edwards_bench)
ecm_tool_output_root(simd_edwards_bench)

if(MSVC)
    target_compile_options(simd_edwards_bench PRIVATE /utf-8)
endif()

# Only the two SIMD TUs get AVX512 codegen; the bench TU and the scalar reference stay at
# the baseline so the binary reports a clean error on a CPU without AVX512-IFMA instead of
# executing illegal instructions.
if(MSVC)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/cpu/simd_edwards.cpp ${CMAKE_SOURCE_DIR}/src/cpu/simd_mont_ifma.cpp
        TARGET_DIRECTORY simd_edwards_bench PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/cpu/simd_edwards.cpp ${CMAKE_SOURCE_DIR}/src/cpu/simd_mont_ifma.cpp
        TARGET_DIRECTORY simd_edwards_bench PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512ifma;-mavx512dq")
endif()

# ---------------------------------------------------------------------------
# opencl_ecm_montsqr: OpenCL Montgomery-square micro-benchmark
# (main() behind BUILD_OPENCL_ECM_MONTSQR_MAIN=1).
# ---------------------------------------------------------------------------
add_executable(opencl_ecm_montsqr
    ${CMAKE_SOURCE_DIR}/src/opencl_ecm_montsqr_bench.cpp
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
ecm_tool_output_root(opencl_ecm_montsqr)

if(MSVC)
    # NVIDIA's OpenCL JIT of mont_priv_opt.cl can overflow the default 1 MB thread stack.
    target_link_options(opencl_ecm_montsqr PRIVATE "/STACK:16777216")
endif()

# ---------------------------------------------------------------------------
# cpu_mont_bench: CPU Montgomery benchmark (AVX512/AVX2 batched CIOS)
# ---------------------------------------------------------------------------
add_executable(cpu_mont_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_mont_bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_mont_avx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_mont_scalar.cpp
)

target_include_directories(cpu_mont_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cpu_mont_bench PRIVATE BUILD_CPU_MONT_MAIN=1)

ecm_target_link_gmp(cpu_mont_bench)
ecm_tool_output_root(cpu_mont_bench)

if(MSVC)
    # AVX512 intrinsics require /arch:AVX512 on MSVC (avx512f, avx512dq are used).
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_mont_avx.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
endif()
