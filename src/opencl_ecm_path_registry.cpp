#include "opencl_ecm_path_registry.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMontNoMinLimbs = 0;
constexpr uint32_t kMontNoMaxLimbs = 0;

constexpr uint32_t kAddSubNoMinLimbs = 0;
constexpr uint32_t kAddSubNoMaxLimbs = 0;


#define ECM_MONT_ALIAS_TABLE(side, S)                                                             \
    static const char *const kMontAliases_##side##_unroll_192b[] = {                              \
        "unroll_192b", "unroll_192", "mont_" S "_priv_unroll_only_192", nullptr};            \
    static const char *const kMontAliases_##side##_unroll_256b[] = {                              \
        "unroll_256b", "unroll_256", "mont_" S "_priv_unroll_only_256", nullptr};            \
    static const char *const kMontAliases_##side##_unroll_384b[] = {                              \
        "unroll_384b", "unroll_384", "mont_" S "_priv_unroll_only_384", nullptr};            \
    static const char *const kMontAliases_##side##_unroll_512b[] = {                              \
        "unroll_512b", "unroll_512", "mont_" S "_priv_unroll_only_512", nullptr};            \
    static const char *const kMontAliases_##side##_unroll_768b[] = {                              \
        "unroll_768b", "unroll_768", "mont_" S "_priv_unroll_only_768b", nullptr};          \
    static const char *const kMontAliases_##side##_unroll_1024b[] = {                             \
        "unroll_1024b", "unroll_1024", "mont_" S "_priv_unroll_only_1024b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_1536b[] = {                             \
        "unroll_1536b", "unroll_1536", "mont_" S "_priv_unroll_only_1536b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_2048b[] = {                             \
        "unroll_2048b", "unroll_2048", "mont_" S "_priv_unroll_only_2048b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_2560b[] = {                             \
        "unroll_2560b", "unroll_2560", "mont_" S "_priv_unroll_only_2560b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_3072b[] = {                             \
        "unroll_3072b", "unroll_3072", "mont_" S "_priv_unroll_only_3072b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_3584b[] = {                             \
        "unroll_3584b", "unroll_3584", "mont_" S "_priv_unroll_only_3584b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_4096b[] = {                             \
        "unroll_4096b", "unroll_4096", "mont_" S "_priv_unroll_only_4096b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_4608b[] = {                             \
        "unroll_4608b", "unroll_4608", "mont_" S "_priv_unroll_only_4608b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_5120b[] = {                             \
        "unroll_5120b", "unroll_5120", "mont_" S "_priv_unroll_only_5120b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_5632b[] = {                             \
        "unroll_5632b", "unroll_5632", "mont_" S "_priv_unroll_only_5632b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_6144b[] = {                             \
        "unroll_6144b", "unroll_6144", "mont_" S "_priv_unroll_only_6144b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_6656b[] = {                             \
        "unroll_6656b", "unroll_6656", "mont_" S "_priv_unroll_only_6656b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_7168b[] = {                             \
        "unroll_7168b", "unroll_7168", "mont_" S "_priv_unroll_only_7168b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_7680b[] = {                             \
        "unroll_7680b", "unroll_7680", "mont_" S "_priv_unroll_only_7680b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll_8192b[] = {                             \
        "unroll_8192b", "unroll_8192", "mont_" S "_priv_unroll_only_8192b", nullptr};       \
    static const char *const kMontAliases_##side##_unroll64_4096[] = {"unroll64_4096", nullptr};  \
    static const char *const kMontAliases_##side##_fips4096[] = {"fips4096", nullptr};            \
    static const char *const kMontAliases_##side##_karatsuba_2048b[] = {"karatsuba_2048b", nullptr}; \
    static const char *const kMontAliases_##side##_karatsuba_2048b_mt4[] = {"karatsuba_2048b_mt4", nullptr}; \
    static const char *const kMontAliases_##side##_fips4096_mt8[] = {"fips4096_mt8", nullptr};    \
    static const char *const kMontAliases_##side##_fips4096_mt16[] = {"fips4096_mt16", nullptr};  \
    static const char *const kMontAliases_##side##_unroll32[] = {                                 \
        "unroll32", "mont_" S "_priv_unroll32", "mont_" S "_stage1_unroll32", nullptr};           \
    static const char *const kMontAliases_##side##_priv_opt[] = {                                 \
        "priv_opt", "mont_" S "_priv_opt", "mont_" S "_stage1_priv_opt", nullptr}; \
\
    static const char *const kMontAliases_##side##_unroll_manual_192b[] = {                       \
        "unroll_manual_192b", "unroll_manual_192", "mont_" S "_priv_unroll_manual_192b", nullptr};      \
    static const char *const kMontAliases_##side##_unroll_manual_256b[] = {                       \
        "unroll_manual_256b", "unroll_manual_256", "mont_" S "_priv_unroll_manual_256b", nullptr};      \
    static const char *const kMontAliases_##side##_unroll_manual_384b[] = {                       \
        "unroll_manual_384b", "unroll_manual_384", "mont_" S "_priv_unroll_manual_384b", nullptr};      \
    static const char *const kMontAliases_##side##_unroll_manual_512b[] = {                       \
        "unroll_manual_512b", "unroll_manual_512", "mont_" S "_priv_unroll_manual_512b", nullptr};      \
    static const char *const kMontAliases_##side##_unroll_manual_768b[] = {                       \
        "unroll_manual_768b", "unroll_manual_768", "mont_" S "_priv_unroll_manual_768b", nullptr};      \
    static const char *const kMontAliases_##side##_unroll_manual_1024b[] = {                      \
        "unroll_manual_1024b", "unroll_manual_1024", "mont_" S "_priv_unroll_manual_1024b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_1536b[] = {                      \
    //     "unroll_manual_1536b", "unroll_manual_1536", "mont_" S "_priv_unroll_manual_1536b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_2048b[] = {                      \
    //     "unroll_manual_2048b", "unroll_manual_2048", "mont_" S "_priv_unroll_manual_2048b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_2560b[] = {                      \
    //     "unroll_manual_2560b", "unroll_manual_2560", "mont_" S "_priv_unroll_manual_2560b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_3072b[] = {                      \
    //     "unroll_manual_3072b", "unroll_manual_3072", "mont_" S "_priv_unroll_manual_3072b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_3584b[] = {                      \
    //     "unroll_manual_3584b", "unroll_manual_3584", "mont_" S "_priv_unroll_manual_3584b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_4096b[] = {                      \
    //     "unroll_manual_4096b", "unroll_manual_4096", "mont_" S "_priv_unroll_manual_4096b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_4608b[] = {                      \
    //     "unroll_manual_4608b", "unroll_manual_4608", "mont_" S "_priv_unroll_manual_4608b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_5120b[] = {                      \
    //     "unroll_manual_5120b", "unroll_manual_5120", "mont_" S "_priv_unroll_manual_5120b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_5632b[] = {                      \
    //     "unroll_manual_5632b", "unroll_manual_5632", "mont_" S "_priv_unroll_manual_5632b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_6144b[] = {                      \
    //     "unroll_manual_6144b", "unroll_manual_6144", "mont_" S "_priv_unroll_manual_6144b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_6656b[] = {                      \
    //     "unroll_manual_6656b", "unroll_manual_6656", "mont_" S "_priv_unroll_manual_6656b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_7168b[] = {                      \
    //     "unroll_manual_7168b", "unroll_manual_7168", "mont_" S "_priv_unroll_manual_7168b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_7680b[] = {                      \
    //     "unroll_manual_7680b", "unroll_manual_7680", "mont_" S "_priv_unroll_manual_7680b", nullptr};   \
    // static const char *const kMontAliases_##side##_unroll_manual_8192b[] = {                      \
    //     "unroll_manual_8192b", "unroll_manual_8192", "mont_" S "_priv_unroll_manual_8192b", nullptr};

ECM_MONT_ALIAS_TABLE(mul, "mul")
ECM_MONT_ALIAS_TABLE(sqr, "sqr")

// Each unrolled width has TWO variants with distinct ids:
//   auto   (#pragma unroll loop)        -> excluded on Android (gpu_vendor_exclude=OS_ANDROID)
//   manual (constant-index straight)    -> Android only (os_mask=OS_ANDROID)
// Both share the "unroll_*b" alias so that explicit --mul unroll_768b resolves to
// whichever variant fits the platform (auto on desktop, manual on Android).
// Both appear in Android dropdowns for manual testing; auto only on desktop.
#define ECM_MONT_OPERATORS(X)                                                           \
    X(unroll_192b,   unroll_192b,    6,   0,   6,   6, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_256b,   unroll_256b,    8,   0,   8,   8, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_384b,   unroll_384b,   10,   0,  12,  12, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_512b,   unroll_512b,   20,   0,  16,  16, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_768b,   unroll_768b,   22,   0,  24,  24, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_1024b,  unroll_1024b,  24,   0,  32,  32, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_1536b,  unroll_1536b,  25,   0,  48,  48, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_2048b,  unroll_2048b,  26,   0,  64,  64, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_2560b,  unroll_2560b,  27,   0,  80,  80, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_3072b,  unroll_3072b,  28,   0,  96,  96, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_3584b,  unroll_3584b,  29,   0, 112, 112, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_4096b,  unroll_4096b,  30,   0, 128, 128, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_4608b,  unroll_4608b,  31,   0, 144, 144, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_5120b,  unroll_5120b,  32,   0, 160, 160, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_5632b,  unroll_5632b,  33,   0, 176, 176, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_6144b,  unroll_6144b,  34,   0, 192, 192, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_6656b,  unroll_6656b,  35,   0, 208, 208, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_7168b,  unroll_7168b,  36,   0, 224, 224, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_7680b,  unroll_7680b,  37,   0, 240, 240, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_8192b,  unroll_8192b,  38,   0, 256, 256, OS_ANY, GPU_ANY, true, 1, 0)  \
\
    X(unroll64_4096, unroll_4096b,  23,  96, 128, 128, OS_ANY, GPU_ANY, true, 1, 0)     \
    X(fips4096, fips_4096b,         27,  96, 128, 128, OS_ANY, GPU_ANY, true, 1, 0)     \
    X(karatsuba_2048b, karatsuba_2048b, -1,  48,  64,  64, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(karatsuba_2048b_mt4, karatsuba_2048b, -1,  48,  64,  64, OS_ANY, GPU_ANY, true, 4, 320)  \
    X(fips4096_mt8, fips_4096b,     29,  96, 128, 128, OS_ANY, GPU_ANY, true, 8, 897)   \
    X(fips4096_mt16, fips_4096b,    31,  96, 128, 128, OS_ANY, GPU_ANY, true, 16, 897)  \
\
    X(unroll32, unroll_32,   -1,   0,   0,   0, OS_ANY, GPU_ANY, false, 1, 0) \
    X(priv_opt, priv_opt,   127,   0,   0,   0, OS_ANY, GPU_ANY, false, 1, 0) \
\
    X(unroll_manual_192b,   unroll_manual_192b,  -1,   0,   6,   6, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_manual_256b,   unroll_manual_256b,  -1,   0,   8,   8, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_manual_384b,   unroll_manual_384b,  -1,   0,  12,  12, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_manual_512b,   unroll_manual_512b,  -1,   0,  16,  16, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_manual_768b,   unroll_manual_768b,  -1,   0,  24,  24, OS_ANY, GPU_ANY, true, 1, 0)  \
    X(unroll_manual_1024b,  unroll_manual_1024b, -1,   0,  32,  32, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_1536b,  unroll_manual_1536b, -1,   0,  48,  48, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_2048b,  unroll_manual_2048b, -1,   0,  64,  64, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_2560b,  unroll_manual_2560b, -1,   0,  80,  80, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_3072b,  unroll_manual_3072b, -1,   0,  96,  96, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_3584b,  unroll_manual_3584b, -1,   0, 112, 112, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_4096b,  unroll_manual_4096b, -1,   0, 128, 128, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_4608b,  unroll_manual_4608b, -1,   0, 144, 144, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_5120b,  unroll_manual_5120b, -1,   0, 160, 160, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_5632b,  unroll_manual_5632b, -1,   0, 176, 176, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_6144b,  unroll_manual_6144b, -1,   0, 192, 192, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_6656b,  unroll_manual_6656b, -1,   0, 208, 208, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_7168b,  unroll_manual_7168b, -1,   0, 224, 224, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_7680b,  unroll_manual_7680b, -1,   0, 240, 240, OS_ANY, GPU_ANY, true, 1, 0)  \
    // X(unroll_manual_8192b,  unroll_manual_8192b, -1,   0, 256, 256, OS_ANY, GPU_ANY, true, 1, 0)

#define ECM_MONT_MUL_ROW(idt, path, ...)                                                       \
    {#idt, "mont_mul_" #path, kMontAliases_mul_##idt, "mont_mul/mont_mul_" #path ".cl", __VA_ARGS__},
#define ECM_MONT_SQR_ROW(idt, path, ...)                                                       \
    {#idt, "mont_sqr_" #path, kMontAliases_sqr_##idt, "mont_mul/mont_mul_" #path ".cl", __VA_ARGS__},

constexpr EcmMontPathDescriptor kMontMulRegistry[] = {ECM_MONT_OPERATORS(ECM_MONT_MUL_ROW)};
constexpr EcmMontPathDescriptor kMontSqrRegistry[] = {ECM_MONT_OPERATORS(ECM_MONT_SQR_ROW)};


// ── ADDSUB alias table (mirrors ECM_MONT_ALIAS_TABLE pattern) ────────────
// Each width gets two variants: asm (AMD-specific, assembly-optimised) and
// unroll (generic, portable #pragma unroll).
// side ∈ {Add, Sub}  →  generates kAddAliases_* or kSubAliases_*
// S    ∈ {"add_mod", "sub_mod"}  →  full CL function prefix
#define ECM_ADDSUB_ALIAS_TABLE(side, S)                                                           \
    static const char *const k##side##Aliases_asm_128b[] = {"asm_128b", S "_asm_128b", nullptr};  \
    static const char *const k##side##Aliases_asm_192b[] = {"asm_192b", S "_asm_192b", nullptr};  \
    static const char *const k##side##Aliases_asm_256b[] = {"asm_256b", S "_asm_256b", nullptr};  \
    static const char *const k##side##Aliases_asm_384b[] = {"asm_384b", S "_asm_384b", nullptr};  \
    static const char *const k##side##Aliases_asm_512b[] = {"asm_512b", S "_asm_512b", nullptr};  \
    static const char *const k##side##Aliases_asm_768b[] = {"asm_768b", S "_asm_768b", nullptr};  \
    static const char *const k##side##Aliases_asm_1024b[] = {"asm_1024b", S "_asm_1024b", nullptr}; \
    static const char *const k##side##Aliases_asm_1536b[] = {"asm_1536b", S "_asm_1536b", nullptr}; \
    static const char *const k##side##Aliases_asm_2048b[] = {"asm_2048b", S "_asm_2048b", nullptr}; \
    static const char *const k##side##Aliases_asm_2560b[] = {"asm_2560b", S "_asm_2560b", nullptr}; \
    static const char *const k##side##Aliases_asm_3072b[] = {"asm_3072b", S "_asm_3072b", nullptr}; \
    static const char *const k##side##Aliases_asm_3584b[] = {"asm_3584b", S "_asm_3584b", nullptr}; \
    static const char *const k##side##Aliases_asm_4096b[] = {"asm_4096b", S "_asm_4096b", nullptr}; \
    static const char *const k##side##Aliases_asm_4608b[] = {"asm_4608b", S "_asm_4608b", nullptr}; \
    static const char *const k##side##Aliases_asm_5120b[] = {"asm_5120b", S "_asm_5120b", nullptr}; \
    static const char *const k##side##Aliases_asm_5632b[] = {"asm_5632b", S "_asm_5632b", nullptr}; \
    static const char *const k##side##Aliases_asm_6144b[] = {"asm_6144b", S "_asm_6144b", nullptr}; \
    static const char *const k##side##Aliases_asm_6656b[] = {"asm_6656b", S "_asm_6656b", nullptr}; \
    static const char *const k##side##Aliases_asm_7168b[] = {"asm_7168b", S "_asm_7168b", nullptr}; \
    static const char *const k##side##Aliases_asm_7680b[] = {"asm_7680b", S "_asm_7680b", nullptr}; \
    static const char *const k##side##Aliases_asm_8192b[] = {"asm_8192b", S "_asm_8192b", nullptr}; \
    static const char *const k##side##Aliases_unroll_128b[] = {"unroll_128b", S "_unroll_128b", nullptr}; \
    static const char *const k##side##Aliases_unroll_192b[] = {"unroll_192b", S "_unroll_192b", nullptr}; \
    static const char *const k##side##Aliases_unroll_256b[] = {"unroll_256b", S "_unroll_256b", nullptr}; \
    static const char *const k##side##Aliases_unroll_384b[] = {"unroll_384b", S "_unroll_384b", nullptr}; \
    static const char *const k##side##Aliases_unroll_512b[] = {"unroll_512b", S "_unroll_512b", nullptr}; \
    static const char *const k##side##Aliases_unroll_768b[] = {"unroll_768b", S "_unroll_768b", nullptr}; \
    static const char *const k##side##Aliases_unroll_1024b[] = {"unroll_1024b", S "_unroll_1024b", nullptr}; \
    static const char *const k##side##Aliases_unroll_1536b[] = {"unroll_1536b", S "_unroll_1536b", nullptr}; \
    static const char *const k##side##Aliases_unroll_2048b[] = {"unroll_2048b", S "_unroll_2048b", nullptr}; \
    static const char *const k##side##Aliases_unroll_2560b[] = {"unroll_2560b", S "_unroll_2560b", nullptr}; \
    static const char *const k##side##Aliases_unroll_3072b[] = {"unroll_3072b", S "_unroll_3072b", nullptr}; \
    static const char *const k##side##Aliases_unroll_3584b[] = {"unroll_3584b", S "_unroll_3584b", nullptr}; \
    static const char *const k##side##Aliases_unroll_4096b[] = {"unroll_4096b", S "_unroll_4096b", nullptr}; \
    static const char *const k##side##Aliases_unroll_4608b[] = {"unroll_4608b", S "_unroll_4608b", nullptr}; \
    static const char *const k##side##Aliases_unroll_5120b[] = {"unroll_5120b", S "_unroll_5120b", nullptr}; \
    static const char *const k##side##Aliases_unroll_5632b[] = {"unroll_5632b", S "_unroll_5632b", nullptr}; \
    static const char *const k##side##Aliases_unroll_6144b[] = {"unroll_6144b", S "_unroll_6144b", nullptr}; \
    static const char *const k##side##Aliases_unroll_6656b[] = {"unroll_6656b", S "_unroll_6656b", nullptr}; \
    static const char *const k##side##Aliases_unroll_7168b[] = {"unroll_7168b", S "_unroll_7168b", nullptr}; \
    static const char *const k##side##Aliases_unroll_7680b[] = {"unroll_7680b", S "_unroll_7680b", nullptr}; \
    static const char *const k##side##Aliases_unroll_8192b[] = {"unroll_8192b", S "_unroll_8192b", nullptr}; \
    static const char *const k##side##Aliases_fused[] = {"fused", S "_fused", nullptr};           \
    static const char *const k##side##Aliases_fused_unroll[] = {"fused_unroll", S "_fused_unroll", nullptr};

ECM_ADDSUB_ALIAS_TABLE(Add, "add_mod")
ECM_ADDSUB_ALIAS_TABLE(Sub, "sub_mod")

#define ECM_ADDSUB_OPERATORS(X)                                     \
    X(asm_128b,         4,   0,   4,   4, OS_DESKTOP, GPU_AMD) \
    X(unroll_128b,      5,   0,   4,   4, OS_ANY, GPU_ANY) \
    X(asm_192b,         6,   0,   6,   6, OS_DESKTOP, GPU_AMD) \
    X(unroll_192b,      7,   0,   6,   6, OS_ANY, GPU_ANY) \
    X(asm_256b,         8,   0,   8,   8, OS_DESKTOP, GPU_AMD) \
    X(unroll_256b,      9,   0,   8,   8, OS_ANY, GPU_ANY) \
    X(asm_384b,        12,   0,  12,  12, OS_DESKTOP, GPU_AMD) \
    X(unroll_384b,     13,   0,  12,  12, OS_ANY, GPU_ANY) \
    X(asm_512b,        16,   0,  16,  16, OS_DESKTOP, GPU_AMD) \
    X(unroll_512b,     17,   0,  16,  16, OS_ANY, GPU_ANY) \
    X(asm_768b,        18,   0,  24,  24, OS_DESKTOP, GPU_AMD) \
    X(unroll_768b,     19,   0,  24,  24, OS_ANY, GPU_ANY) \
    X(asm_1024b,       20,   0,  32,  32, OS_DESKTOP, GPU_AMD) \
    X(unroll_1024b,    21,   0,  32,  32, OS_ANY, GPU_ANY) \
    X(asm_1536b,       22,   0,  48,  48, OS_DESKTOP, GPU_AMD) \
    X(unroll_1536b,    23,   0,  48,  48, OS_ANY, GPU_ANY) \
    X(asm_2048b,       24,   0,  64,  64, OS_DESKTOP, GPU_AMD) \
    X(unroll_2048b,    25,   0,  64,  64, OS_ANY, GPU_ANY) \
    X(asm_2560b,       26,   0,  80,  80, OS_DESKTOP, GPU_AMD) \
    X(unroll_2560b,    27,   0,  80,  80, OS_ANY, GPU_ANY) \
    X(asm_3072b,       28,   0,  96,  96, OS_DESKTOP, GPU_AMD) \
    X(unroll_3072b,    29,   0,  96,  96, OS_ANY, GPU_ANY) \
    X(asm_3584b,       30,   0, 112, 112, OS_DESKTOP, GPU_AMD) \
    X(unroll_3584b,    31,   0, 112, 112, OS_ANY, GPU_ANY) \
    X(asm_4096b,       32,   0, 128, 128, OS_DESKTOP, GPU_AMD) \
    X(unroll_4096b,    33,   0, 128, 128, OS_ANY, GPU_ANY) \
    X(asm_4608b,       34,   0, 144, 144, OS_DESKTOP, GPU_AMD) \
    X(unroll_4608b,    35,   0, 144, 144, OS_ANY, GPU_ANY) \
    X(asm_5120b,       36,   0, 160, 160, OS_DESKTOP, GPU_AMD) \
    X(unroll_5120b,    37,   0, 160, 160, OS_ANY, GPU_ANY) \
    X(asm_5632b,       38,   0, 176, 176, OS_DESKTOP, GPU_AMD) \
    X(unroll_5632b,    39,   0, 176, 176, OS_ANY, GPU_ANY) \
    X(asm_6144b,       40,   0, 192, 192, OS_DESKTOP, GPU_AMD) \
    X(unroll_6144b,    41,   0, 192, 192, OS_ANY, GPU_ANY) \
    X(asm_6656b,       42,   0, 208, 208, OS_DESKTOP, GPU_AMD) \
    X(unroll_6656b,    43,   0, 208, 208, OS_ANY, GPU_ANY) \
    X(asm_7168b,       44,   0, 224, 224, OS_DESKTOP, GPU_AMD) \
    X(unroll_7168b,    45,   0, 224, 224, OS_ANY, GPU_ANY) \
    X(asm_7680b,       46,   0, 240, 240, OS_DESKTOP, GPU_AMD) \
    X(unroll_7680b,    47,   0, 240, 240, OS_ANY, GPU_ANY) \
    X(asm_8192b,       48,   0, 256, 256, OS_DESKTOP, GPU_AMD) \
    X(unroll_8192b,    49,   0, 256, 256, OS_ANY, GPU_ANY) \
    X(fused,         1000,   0,   0,   0, OS_ANY, GPU_ANY) \
    X(fused_unroll,  1001,   0,   0,   0, OS_ANY, GPU_ANY)

#define ECM_ADD_ROW(idt, ...)  \
    {#idt, "add_mod_" #idt, kAddAliases_##idt, "add_mod/add_mod_" #idt ".cl", __VA_ARGS__},
#define ECM_SUB_ROW(idt, ...)  \
    {#idt, "sub_mod_" #idt, kSubAliases_##idt, "sub_mod/sub_mod_" #idt ".cl", __VA_ARGS__},

constexpr EcmAddSubPathDescriptor kAddModRegistry[] = {ECM_ADDSUB_OPERATORS(ECM_ADD_ROW)};
constexpr EcmAddSubPathDescriptor kSubModRegistry[] = {ECM_ADDSUB_OPERATORS(ECM_SUB_ROW)};

static const char *const kSpecialMultAliases_unroll_192b[]  = {"unroll_192b", nullptr};
static const char *const kSpecialMultAliases_unroll_256b[]  = {"unroll_256b", nullptr};
static const char *const kSpecialMultAliases_unroll_384b[]  = {"unroll_384b", nullptr};
static const char *const kSpecialMultAliases_unroll_512b[]  = {"unroll_512b", nullptr};
static const char *const kSpecialMultAliases_unroll_768b[]  = {"unroll_768b", nullptr};
static const char *const kSpecialMultAliases_unroll_1024b[] = {"unroll_1024b", nullptr};
static const char *const kSpecialMultAliases_unroll_1536b[] = {"unroll_1536b", nullptr};
static const char *const kSpecialMultAliases_unroll_2048b[] = {"unroll_2048b", nullptr};
static const char *const kSpecialMultAliases_unroll_2560b[] = {"unroll_2560b", nullptr};
static const char *const kSpecialMultAliases_unroll_3072b[] = {"unroll_3072b", nullptr};
static const char *const kSpecialMultAliases_unroll_3584b[] = {"unroll_3584b", nullptr};
static const char *const kSpecialMultAliases_unroll_4096b[] = {"unroll_4096b", nullptr};
static const char *const kSpecialMultAliases_unroll_4608b[] = {"unroll_4608b", nullptr};
static const char *const kSpecialMultAliases_unroll_5120b[] = {"unroll_5120b", nullptr};
static const char *const kSpecialMultAliases_unroll_5632b[] = {"unroll_5632b", nullptr};
static const char *const kSpecialMultAliases_unroll_6144b[] = {"unroll_6144b", nullptr};
static const char *const kSpecialMultAliases_unroll_6656b[] = {"unroll_6656b", nullptr};
static const char *const kSpecialMultAliases_unroll_7168b[] = {"unroll_7168b", nullptr};
static const char *const kSpecialMultAliases_unroll_7680b[] = {"unroll_7680b", nullptr};
static const char *const kSpecialMultAliases_unroll_8192b[] = {"unroll_8192b", nullptr};
static const char *const kSpecialMultAliases_generic[] = {"generic", nullptr};

#define ECM_SPECIAL_MULT_OPERATORS(X) \
    X(unroll_192b,    5,   6,   6, OS_ANY, GPU_ANY)    \
    X(unroll_256b,    6,   8,   8, OS_ANY, GPU_ANY)    \
    X(unroll_384b,    7,  12,  12, OS_ANY, GPU_ANY)   \
    X(unroll_512b,   10,  16,  16, OS_ANY, GPU_ANY)   \
    X(unroll_768b,   11,  24,  24, OS_ANY, GPU_ANY)   \
    X(unroll_1024b,  12,  32,  32, OS_ANY, GPU_ANY)   \
    X(unroll_1536b,  13,  48,  48, OS_ANY, GPU_ANY)   \
    X(unroll_2048b,  14,  64,  64, OS_ANY, GPU_ANY)   \
    X(unroll_2560b,  15,  80,  80, OS_ANY, GPU_ANY)   \
    X(unroll_3072b,  16,  96,  96, OS_ANY, GPU_ANY)   \
    X(unroll_3584b,  17, 112, 112, OS_ANY, GPU_ANY)   \
    X(unroll_4096b,  18, 128, 128, OS_ANY, GPU_ANY)   \
    X(unroll_4608b,  19, 144, 144, OS_ANY, GPU_ANY)   \
    X(unroll_5120b,  20, 160, 160, OS_ANY, GPU_ANY)   \
    X(unroll_5632b,  21, 176, 176, OS_ANY, GPU_ANY)   \
    X(unroll_6144b,  22, 192, 192, OS_ANY, GPU_ANY)   \
    X(unroll_6656b,  23, 208, 208, OS_ANY, GPU_ANY)   \
    X(unroll_7168b,  24, 224, 224, OS_ANY, GPU_ANY)   \
    X(unroll_7680b,  25, 240, 240, OS_ANY, GPU_ANY)   \
    X(unroll_8192b,  26, 256, 256, OS_ANY, GPU_ANY)   \
    X(generic,     1000,   0,   0, OS_ANY, GPU_ANY)

#define ECM_SPECIAL_MULT_ROW(idt, prio, ...) \
    {#idt, "special_mult_ui32_" #idt, kSpecialMultAliases_##idt, "special_mult/special_mult_" #idt ".cl", prio, __VA_ARGS__},

constexpr EcmSpecialMultPathDescriptor kSpecialMultRegistry[] = {
    ECM_SPECIAL_MULT_OPERATORS(ECM_SPECIAL_MULT_ROW)
};

bool ecm_path_mask_fits(uint32_t required_mask, uint32_t runtime_mask) {
    if (required_mask != 0u && required_mask != OS_ANY && required_mask != GPU_ANY) {
        if ((runtime_mask & required_mask) == 0u) {
            return false;
        }
    }
    return true;
}

static constexpr const char *kEcmStage1CommonConfig = "common/stage1_config.h.cl";
static constexpr const char *kEcmStage1CommonMpPriv = "common/mp_priv.h.cl";
static constexpr const char *kEcmStage1LadderHelpers = "common/ladder_helpers.h.cl";
static constexpr const char *kEcmStage1AsmCommon = "common/asm_common.h.cl";
static constexpr const char *kEcmStage1OperatorIface = "common/operator_iface.h.cl";
static constexpr const char *kEcmStage1Coop = "ecm_stage1_coop.cl";
static constexpr const char *kEcmStage1Entry = "ecm_stage1.cl";
static constexpr const char *kEcmStage1EntryLocal = "ecm_stage1_local.cl";

std::string mont_kernel_path_local(const EcmMontPathDescriptor *desc) {
    if (desc == nullptr || desc->kernel_path == nullptr) return "";
    std::string s(desc->kernel_path);
    // kernel_path is e.g. "mont_mul/mont_mul_unroll_8192b.cl"
    // → needs to become  "mont_mul/mont_mul_unroll_8192b_local.cl"
    size_t dot = s.rfind(".cl");
    if (dot != std::string::npos) {
        s.insert(dot, "_local");
    }
    return s;
}
static constexpr const char *kEcmStage1Sliced = "ecm_stage1_sliced.cl";
static constexpr const char *kEcmStage1SlicedT16x2 = "ecm_stage1_sliced_t16x2.cl";
static constexpr const char *kMontFips4096Kernel = "mont_mul/mont_mul_fips_4096b.cl";

bool mont_kernel_path_needs_fips4096(const char *kernel_path) {
    return kernel_path != nullptr && strcmp(kernel_path, kMontFips4096Kernel) == 0;
}

bool addsub_kernel_path_needs_asm_base(const char *kernel_path) {
    return kernel_path != nullptr && strstr(kernel_path, "_asm_") != nullptr;
}

bool plan_needs_asm_common(const EcmStage1KernelBuildPlan &plan) {
    return (plan.add != nullptr && addsub_kernel_path_needs_asm_base(plan.add->kernel_path)) ||
           (plan.sub != nullptr && addsub_kernel_path_needs_asm_base(plan.sub->kernel_path));
}

int stage1_coop_wg_for_plan(const EcmStage1KernelBuildPlan &plan) {
    int coop_wg = 1;
    if (plan.mul != nullptr) {
        coop_wg = std::max(coop_wg, static_cast<int>(plan.mul->coop_work_group_size));
    }
    if (plan.sqr != nullptr) {
        coop_wg = std::max(coop_wg, static_cast<int>(plan.sqr->coop_work_group_size));
    }
    return coop_wg;
}

void append_unique_kernel_path(std::vector<const char *> &paths, const char *kernel_path) {
    if (kernel_path == nullptr || kernel_path[0] == '\0') {
        return;
    }
    for (const char *existing : paths) {
        if (strcmp(existing, kernel_path) == 0) {
            return;
        }
    }
    paths.push_back(kernel_path);
}

void append_impl_macro(std::string &out, const char *name, const char *symbol) {
    out += "#define ";
    out += name;
    out += " ";
    out += symbol;
    out += "\n";
}

int addsub_id_kernel_path(const char *id) {
    if (id == nullptr) {
        return ECM_ADDSUB_PATH_FUSED_UNROLL;
    }
    static const struct {
        const char *id;
        int path;
    } kMap[] = {
        {"fused", ECM_ADDSUB_PATH_FUSED},
        {"fused_unroll", ECM_ADDSUB_PATH_FUSED_UNROLL},
        {"unroll_4096b", ECM_ADDSUB_PATH_FUSED_UNROLL_B32},
        {"asm_4096b", ECM_ADDSUB_PATH_ASM_B32},
        {"unroll_512b", ECM_ADDSUB_PATH_FUSED_UNROLL_B16},
        {"asm_512b", ECM_ADDSUB_PATH_ASM_B16},
        {"unroll_128b", ECM_ADDSUB_PATH_UNROLL_128B},
        {"asm_128b", ECM_ADDSUB_PATH_ASM_128B},
        {"unroll_192b", ECM_ADDSUB_PATH_UNROLL_192B},
        {"asm_192b", ECM_ADDSUB_PATH_ASM_192B},
        {"unroll_256b", ECM_ADDSUB_PATH_UNROLL_256B},
        {"asm_256b", ECM_ADDSUB_PATH_ASM_256B},
        {"unroll_384b", ECM_ADDSUB_PATH_UNROLL_384B},
        {"asm_384b", ECM_ADDSUB_PATH_ASM_384B},
    };
    for (const auto &entry : kMap) {
        if (strcmp(id, entry.id) == 0) {
            return entry.path;
        }
    }
    return ECM_ADDSUB_PATH_FUSED_UNROLL;
}

bool alias_matches(const char *path, const char *alias) {
    return path != nullptr && alias != nullptr && strcmp(path, alias) == 0;
}

bool aliases_contain(const char *const *aliases, const char *path) {
    if (aliases == nullptr || path == nullptr) {
        return false;
    }
    for (const char *const *p = aliases; *p != nullptr; ++p) {
        if (alias_matches(path, *p)) {
            return true;
        }
    }
    return false;
}


// -----------------------------------------------------------------------------
// Generic operator resolver (shared by mul/sqr, add/sub, special_mult).
//
// All descriptor types expose `.id`, `.aliases`, `.auto_priority`; the only
// family-specific piece is the `fits(desc, limbs, runtime_mask)` predicate, which
// is passed in. Resolution is uniform:
//
//   * "auto"/"default"/empty -> first descriptor that fits, in auto_priority order
//     (manual-only entries, auto_priority < 0, are skipped).
//   * explicit id/alias      -> first alias match that fits this platform/container;
//                               if an alias matched but none fit, fall back to the
//                               same auto best-fit (NO hardcoded per-family fallback
//                               id such as priv_opt / fused_unroll / generic).
//   * unknown alias          -> nullptr, *unknown_path = true.
//
// Registry contract: every family includes one unconstrained, auto-eligible
// catch-all (e.g. priv_opt / fused_unroll / generic) so auto best-fit always
// resolves; that contract -- not special-case code -- guarantees a fallback.
// -----------------------------------------------------------------------------
template <class D, class FitsFn>
const D *resolve_operator_path(const D *reg, size_t count, const char *path, uint32_t limbs,
                               uint32_t runtime_mask, FitsFn fits, bool *unknown_path) {
    if (unknown_path != nullptr) {
        *unknown_path = false;
    }
    auto first_fitting_by_priority = [&]() -> const D * {
        std::vector<const D *> ordered;
        ordered.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (reg[i].auto_priority >= 0) {
                ordered.push_back(&reg[i]);
            }
        }
        std::stable_sort(ordered.begin(), ordered.end(),
                         [](const D *a, const D *b) { return a->auto_priority < b->auto_priority; });
        for (const D *d : ordered) {
            if (fits(d, limbs, runtime_mask)) {
                return d;
            }
        }
        return nullptr;
    };

    if (opencl_ecm_path_is_auto(path)) {
        return first_fitting_by_priority();
    }
    const D *matched = nullptr;
    for (size_t i = 0; i < count; ++i) {
        if (!aliases_contain(reg[i].aliases, path)) {
            continue;
        }
        if (fits(&reg[i], limbs, runtime_mask)) {
            return &reg[i];
        }
        if (matched == nullptr) {
            matched = &reg[i];
        }
    }
    if (matched != nullptr) {
        return first_fitting_by_priority();
    }
    if (unknown_path != nullptr) {
        *unknown_path = true;
    }
    return nullptr;
}

const EcmAddSubPathDescriptor *find_addsub_by_kernel_path(const EcmAddSubPathDescriptor *registry,
                                                          size_t count, int kernel_path) {
    for (size_t i = 0; i < count; ++i) {
        if (ecm_addsub_descriptor_kernel_path(&registry[i]) == kernel_path) {
            return &registry[i];
        }
    }
    return nullptr;
}

void append_define(std::string &opts, const char *macro, int value) {
    if (macro == nullptr || macro[0] == '\0') {
        return;
    }
    opts += " -D";
    opts += macro;
    opts += '=';
    opts += std::to_string(value);
}

int mont_kernel_path_for_plan(const EcmMontPathDescriptor *desc, uint32_t plan_limbs) {
    if (desc == nullptr || !desc->fixed_width) {
        return 0;
    }
    const uint32_t operator_limbs = ecm_mont_operator_limbs(desc);
    if (operator_limbs == 0u || plan_limbs != operator_limbs) {
        return 0;
    }
    return ecm_coop_kernel_path_from_desc(desc);
}

} // namespace

// Auto-fallback for mont mul/sqr: first-fitting descriptor in priority order.
// Used by stage1_clamp_mont_desc when the selected operator doesn't fit this
// limb size / container / platform.  Equivalent to auto-resolution with a null
// path — no hardcoded id strings.
const EcmMontPathDescriptor *mont_auto_fallback(const EcmMontPathDescriptor *registry, size_t count,
                                                uint32_t limbs, uint32_t runtime_mask) {
    std::vector<const EcmMontPathDescriptor *> ordered;
    ordered.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].auto_priority >= 0) {
            ordered.push_back(&registry[i]);
        }
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const EcmMontPathDescriptor *a, const EcmMontPathDescriptor *b) {
                         return a->auto_priority < b->auto_priority;
                     });
    for (const EcmMontPathDescriptor *d : ordered) {
        if (ecm_mont_path_fits(d, limbs, runtime_mask)) {
            return d;
        }
    }
    return nullptr;
}

bool ecm_path_limbs_fits(uint32_t min_limbs, uint32_t max_limbs, uint32_t limbs) {
    if (min_limbs > 0 && limbs < min_limbs) {
        return false;
    }
    if (max_limbs == 0) {
        return true;
    }
    return limbs <= max_limbs;
}

uint32_t ecm_path_host_os_mask() {
#if defined(_WIN32)
    return OS_WINDOWS;
#elif defined(__ANDROID__)
    return OS_ANDROID;
#elif defined(__APPLE__)
    return OS_MACOS;
#elif defined(__linux__)
    return OS_LINUX;
#else
    return OS_ANY;
#endif
}

uint32_t ecm_path_gpu_vendor_from_cl_vendor_string(const char *vendor_lower) {
    if (vendor_lower == nullptr || vendor_lower[0] == '\0') {
        return 0;
    }
    if (std::strstr(vendor_lower, "advanced micro devices") != nullptr ||
        std::strstr(vendor_lower, "amd") != nullptr) {
        return GPU_AMD;
    }
    if (std::strstr(vendor_lower, "nvidia") != nullptr) {
        return GPU_NVIDIA;
    }
    if (std::strstr(vendor_lower, "intel") != nullptr) {
        return GPU_INTEL;
    }
    if (std::strstr(vendor_lower, "qualcomm") != nullptr) {
        return GPU_QUALCOMM;
    }
    if (std::strstr(vendor_lower, "huawei") != nullptr ||
        std::strstr(vendor_lower, "hisilicon") != nullptr) {
        return GPU_HUAWEI;
    }
    if (std::strstr(vendor_lower, "apple") != nullptr) {
        return GPU_APPLE;
    }
    return 0;
}

uint32_t ecm_mont_operator_limbs(const EcmMontPathDescriptor *desc) {
    if (desc == nullptr || !desc->fixed_width || desc->max_limbs == 0) {
        return 0;
    }
    return desc->max_limbs;
}

bool ecm_mont_path_fits(const EcmMontPathDescriptor *desc, uint32_t limbs, uint32_t runtime_mask) {
    if (desc == nullptr) {
        return false;
    }
    // Fixed-width operators must declare their exact container size.
    assert(!desc->fixed_width || desc->max_container_limbs > 0);
    if (!ecm_path_limbs_fits(desc->min_limbs, desc->max_limbs, limbs)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->os_mask, runtime_mask & OS_ANY)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->gpu_vendor_mask, runtime_mask & GPU_ANY)) {
        return false;
    }
    // max_container_limbs check
    if (desc->max_container_limbs == 0) {
        return true;
    }
    if (desc->fixed_width && desc->max_limbs > 0) {
        return limbs >= desc->max_limbs;
    }
    return limbs <= desc->max_container_limbs;
}

bool ecm_addsub_path_fits(const EcmAddSubPathDescriptor *desc, uint32_t limbs, uint32_t runtime_mask) {
    if (desc == nullptr) {
        return false;
    }
    if (!ecm_path_limbs_fits(desc->min_limbs, desc->max_limbs, limbs)) {
        return false;
    }
    if (desc->max_container_limbs > 0 && limbs > desc->max_container_limbs) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->os_mask, runtime_mask & OS_ANY)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->gpu_vendor_mask, runtime_mask & GPU_ANY)) {
        return false;
    }
    return true;
}

bool ecm_special_mult_path_fits(const EcmSpecialMultPathDescriptor *desc, uint32_t limbs,
                                    uint32_t runtime_mask) {
    if (desc == nullptr) {
        return false;
    }
    if (!ecm_path_limbs_fits(desc->min_limbs, desc->max_limbs, limbs)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->os_mask, runtime_mask & OS_ANY)) {
        return false;
    }
    if (!ecm_path_mask_fits(desc->gpu_vendor_mask, runtime_mask & GPU_ANY)) {
        return false;
    }
    return true;
}

bool opencl_ecm_path_is_auto(const char *path) {
    return path == nullptr || path[0] == '\0' || strcmp(path, "auto") == 0 ||
           strcmp(path, "default") == 0;
}

int ecm_addsub_descriptor_kernel_path(const EcmAddSubPathDescriptor *desc) {
    return addsub_id_kernel_path(desc != nullptr ? desc->id : nullptr);
}

int ecm_coop_kernel_path_from_desc(const EcmMontPathDescriptor *desc) {
    if (desc == nullptr || desc->cl_name == nullptr) return 0;
    if (strstr(desc->cl_name, "fips4096_mt16"))  return EcmCoopKernelPath_FIPS4096_MT16;
    if (strstr(desc->cl_name, "fips4096_mt8"))   return EcmCoopKernelPath_FIPS4096_MT8;
    if (strstr(desc->cl_name, "fips_4096b"))     return EcmCoopKernelPath_FIPS4096;
    if (strstr(desc->cl_name, "karatsuba_2048b")) return EcmCoopKernelPath_K2048_MT4;
    return EcmCoopKernelPath_None;
}

std::vector<const char *> opencl_ecm_stage1_kernel_source_paths(
    const EcmStage1KernelBuildPlan &plan) {
    std::vector<const char *> paths;
    // Keep dynamically-constructed strings alive beyond function return.
    static std::vector<std::string> local_strings;
    local_strings.clear();
    append_unique_kernel_path(paths, kEcmStage1CommonConfig);
    append_unique_kernel_path(paths, kEcmStage1CommonMpPriv);
    append_unique_kernel_path(paths, kEcmStage1LadderHelpers);
    if (plan_needs_asm_common(plan)) {
        append_unique_kernel_path(paths, kEcmStage1AsmCommon);
    }
    if (plan.mul != nullptr) {
        if (plan.local) {
            local_strings.push_back(mont_kernel_path_local(plan.mul));
            const std::string &s = local_strings.back();
            if (!s.empty()) append_unique_kernel_path(paths, s.c_str());
        } else {
            append_unique_kernel_path(paths, plan.mul->kernel_path);
        }
    }
    if (plan.sqr != nullptr) {
        if (plan.local) {
            local_strings.push_back(mont_kernel_path_local(plan.sqr));
            const std::string &s = local_strings.back();
            if (!s.empty()) append_unique_kernel_path(paths, s.c_str());
        } else {
            append_unique_kernel_path(paths, plan.sqr->kernel_path);
        }
    }
    if (plan.add != nullptr) {
        append_unique_kernel_path(paths, plan.add->kernel_path);
    }
    if (plan.sub != nullptr) {
        append_unique_kernel_path(paths, plan.sub->kernel_path);
    }
    if (plan.special_mult != nullptr && plan.special_mult->kernel_path != nullptr) {
        append_unique_kernel_path(paths, plan.special_mult->kernel_path);
    }
    if (!plan.local) {
        append_unique_kernel_path(paths, kEcmStage1OperatorIface);
        if (stage1_coop_wg_for_plan(plan) > 1) {
            append_unique_kernel_path(paths, kEcmStage1Coop);
        }
    }
    append_unique_kernel_path(paths, plan.local ? kEcmStage1EntryLocal : kEcmStage1Entry);
    if (!plan.local) {
        append_unique_kernel_path(paths, kEcmStage1Entry);
    }
    // Sliced kernels hard-reference 1024b fixed-width operators; ensure their
    // definitions are included regardless of which operators are selected.
    append_unique_kernel_path(paths, "mont_mul/mont_mul_unroll_1024b.cl");
    append_unique_kernel_path(paths, "add_mod/add_mod_asm_1024b.cl");
    append_unique_kernel_path(paths, "sub_mod/sub_mod_asm_1024b.cl");
    append_unique_kernel_path(paths, "special_mult/special_mult_unroll_1024b.cl");
    if (!plan.local) {
        append_unique_kernel_path(paths, kEcmStage1Entry);
    }
    // Always include sliced — it has NVIDIA stubs for the AMD intrinsics
    append_unique_kernel_path(paths, kEcmStage1Sliced);
    append_unique_kernel_path(paths, kEcmStage1SlicedT16x2);
    return paths;
}

std::string opencl_ecm_stage1_assemble_kernel_source(
    const EcmStage1KernelBuildPlan &plan,
    const std::function<std::string(const char *rel_path)> &load_file) {
    std::string source;
    source.reserve(65536);
    if (plan.mul == nullptr || plan.sqr == nullptr || plan.add == nullptr || plan.sub == nullptr ||
        plan.special_mult == nullptr) {
        return source;
    }
    if (plan.local) {
        // Local kernel: suffix _local on mont mul/sqr cl_names and kernel paths.
        std::string mul_local(plan.mul->cl_name);
        mul_local += "_local";
        std::string sqr_local(plan.sqr->cl_name);
        sqr_local += "_local";
        append_impl_macro(source, "ECM_STAGE1_MUL_IMPL", mul_local.c_str());
        append_impl_macro(source, "ECM_STAGE1_SQR_IMPL", sqr_local.c_str());
    } else {
        append_impl_macro(source, "ECM_STAGE1_MUL_IMPL", plan.mul->cl_name);
        append_impl_macro(source, "ECM_STAGE1_SQR_IMPL", plan.sqr->cl_name);
    }
    append_impl_macro(source, "ECM_STAGE1_ADD_IMPL", plan.add->cl_name);
    append_impl_macro(source, "ECM_STAGE1_SUB_IMPL", plan.sub->cl_name);
    append_impl_macro(source, "ECM_STAGE1_SPECIAL_MULT_IMPL", plan.special_mult->cl_name);
    source += "\n";

    const std::vector<const char *> paths = opencl_ecm_stage1_kernel_source_paths(plan);
    for (const char *rel_path : paths) {
        const std::string chunk = load_file(rel_path);
        if (chunk.empty()) {
            return std::string();
        }
        if (!source.empty() && source.back() != '\n') {
            source += "\n";
        }
        source += chunk;
        if (chunk.back() != '\n') {
            source += "\n";
        }
    }
    return source;
}

size_t opencl_ecm_mont_mul_registry_count() {
    return sizeof(kMontMulRegistry) / sizeof(kMontMulRegistry[0]);
}

const EcmMontPathDescriptor *opencl_ecm_mont_mul_registry_entry(size_t index) {
    if (index >= opencl_ecm_mont_mul_registry_count()) {
        return nullptr;
    }
    return &kMontMulRegistry[index];
}

size_t opencl_ecm_mont_sqr_registry_count() {
    return sizeof(kMontSqrRegistry) / sizeof(kMontSqrRegistry[0]);
}

const EcmMontPathDescriptor *opencl_ecm_mont_sqr_registry_entry(size_t index) {
    if (index >= opencl_ecm_mont_sqr_registry_count()) {
        return nullptr;
    }
    return &kMontSqrRegistry[index];
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_mul(const char *path, const EcmPathContext &ctx,
                                                         bool *unknown_path) {
    return resolve_operator_path(kMontMulRegistry, opencl_ecm_mont_mul_registry_count(), path,
                                 ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask, ecm_mont_path_fits,
                                 unknown_path);
}

const EcmMontPathDescriptor *opencl_ecm_resolve_mont_sqr(const char *path, const EcmPathContext &ctx,
                                                         bool *unknown_path) {
    return resolve_operator_path(kMontSqrRegistry, opencl_ecm_mont_sqr_registry_count(), path,
                                 ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask, ecm_mont_path_fits,
                                 unknown_path);
}

size_t opencl_ecm_addmod_registry_count() {
    return sizeof(kAddModRegistry) / sizeof(kAddModRegistry[0]);
}

const EcmAddSubPathDescriptor *opencl_ecm_addmod_registry_entry(size_t index) {
    if (index >= opencl_ecm_addmod_registry_count()) {
        return nullptr;
    }
    return &kAddModRegistry[index];
}

const EcmAddSubPathDescriptor *opencl_ecm_addmod_descriptor_by_kernel_path(int kernel_path) {
    return find_addsub_by_kernel_path(kAddModRegistry, opencl_ecm_addmod_registry_count(),
                                      kernel_path);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addmod_path(const char *path,
                                                              const EcmPathContext &ctx) {
    return resolve_operator_path(kAddModRegistry, opencl_ecm_addmod_registry_count(), path,
                                 ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask, ecm_addsub_path_fits,
                                 nullptr);
}

size_t opencl_ecm_submod_registry_count() {
    return sizeof(kSubModRegistry) / sizeof(kSubModRegistry[0]);
}

const EcmAddSubPathDescriptor *opencl_ecm_submod_registry_entry(size_t index) {
    if (index >= opencl_ecm_submod_registry_count()) {
        return nullptr;
    }
    return &kSubModRegistry[index];
}

const EcmAddSubPathDescriptor *opencl_ecm_submod_descriptor_by_kernel_path(int kernel_path) {
    return find_addsub_by_kernel_path(kSubModRegistry, opencl_ecm_submod_registry_count(),
                                      kernel_path);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_submod_path(const char *path,
                                                              const EcmPathContext &ctx) {
    return resolve_operator_path(kSubModRegistry, opencl_ecm_submod_registry_count(), path,
                                 ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask, ecm_addsub_path_fits,
                                 nullptr);
}

size_t opencl_ecm_special_mult_registry_count() {
    return sizeof(kSpecialMultRegistry) / sizeof(kSpecialMultRegistry[0]);
}

const EcmSpecialMultPathDescriptor *opencl_ecm_special_mult_registry_entry(size_t index) {
    if (index >= opencl_ecm_special_mult_registry_count()) {
        return nullptr;
    }
    return &kSpecialMultRegistry[index];
}

const EcmSpecialMultPathDescriptor *opencl_ecm_resolve_special_mult(const char *path,
                                                                     const EcmPathContext &ctx) {
    return resolve_operator_path(kSpecialMultRegistry, opencl_ecm_special_mult_registry_count(),
                                 path, ctx.limbs, ctx.os_mask | ctx.gpu_vendor_mask,
                                 ecm_special_mult_path_fits, nullptr);
}

int ecm_special_mult_descriptor_kernel_path(const EcmSpecialMultPathDescriptor *desc) {
    if (desc == nullptr || desc->id == nullptr) {
        return 0;
    }
    if (strcmp(desc->id, "unroll_512b") == 0) return 1;
    if (strcmp(desc->id, "generic") == 0) return 0;
    return 0;
}

EcmStage1KernelBuildPlan opencl_ecm_stage1_make_build_plan(
    uint32_t limbs, uint32_t tpi, const EcmMontPathDescriptor *mul,
    const EcmMontPathDescriptor *sqr, const EcmAddSubPathDescriptor *add,
    const EcmAddSubPathDescriptor *sub, const EcmSpecialMultPathDescriptor *special_mult,
    int stage1_force_normalize, int add_mod_fused_unroll, bool local, int wg_size) {
    EcmStage1KernelBuildPlan plan{};
    plan.limbs = limbs;
    plan.tpi = tpi;
    plan.stage1_force_normalize = stage1_force_normalize;
    plan.add_mod_fused_unroll = add_mod_fused_unroll;
    plan.mul = mul;
    plan.sqr = sqr;
    plan.add = add;
    plan.sub = sub;
    plan.special_mult = special_mult;
    plan.local = local;
    plan.wg_size = wg_size;
    return plan;
}

bool opencl_ecm_stage1_build_plan_equal(const EcmStage1KernelBuildPlan &a,
                                        const EcmStage1KernelBuildPlan &b) {
    return a.limbs == b.limbs && a.tpi == b.tpi &&
           a.stage1_force_normalize == b.stage1_force_normalize &&
           a.add_mod_fused_unroll == b.add_mod_fused_unroll && a.mul == b.mul && a.sqr == b.sqr &&
           a.add == b.add && a.sub == b.sub && a.special_mult == b.special_mult &&
           a.local == b.local && a.wg_size == b.wg_size;
}

std::string opencl_ecm_stage1_generate_build_options(const EcmStage1KernelBuildPlan &plan) {
    std::string opts;
    opts.reserve(512);
    opts += "-DMAX_LIMBS=";
    opts += std::to_string(plan.limbs);
    opts += " -DTPI=";
    opts += std::to_string(plan.tpi);
    append_define(opts, "ECM_STAGE1_FORCE_NORMALIZE", plan.stage1_force_normalize);
    append_define(opts, "MP_ADD_MOD_FUSED_UNROLL", plan.add_mod_fused_unroll);
    append_define(opts, "ECM_STAGE1_KERNEL_REV", 14);
    append_define(opts, "MP_LIMB_BITS", 32);

    append_define(opts, "ECM_STAGE1_MUL_PATH", mont_kernel_path_for_plan(plan.mul, plan.limbs));
    append_define(opts, "ECM_STAGE1_SQR_PATH", mont_kernel_path_for_plan(plan.sqr, plan.limbs));

    const int coop_wg = stage1_coop_wg_for_plan(plan);
    int coop_scratch = 0;
    uint32_t coop_container_limbs = 128u;
    if (plan.mul != nullptr && plan.mul->coop_work_group_size > 1u) {
        coop_scratch = std::max(coop_scratch, static_cast<int>(plan.mul->local_scratch_u32));
        if (plan.mul->max_container_limbs > 0u)
            coop_container_limbs = plan.mul->max_container_limbs;
    }
    if (plan.sqr != nullptr && plan.sqr->coop_work_group_size > 1u) {
        coop_scratch = std::max(coop_scratch, static_cast<int>(plan.sqr->local_scratch_u32));
        if (plan.sqr->max_container_limbs > 0u)
            coop_container_limbs = plan.sqr->max_container_limbs;
    }
    const bool has_fips4096 =
        mont_kernel_path_needs_fips4096(plan.mul != nullptr ? plan.mul->kernel_path : nullptr) ||
        mont_kernel_path_needs_fips4096(plan.sqr != nullptr ? plan.sqr->kernel_path : nullptr);
    append_define(opts, "ECM_STAGE1_COOP_WG", coop_wg);
    if (coop_wg > 1) append_define(opts, "ECM_STAGE1_USE_COOP_WG", 1);
    append_define(opts, "ECM_STAGE1_COOP_SCRATCH_U32", coop_scratch);
    append_define(opts, "ECM_COOP_CONTAINER_LIMBS", coop_container_limbs);
    append_define(opts, "ECM_STAGE1_HAS_FIPS4096", has_fips4096 ? 1 : 0);

    if (plan.local) {
        uint32_t wg = (plan.wg_size > 0) ? (uint32_t)plan.wg_size
                        : (plan.limbs <= 128u) ? 16u : 8u;
        append_define(opts, "ECM_STAGE1_WG_SIZE", static_cast<int>(wg));
    } else if (plan.wg_size > 0) {
        append_define(opts, "ECM_STAGE1_WG_SIZE", plan.wg_size);
    }

    return opts;
}

const EcmMontPathDescriptor *opencl_ecm_stage1_compatible_mont_fallback(size_t n_bit_size, uint32_t limbs) {
    EcmPathContext ctx{};
    ctx.limbs = limbs;
    ctx.n_bit_size = n_bit_size;
    return opencl_ecm_resolve_mont_mul(nullptr, ctx, nullptr);
}

const char *opencl_ecm_special_mult_cl_name(const EcmSpecialMultPathDescriptor *desc,
                                             const char *fallback_cl_name) {
    if (desc != nullptr && desc->cl_name != nullptr) {
        return desc->cl_name;
    }
    return fallback_cl_name != nullptr ? fallback_cl_name : "special_mult_ui32_generic";
}

int opencl_ecm_parse_addsub_path(const char *path) {
    if (opencl_ecm_path_is_auto(path)) {
        return -1;
    }
    for (size_t i = 0; i < opencl_ecm_addmod_registry_count(); ++i) {
        const EcmAddSubPathDescriptor *desc = opencl_ecm_addmod_registry_entry(i);
        if (desc != nullptr && aliases_contain(desc->aliases, path)) {
            return ecm_addsub_descriptor_kernel_path(desc);
        }
    }
    for (size_t i = 0; i < opencl_ecm_submod_registry_count(); ++i) {
        const EcmAddSubPathDescriptor *desc = opencl_ecm_submod_registry_entry(i);
        if (desc != nullptr && aliases_contain(desc->aliases, path)) {
            return ecm_addsub_descriptor_kernel_path(desc);
        }
    }
    return -2;
}

const char *opencl_ecm_addsub_path_name(int path_id) {
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    if (add_d != nullptr && add_d->cl_name != nullptr) {
        return add_d->cl_name;
    }
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    if (sub_d != nullptr && sub_d->cl_name != nullptr) {
        return sub_d->cl_name;
    }
    return "unknown";
}

bool opencl_ecm_addsub_path_needs_asm_b32(int path_id) {
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    if (add_d != nullptr && strcmp(add_d->id, "asm_4096b") == 0) {
        return true;
    }
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    return sub_d != nullptr && strcmp(sub_d->id, "asm_4096b") == 0;
}

bool opencl_ecm_addsub_path_needs_asm_b16(int path_id) {
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    if (add_d != nullptr && strcmp(add_d->id, "asm_512b") == 0) {
        return true;
    }
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    return sub_d != nullptr && strcmp(sub_d->id, "asm_512b") == 0;
}

bool opencl_ecm_addsub_path_needs_addsub_bits(int path_id) {
    static const char *const kBitsIds[] = {"asm_128b", "unroll_128b", "asm_192b", "unroll_192b",
                                           "asm_256b", "unroll_256b", "asm_384b", "unroll_384b"};
    const EcmAddSubPathDescriptor *add_d = opencl_ecm_addmod_descriptor_by_kernel_path(path_id);
    const EcmAddSubPathDescriptor *sub_d = opencl_ecm_submod_descriptor_by_kernel_path(path_id);
    for (const char *bid : kBitsIds) {
        if (add_d != nullptr && add_d->id != nullptr && strcmp(add_d->id, bid) == 0) {
            return true;
        }
        if (sub_d != nullptr && sub_d->id != nullptr && strcmp(sub_d->id, bid) == 0) {
            return true;
        }
    }
    return false;
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_add_path(const char *path,
                                                                  const EcmPathContext &ctx) {
    return opencl_ecm_resolve_addmod_path(path, ctx);
}

const EcmAddSubPathDescriptor *opencl_ecm_resolve_addsub_sub_path(const char *path,
                                                                  const EcmPathContext &ctx) {
    return opencl_ecm_resolve_submod_path(path, ctx);
}

namespace {

std::string showkernel_fmt_aliases(const char *const *aliases, const char *id) {
    std::string s;
    if (aliases != nullptr) {
        for (const char *const *p = aliases; *p != nullptr; ++p) {
            if (id != nullptr && std::strcmp(*p, id) == 0) {
                continue;  // primary id is shown separately; skip it in the alias list
            }
            if (!s.empty()) {
                s += ", ";
            }
            s += *p;
        }
    }
    return s;
}

const char *showkernel_os_name(uint32_t bit) {
    switch (bit) {
    case OS_WINDOWS: return "win";
    case OS_ANDROID: return "android";
    case OS_LINUX: return "linux";
    case OS_MACOS: return "mac";
    default: return "?";
    }
}

const char *showkernel_gpu_name(uint32_t bit) {
    switch (bit) {
    case GPU_AMD: return "amd";
    case GPU_NVIDIA: return "nvidia";
    case GPU_INTEL: return "intel";
    case GPU_QUALCOMM: return "qualcomm";
    case GPU_HUAWEI: return "huawei";
    case GPU_APPLE: return "apple";
    default: return "?";
    }
}

std::string showkernel_join_bits(uint32_t mask, const char *(*nm)(uint32_t)) {
    std::string s;
    for (uint32_t b = mask; b != 0u; b &= (b - 1u)) {
        const uint32_t lb = b & (0u - b);
        if (!s.empty()) {
            s += "|";
        }
        s += nm(lb);
    }
    return s.empty() ? std::string("-") : s;
}

// Decode os/gpu whitelist masks.
std::string showkernel_platform(uint32_t os_mask, uint32_t gpu_mask) {
    std::string s = "os=";
    s += (os_mask == 0u || os_mask == OS_ANY)
            ? "any"
            : showkernel_join_bits(os_mask & OS_ANY, showkernel_os_name);
    s += " gpu=";
    s += (gpu_mask == 0u || gpu_mask == GPU_ANY)
            ? "any"
            : showkernel_join_bits(gpu_mask & GPU_ANY, showkernel_gpu_name);
    return s;
}

void showkernel_print_row(FILE *out, const char *id, const char *cl_name, const char *kernel_path,
                            const char *const *aliases, int prio, uint32_t min_limbs,
                            uint32_t max_limbs, const std::string &platform) {
    const std::string al = showkernel_fmt_aliases(aliases, id);
    fprintf(out, "  %-20s -> %s\n", id != nullptr ? id : "(null)",
            cl_name != nullptr ? cl_name : "(null)");
    fprintf(out, "      file=%s  prio=%d  limbs=%u..%u  %s\n",
            kernel_path != nullptr ? kernel_path : "-", prio, min_limbs, max_limbs,
            platform.c_str());
    if (!al.empty()) {
        fprintf(out, "      aliases: %s\n", al.c_str());
    }
}

} // namespace

void opencl_ecm_print_available_kernels(FILE *out) {
    if (out == nullptr) {
        out = stdout;
    }
    fprintf(out, "ECM OpenCL operator registry (mul/sqr, add, sub, special_mult resolved "
                 "independently).\n");
    fprintf(out, "Select with --mul/--sqr/--add/--sub/--special-mult <id|alias> (or auto/default).\n");
    fprintf(out, "prio: lower = preferred by auto-select (-1 = manual only); limbs: container "
                 "range; excl: excluded OS/GPU.\n\n");

    fprintf(out, "== Montgomery mul (--mul) ==\n");
    for (size_t i = 0; i < opencl_ecm_mont_mul_registry_count(); ++i) {
        const EcmMontPathDescriptor *d = opencl_ecm_mont_mul_registry_entry(i);
        if (d == nullptr) continue;
        showkernel_print_row(
            out, d->id, d->cl_name, d->kernel_path, d->aliases,
            d->auto_priority, d->min_limbs, d->max_limbs, 
            showkernel_platform(d->os_mask, d->gpu_vendor_mask)
        );
    }

    fprintf(out, "\n== Montgomery sqr (--sqr) ==\n");
    for (size_t i = 0; i < opencl_ecm_mont_sqr_registry_count(); ++i) {
        const EcmMontPathDescriptor *d = opencl_ecm_mont_sqr_registry_entry(i);
        if (d == nullptr) continue;
        showkernel_print_row(
            out, d->id, d->cl_name, d->kernel_path, d->aliases,
            d->auto_priority, d->min_limbs, d->max_limbs, 
            showkernel_platform(d->os_mask, d->gpu_vendor_mask)
        );
    }

    fprintf(out, "\n== Modular add (--add) ==\n");
    for (size_t i = 0; i < opencl_ecm_addmod_registry_count(); ++i) {
        const EcmAddSubPathDescriptor *d = opencl_ecm_addmod_registry_entry(i);
        if (d == nullptr) continue;
        showkernel_print_row(
            out, d->id, d->cl_name, d->kernel_path, d->aliases,
            d->auto_priority, d->min_limbs, d->max_limbs, 
            showkernel_platform(d->os_mask, d->gpu_vendor_mask)
        );
    }

    fprintf(out, "\n== Modular sub (--sub) ==\n");
    for (size_t i = 0; i < opencl_ecm_submod_registry_count(); ++i) {
        const EcmAddSubPathDescriptor *d = opencl_ecm_submod_registry_entry(i);
        if (d == nullptr) continue;
        showkernel_print_row(
            out, d->id, d->cl_name, d->kernel_path, d->aliases,
            d->auto_priority, d->min_limbs, d->max_limbs, 
            showkernel_platform(d->os_mask, d->gpu_vendor_mask)
        );
    }

    fprintf(out, "\n== special_mult (--special-mult) ==\n");
    for (size_t i = 0; i < opencl_ecm_special_mult_registry_count(); ++i) {
        const EcmSpecialMultPathDescriptor *d = opencl_ecm_special_mult_registry_entry(i);
        if (d == nullptr) continue;
        showkernel_print_row(
            out, d->id, d->cl_name, d->kernel_path, d->aliases,
            d->auto_priority, d->min_limbs, d->max_limbs, 
            showkernel_platform(d->os_mask, d->gpu_vendor_mask)
        );
    }
    fprintf(out, "\nSee docs/architecture/OPENCL.md for details.\n");
}
