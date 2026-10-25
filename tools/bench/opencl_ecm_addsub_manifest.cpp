#include "opencl_ecm_addsub_manifest.h"

#include <cstdio>
#include <cstring>

namespace {

#if defined(__ANDROID__)
constexpr bool kAndroidBenchLite = true;
#else
constexpr bool kAndroidBenchLite = false;
#endif

// rel paths already carry the "bench/..." prefix and resolve relative to the stage1
// kernel root (kernels/opencl). No cgbn root prepend anymore.
void push_if(std::vector<std::string> &paths, const char *rel) {
    paths.emplace_back(rel);
}

EcmAddSubBenchKernel kspec(const char *kname, const char *label, EcmAddSubTier tier, bool is_add,
                           bool amd_only, bool use_wg, int lpt_chunk, const char *compare,
                           bool hot_inner_loop = false) {
    return {kname, label, tier, is_add, amd_only, use_wg, lpt_chunk, compare, hot_inner_loop};
}

} // namespace

EcmAddSubBuildManifest opencl_ecm_addsub_build_manifest(uint32_t words, bool asm_enabled,
                                                        bool asm_b64_enabled) {
    EcmAddSubBuildManifest m;
    push_if(m.source_paths, "bench/ecm_addsub_bench.cl");
    // Manual full-unroll sources are ~1 MiB and include 4096-bit bodies; skip on Android @1024+.
    const bool skip_manual_unroll = kAndroidBenchLite && words >= 32u;
    if (!skip_manual_unroll) {
        push_if(m.source_paths, "bench/mp_addsub/generated/add_fused_unroll_manual.cl");
        push_if(m.source_paths, "bench/mp_addsub/generated/sub_fused_unroll_manual.cl");
    }
    push_if(m.source_paths, "bench/mp_addsub/generated/fused_unroll_auto.cl");
    if (asm_enabled) {
        push_if(m.source_paths, "bench/mp_addsub/asm_base.cl");
        push_if(m.source_paths, "bench/mp_addsub/generated/asm_block16.cl");
        if (words == 128u) {
            push_if(m.source_paths, "bench/mp_addsub/generated/asm_block32.cl");
            push_if(m.source_paths, "bench/mp_addsub/generated/asm_sub_block32.cl");
            if (asm_b64_enabled) {
                push_if(m.source_paths, "bench/mp_addsub/generated/asm_block64.cl");
                push_if(m.source_paths, "bench/mp_addsub/generated/asm_sub_block64.cl");
            }
        }
        push_if(m.source_paths, "bench/mp_addsub/generated/asm_add_kernels.cl");
        push_if(m.source_paths, "bench/mp_addsub/generated/asm_sub_kernels.cl");
    }
    return m;
}

static void push_add_asm(std::vector<EcmAddSubBenchKernel> &k, uint32_t words, bool asm_b64,
                         bool bench_unroll_only) {
    auto add = [&](const char *kname, const char *label) {
        k.push_back(kspec(kname, label, EcmAddSubTier::Asm, true, true, false, 0, "fused_unroll"));
    };
    if (words == 128u) {
        if (asm_b64) {
            add("ecm_mp_add_mod_fused_unroll_asm_b64", "fused_unroll_asm_b64");
        }
        add("ecm_mp_add_mod_fused_unroll_asm_b32", "fused_unroll_asm_b32");
        add("ecm_mp_add_mod_fused_unroll_asm_soft_b16", "fused_unroll_asm_soft_b16");
        add("ecm_mp_add_mod_fused_unroll_asm_b16", "fused_unroll_asm_b16");
        add("ecm_mp_add_mod_fused_unroll_asm_soft", "fused_unroll_asm_soft_b8");
        add("ecm_mp_add_mod_fused_unroll_asm_asmfix", "fused_unroll_asm_asmfix_b8");
        add("ecm_mp_add_mod_fused_unroll_asm", "fused_unroll_asm_b8");
    } else if (words == 16u) {
        if (!bench_unroll_only) {
            add("ecm_mp_add_mod_fused_asm_b16_vccsoft", "fused_asm_b16_vccsoft");
            add("ecm_mp_add_mod_fused_asm_b16", "fused_asm_b16");
        }
    } else if (words == 8u && !bench_unroll_only) {
        add("ecm_mp_add_mod_fused_asm8_vccsoft", "fused_asm_b8_vccsoft");
        add("ecm_mp_add_mod_fused_asm8_asmfix", "fused_asm_b8_asmfix");
        add("ecm_mp_add_mod_fused_asm8", "fused_asm_b8");
        add("ecm_mp_add_mod_fused_unroll_asm", "fused_unroll_asm_b8");
    }
}

std::vector<EcmAddSubBenchKernel> opencl_ecm_addsub_add_kernels(uint32_t words, bool asm_enabled,
                                                              bool asm_b64_enabled,
                                                              bool bench_unroll_only) {
    std::vector<EcmAddSubBenchKernel> k;
    k.push_back(kspec("ecm_mp_add_mod_fused_unroll_auto_hot", "fused_unroll_auto_hot",
                      EcmAddSubTier::FusedUnrollAuto, true, false, false, 0, "fused_unroll_auto", true));
    k.push_back(kspec("ecm_mp_add_mod_fused_hot", "fused_hot", EcmAddSubTier::Fused, true, false, false, 0,
                      "fused_unroll_auto_hot", true));
    if (asm_enabled) {
        push_add_asm(k, words, asm_b64_enabled, bench_unroll_only);
    }
    const int lpt_chunks[] = {64, 48, 32, 16};
    for (int chunk : lpt_chunks) {
        if (words % (uint32_t)chunk != 0u || words / (uint32_t)chunk <= 1u) {
            continue;
        }
        char kname[64];
        char label[32];
        snprintf(kname, sizeof(kname), "ecm_mp_add_mod_fused_lpt%d", chunk);
        snprintf(label, sizeof(label), "fused_lpt%d", chunk);
        k.push_back(kspec(kname, label, EcmAddSubTier::Lpt, true, false, true, chunk, "fused_unroll"));
    }
    if (!(kAndroidBenchLite && words >= 32u)) {
        k.push_back(kspec("ecm_mp_add_mod_fused_unroll", "fused_unroll", EcmAddSubTier::FusedUnrollManual,
                          true, false, false, 0, "fused"));
        k.push_back(kspec("ecm_mp_add_mod_fused_unroll_priv", "fused_unroll_priv",
                          EcmAddSubTier::FusedUnrollManual, true, false, false, 0, "fused_unroll"));
    }
    k.push_back(kspec("ecm_mp_add_mod_fused_unroll_auto", "fused_unroll_auto", EcmAddSubTier::FusedUnrollAuto,
                      true, false, false, 0, "fused_unroll"));
    if (!bench_unroll_only) {
        k.push_back(kspec("ecm_mp_add_mod_fused", "fused", EcmAddSubTier::Fused, true, false, false, 0,
                          "fused_unroll_auto"));
        k.push_back(kspec("ecm_mp_add_mod_mask", "mask", EcmAddSubTier::Basic, true, false, false, 0, "fused"));
        k.push_back(kspec("ecm_mp_add_mod_legacy", "legacy", EcmAddSubTier::Basic, true, false, false, 0,
                          "fused"));
    }
    return k;
}

std::vector<EcmAddSubBenchKernel> opencl_ecm_addsub_sub_kernels(uint32_t words, bool asm_enabled,
                                                              bool asm_b64_enabled) {
    std::vector<EcmAddSubBenchKernel> k;
    k.push_back(kspec("ecm_mp_sub_mod_fused_unroll_auto_hot", "fused_unroll_auto_hot",
                      EcmAddSubTier::FusedUnrollAuto, false, false, false, 0, "fused_unroll_auto", true));
    if (asm_enabled && words == 128u) {
        if (asm_b64_enabled) {
            k.push_back(kspec("ecm_mp_sub_mod_fused_unroll_asm_b64", "fused_unroll_asm_b64",
                              EcmAddSubTier::Asm, false, true, false, 0, "fused_unroll"));
        }
        k.push_back(kspec("ecm_mp_sub_mod_fused_unroll_asm_b32", "fused_unroll_asm_b32", EcmAddSubTier::Asm,
                          false, true, false, 0, "fused_unroll"));
    }
    if (!(kAndroidBenchLite && words >= 32u)) {
        k.push_back(kspec("ecm_mp_sub_mod_fused_unroll", "fused_unroll", EcmAddSubTier::FusedUnrollManual,
                          false, false, false, 0, "ecm_mp_sub_mod"));
        k.push_back(kspec("ecm_mp_sub_mod_fused_unroll_priv", "fused_unroll_priv",
                          EcmAddSubTier::FusedUnrollManual, false, false, false, 0, "fused_unroll"));
    }
    k.push_back(kspec("ecm_mp_sub_mod_fused_unroll_auto", "fused_unroll_auto", EcmAddSubTier::FusedUnrollAuto,
                      false, false, false, 0, "fused_unroll"));
    k.push_back(kspec("ecm_mp_sub_mod", "fused_loop", EcmAddSubTier::Fused, false, false, false, 0,
                      "fused_unroll_auto"));
    return k;
}
