#include "opencl_ecm_montsqr_manifest.h"

#include <cstdio>
#include <string>

namespace {

#if defined(__ANDROID__)
constexpr bool kAndroidBenchLite = true;
#else
constexpr bool kAndroidBenchLite = false;
#endif

void push(std::vector<std::string>& paths, const char* rel) {
    paths.emplace_back(rel);
}

EcmMontSqrBenchKernel kspec(
        std::string name,
        std::string label,
        bool is_mul,
        MontDispatch dispatch,
        uint32_t required_words = 0,
        uint32_t mt_local = 0) {
    return {std::move(name), std::move(label), is_mul, dispatch, required_words, mt_local};
}

void push_512_kernels(std::vector<EcmMontSqrBenchKernel>& out, bool is_mul) {
    const char* op = is_mul ? "mul" : "sqr";
    auto add = [&](const char* suffix, const char* disp, MontDispatch d, uint32_t req = 16u,
                   uint32_t mt = 0) {
        char kname[96];
        std::snprintf(kname, sizeof(kname), "ecm_mont_%s_priv_%s_bench", op, suffix);
        out.push_back(kspec(kname, disp, is_mul, d, req, mt));
    };
    if (is_mul) {
        add("unroll_only_512", "mont_mul_priv_unroll_only_512", MontDispatch::PrivUnroll);
        add("unroll_only_512_manual", "mont_mul_priv_unroll_only_512_manual", MontDispatch::PrivUnroll);
    } else {
        add("unroll_only_512", "mont_sqr_priv_unroll_only_512", MontDispatch::PrivUnroll);
    }
    add("fips512", is_mul ? "mont_mul_priv_fips512" : "mont_sqr_priv_fips512", MontDispatch::PrivUnroll);
    // Disabled @512: fips512_mt* paths underperform on mobile; omit from bench/build.
    // add("fips512_mt4", ...);
    // add("fips512_mt8", ...);
    // add("fips512_mt16", ...);
    // add("fips512_mt8_cs", ...);
    // add("fips512_mt16_cs", ...);
    add("local_only_512", is_mul ? "mont_mul_priv_local_only_512" : "mont_sqr_priv_local_only_512",
        MontDispatch::PrivLocal512);
    if (is_mul) {
        add("opt2_512_local", "mont_mul_priv_opt2_512_local", MontDispatch::PrivOpt2Local512);
    } else {
        add("opt2_512_local", "mont_sqr_priv_opt2_512_local", MontDispatch::PrivOpt2Local512);
    }
}

} // namespace

EcmMontSqrBuildManifest opencl_ecm_montsqr_build_manifest(uint32_t words, bool use_wg) {
    EcmMontSqrBuildManifest m;
    const bool include_wg = use_wg && words != 16u;
    if (include_wg) {
        push(m.source_paths, "mont_wg.cl");
    }
    push(m.source_paths, "mont_priv.cl");
    push(m.source_paths, "mont_priv_opt.cl");
    if (words == 16u) {
        push(m.source_paths, "mont_mul_unroll_only_512_manual_generated.cl");
    }
    push(m.source_paths, "mont_priv_bench.cl");
    push(m.source_paths, "mont_priv_opt_bench.cl");
    if (include_wg) {
        push(m.source_paths, "mont_wg_bench.cl");
    }
    return m;
}

std::vector<EcmMontSqrBenchKernel> opencl_ecm_montsqr_mul_kernels(uint32_t words, bool use_wg) {
    std::vector<EcmMontSqrBenchKernel> k;
    k.push_back(kspec("ecm_mont_mul_priv_bench", "mont_mul_priv", true, MontDispatch::PrivLegacy));
    k.push_back(kspec("ecm_mont_mul_priv_opt_bench", "mont_mul_priv_opt", true, MontDispatch::PrivOpt));
    if (words == 16u) {
        push_512_kernels(k, true);
    }
    k.push_back(kspec("ecm_mont_mul_priv_unroll32_bench", "mont_mul_priv_unroll32", true,
                      MontDispatch::PrivUnroll, words));
    if (!(kAndroidBenchLite && words >= 32u)) {
        k.push_back(kspec("ecm_mont_mul_priv_unroll64_bench", "mont_mul_priv_unroll64", true,
                          MontDispatch::PrivUnroll, words));
    }
    if (use_wg && words != 16u) {
        k.push_back(kspec("cgbn_mont_mul_wg_bench", "mont_mul_wg", true, MontDispatch::Wg));
    }
    return k;
}

std::vector<EcmMontSqrBenchKernel> opencl_ecm_montsqr_sqr_kernels(uint32_t words, bool use_wg) {
    std::vector<EcmMontSqrBenchKernel> k;
    k.push_back(kspec("ecm_mont_sqr_priv_bench", "mont_sqr_priv", false, MontDispatch::PrivLegacy));
    k.push_back(kspec("ecm_mont_sqr_priv_opt_bench", "mont_sqr_priv_opt", false, MontDispatch::PrivOpt));
    if (words == 16u) {
        push_512_kernels(k, false);
    }
    k.push_back(kspec("ecm_mont_sqr_priv_unroll32_bench", "mont_sqr_priv_unroll32", false,
                      MontDispatch::PrivUnroll, words));
    if (!(kAndroidBenchLite && words >= 32u)) {
        k.push_back(kspec("ecm_mont_sqr_priv_unroll64_bench", "mont_sqr_priv_unroll64", false,
                          MontDispatch::PrivUnroll, words));
    }
    if (use_wg && words != 16u) {
        k.push_back(kspec("cgbn_mont_sqr_wg_bench", "mont_sqr_wg", false, MontDispatch::Wg));
    }
    return k;
}
