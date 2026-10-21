#include "host_policy.h"
#include "../../runtime/cuda_ecm_shim.h"
#include "../../../../storage/cache/ecm_stage1_exp_cache.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace ecm_cuda_mont_host {

MontRequestedSpec capture_mont_request(const char *algorithm) {
    MontRequestedSpec request;
    request.default_plan_cache=ecm_exp_cache_get_dir();
    for(const auto *key:{"ECM_GPU_STAGE1_ALGO","ECM_PRAC_WINDOW","ECM_PRAC_VARIANT",
        "ECM_PRAC_REG_TARGET","ECM_STAGE1_TPI","ECM_PRAC_CONSTANTS","ECM_PRAC_PLAN_CACHE",
        "ECM_PRAC_TARGET_MS","ECM_GPU_STAGE1_SAMPLE_SECONDS","ECM_GPU_DUMP",
        "ECM_PRAC_WINDOW_COUNT","ECM_PRAC_WINDOW_CHUNK","ECM_PRAC_WINDOW_WARMUP","ECM_PRAC_WINDOW_DUMP"})
        if(const auto *value=algorithm && std::strcmp(key,"ECM_GPU_STAGE1_ALGO")==0?algorithm:std::getenv(key))
            request.values.emplace(key,value);
    return request;
}

const char *stage1_algorithm(const MontRequestedSpec &request) {
  const char *algorithm = request.get("ECM_GPU_STAGE1_ALGO");
  const char *window = request.get("ECM_PRAC_WINDOW");
  const char *variant = request.get("ECM_PRAC_VARIANT");
  if (((window && *window) || (variant && *variant && strcmp(variant, "baseline") != 0)) &&
      (!algorithm || strcmp(algorithm, "prac") != 0)) {
    outputf(OUTPUT_ERROR, "GPU: PRAC window/variant requires ECM_GPU_STAGE1_ALGO=prac\n");
    return nullptr;
  }
  if (algorithm && *algorithm && strcmp(algorithm, "ladder") != 0) {
    if (strcmp(algorithm, "prac") != 0 && strcmp(algorithm, "resident") != 0) {
      outputf(OUTPUT_ERROR, "GPU: ECM_GPU_STAGE1_ALGO must be ladder, resident or prac\n");
      return nullptr;
    }
  }
  return algorithm && *algorithm ? algorithm : "ladder";
}

PracPolicy read_prac_policy(const MontRequestedSpec &request,bool prac) {
        int mode = prac ? ECM_DOMAIN_PRAC : ECM_DOMAIN_LADDER;
        if (prac) if (const char *cap = request.get("ECM_PRAC_REG_TARGET")) {
            if (strcmp(cap, "255") == 0) mode = ECM_DOMAIN_PRAC_NATURAL;
            else if (strcmp(cap, "168") == 0) mode = ECM_DOMAIN_PRAC_168;
            else if (strcmp(cap, "128") == 0) mode = ECM_DOMAIN_PRAC_SINGLE_COMPACT_128;
            else if (*cap && strcmp(cap, "0") != 0) throw std::runtime_error("ECM_PRAC_REG_TARGET must be 0, 128, 168 or 255");
        }
        bool compact = false, outlined = false, single_add = false, single_compact = false;
        if (const char *variant = request.get("ECM_PRAC_VARIANT")) {
            if (strcmp(variant, "compact") == 0) compact = true;
            else if (strcmp(variant, "outline-add") == 0) outlined = true;
            else if (strcmp(variant, "single-add") == 0) single_add = true;
            else if (strcmp(variant, "single-compact") == 0) single_add = single_compact = true;
            else if (*variant && strcmp(variant, "baseline") != 0)
                throw std::runtime_error("ECM_PRAC_VARIANT must be baseline, compact, outline-add, single-add or single-compact");
        }
        if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_128 && !single_compact)
            throw std::runtime_error("128-register policy requires single-compact variant");
        if (single_add) {
            if (!prac) throw std::runtime_error("single-add variant requires PRAC");
            if (mode == ECM_DOMAIN_PRAC_NATURAL) mode = single_compact ? ECM_DOMAIN_PRAC_SINGLE_COMPACT : ECM_DOMAIN_PRAC_SINGLE_ADD;
            else if (mode == ECM_DOMAIN_PRAC_168) mode = single_compact ? ECM_DOMAIN_PRAC_SINGLE_COMPACT_168 : ECM_DOMAIN_PRAC_SINGLE_ADD_168;
            else if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_128 && single_compact) { /* Already selected. */ }
            else throw std::runtime_error("single-add variant requires register policy 255 or 168");
        }
        if (outlined) {
            if (!prac) throw std::runtime_error("outline-add variant requires PRAC");
            if (mode == ECM_DOMAIN_PRAC_NATURAL) mode = ECM_DOMAIN_PRAC_OUTLINE_ADD;
            else if (mode == ECM_DOMAIN_PRAC_168) mode = ECM_DOMAIN_PRAC_OUTLINE_ADD_168;
            else throw std::runtime_error("outline-add variant requires register policy 255 or 168");
        }
        if (compact) {
            if (!prac) throw std::runtime_error("compact variant requires PRAC");
            if (mode == ECM_DOMAIN_PRAC_NATURAL) mode = ECM_DOMAIN_PRAC_COMPACT;
            else if (mode == ECM_DOMAIN_PRAC_168) mode = ECM_DOMAIN_PRAC_COMPACT_168;
            else throw std::runtime_error("compact variant requires register policy 255 or 168");
        }
        uint32_t requested_tpi = 0;
        if (const char *env = request.get("ECM_STAGE1_TPI")) {
            if (!*env || strcmp(env, "0") == 0) requested_tpi = 0;
            else if (strcmp(env, "16") == 0) requested_tpi = 16;
            else if (strcmp(env, "32") == 0) requested_tpi = 32;
            else throw std::runtime_error("ECM_STAGE1_TPI must be 0, 16 or 32");
        }
        return {mode,compact,outlined,single_add,single_compact,requested_tpi};
}

const char *prac_constants_policy(const MontRequestedSpec &request,bool prac,int &mode,uint32_t bits,uint32_t requested_tpi,mpz_srcptr N,size_t nbits) {
        const char *constants = "none";
        if (const char *env = request.get("ECM_PRAC_CONSTANTS")) {
            if (!*env || strcmp(env, "none") == 0) { /* Existing policy. */ }
            else if (strcmp(env, "runtime") == 0 || strcmp(env, "np0") == 0 || strcmp(env, "m4423") == 0)
                constants = env;
            else throw std::runtime_error("ECM_PRAC_CONSTANTS must be none, runtime, np0 or m4423");
        }
        if (strcmp(constants, "none") != 0) {
            if (!prac || mode != ECM_DOMAIN_PRAC_SINGLE_COMPACT_168)
                throw std::runtime_error("PRAC constants require single-compact variant and register policy 168");
            if (bits != 4608 || requested_tpi == 32)
                throw std::runtime_error("PRAC constants require the selected 4608/TPI16 tier");
            if (strcmp(constants, "m4423") == 0) {
                mpz_t next; mpz_init(next); mpz_add_ui(next, N, 1);
                const bool exact = nbits == 4423 && mpz_popcount(next) == 1;
                mpz_clear(next);
                if (!exact) throw std::runtime_error("PRAC m4423 constants require N exactly 2^4423-1");
                mode = ECM_DOMAIN_PRAC_CONSTANT_M4423;
            } else if (strcmp(constants, "np0") == 0) {
                if (uint32_t(mpz_get_ui(N)) != 0xffffffffu)
                    throw std::runtime_error("PRAC np0 constants require N mod 2^32 = 0xffffffff");
                mode = ECM_DOMAIN_PRAC_CONSTANT_NP0;
            } else mode = ECM_DOMAIN_PRAC_CONSTANT_RUNTIME;
        }
        return constants;
}

std::string prac_plan_cache(const MontRequestedSpec &request) {
            std::string cache = request.default_plan_cache;
            if (const char *env = request.get("ECM_PRAC_PLAN_CACHE")) cache = env;
            return cache;
}

double prac_target_ms(const MontRequestedSpec &request,bool prac) {
        double target_ms = 100;
        if (prac) {
            if (const char *value = request.get("ECM_PRAC_TARGET_MS")) {
                char *end = nullptr;
                target_ms = std::strtod(value, &end);
                if (end == value || *end || !std::isfinite(target_ms) || target_ms < 10 || target_ms > 500)
                    throw std::runtime_error("ECM_PRAC_TARGET_MS must be finite and in [10,500]");
            }
            outputf(OUTPUT_ALWAYS, "GPU: PRAC slice target=%.3f ms, adaptive bounds=%.3f..%.3f ms\n",
                target_ms, target_ms * 0.8, target_ms * 1.2);
        }
        return target_ms;
}

double prac_sample_limit(const MontRequestedSpec &request) {
        double sample_limit = 0;
        if (const char *env = request.get("ECM_GPU_STAGE1_SAMPLE_SECONDS")) {
            sample_limit = atof(env);
            if (!(sample_limit >= 0 && sample_limit < 1e9)) throw std::runtime_error("invalid sample limit");
        }
        return sample_limit;
}

bool gpu_dump_enabled(const MontRequestedSpec &request) {
  const char *dump_env = request.get("ECM_GPU_DUMP");
  return (dump_env != NULL && atoi(dump_env) != 0);
}

static uint64_t prac_window_integer(const MontRequestedSpec &request,const char *key, uint64_t fallback) {
    const char *value = request.get(key);
    if (!value || !*value) return fallback;
    uint64_t result = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || result > (UINT64_MAX - uint64_t(*p - '0')) / 10)
            throw std::runtime_error(std::string("invalid ") + key);
        result = result * 10 + uint64_t(*p - '0');
    }
    return result;
}
PracWindow prac_window_settings(const MontRequestedSpec &request,bool prac, uint64_t records) {
    PracWindow w;
    const char *position = request.get("ECM_PRAC_WINDOW");
    if (!position || !*position) return w;
    if (!prac) throw std::runtime_error("ECM_PRAC_WINDOW requires PRAC");
    w.position = position;
    if (w.position != "prefix" && w.position != "middle" && w.position != "tail")
        throw std::runtime_error("ECM_PRAC_WINDOW must be prefix, middle or tail");
    uint64_t count = prac_window_integer(request,"ECM_PRAC_WINDOW_COUNT", 16);
    if (!count || count > 32 || !records) throw std::runtime_error("PRAC window count must be 1..32");
    w.count = std::min(count, records);
    w.chunk = prac_window_integer(request,"ECM_PRAC_WINDOW_CHUNK", 0);
    if (!w.chunk) w.chunk = w.count;
    if (w.chunk > w.count) throw std::runtime_error("PRAC window chunk must not exceed selected count");
    if (w.position == "middle") w.first = (records - w.count) / 2;
    else if (w.position == "tail") w.first = records - w.count;
    uint64_t warmup = prac_window_integer(request,"ECM_PRAC_WINDOW_WARMUP", 2);
    if (warmup > 32) throw std::runtime_error("PRAC window warmup must be 0..32");
    w.warmup = uint32_t(warmup);
    if (const char *seconds = request.get("ECM_GPU_STAGE1_SAMPLE_SECONDS")) {
        char *end = nullptr;
        w.seconds = std::strtod(seconds, &end);
        if (end == seconds || *end || !std::isfinite(w.seconds) || w.seconds <= 0 || w.seconds > 600)
            throw std::runtime_error("PRAC window sample seconds must be finite and in (0,600]");
    }
    uint64_t dump = prac_window_integer(request,"ECM_PRAC_WINDOW_DUMP", 0);
    if (dump > 1) throw std::runtime_error("PRAC window dump must be 0 or 1");
    w.dump = dump != 0;
    return w;
}



}
