#include "preparation.h"
#include "ecm/stage1/selection.h"
#include "../../io/exponent_cache.h"
#include "ecm/io/save.h"
#include "ecm/platform/logging.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace ecm_stage1 {
bool compute_batch_s(mpz_t s, double B1, unsigned torsion, const std::string& cache_dir){
    if(B1 < 2.0) {
        mpz_set_ui(s, torsion);
        return true;
    }
    const uint64_t limit64 = (uint64_t)std::floor(B1 + 0.0001);
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }
    /* The exponent is cached on disk (validated on load, see ecm_stage1_exp_cache.h): at
       B1 = 260e6 building it costs ~10 s and every task in a worktodo would repeat that. */
    std::string status;
    if (!ecm_build_lcm_exponent_cached(s, limit64, torsion, cache_dir, &status)) {
        return false;
    }
    ecm_ts_fprintf(stdout, "stage1 exponent built: %s (torsion=%u)\n", status.c_str(), torsion);
    return true;
}

uint64_t random_sigma_u64() {
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned)time(nullptr));
        seeded = true;
    }
    uint64_t sigma;
    do {
        uint32_t hi = 0, lo = 0;
        sigma = ((uint64_t)(rand() & 0x1F)) << 48;
        sigma += ((uint64_t)(rand() & 0xFFFF)) << 32;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        const unsigned __int64 tsc = __rdtsc();
        hi = (uint32_t)(tsc >> 32);
        lo = (uint32_t)tsc;
#endif
        sigma += (uint64_t)(lo ^ hi ^ ((uint32_t)rand() << 16));
    } while (sigma <= 5);
    return sigma;
}

bool fixed_sigma_batch_exceeds_prime95_range(uint64_t sigma, uint32_t curves) {
    return curves != 0 &&
           (sigma > static_cast<uint64_t>(INT64_MAX) ||
            static_cast<uint64_t>(curves - 1) > static_cast<uint64_t>(INT64_MAX) - sigma);
}

ecm_save::Naming save_naming(const Stage1Options& opt) {
    ecm_save::Naming naming;
    naming.curve = ecm_save::curve_prefix(opt.execution.selection.curve == CurveModel::Edwards,
        opt.execution.selection.target == ExecutionTarget::Gpu ? opt.execution.selection.gpu_param : 0);
    naming.target_kbnc = opt.identity.target_kbnc;
    naming.target_n = opt.identity.handoff_n;
    naming.mersenne = opt.identity.handoff_k_str == "1" && opt.identity.handoff_b == 2 &&
                      opt.identity.handoff_c == -1;
    return naming;
}

} // namespace ecm_stage1
