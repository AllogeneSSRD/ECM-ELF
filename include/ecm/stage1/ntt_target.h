#pragma once
#include "ecm/stage1/task.h"

namespace ecm_stage1 {

// The task modulus remains the cofactor. Only device arithmetic uses M_p.
inline unsigned ntt_arithmetic_bits(mpz_srcptr target, const Stage1Options& options) {
    if (!target || mpz_cmp_ui(target, 1) <= 0)
        return 0;
    const auto& identity = options.identity;
    const auto bits = mpz_sizeinbase(target, 2);
    if (identity.handoff_factors.empty())
        return bits >= 32 && bits <= 262144 && mpz_scan0(target, 0) == bits ? unsigned(bits) : 0;
    const unsigned p = identity.handoff_n;
    if (!options.execution.ntt.allow_known_factors || !identity.target_kbnc ||
        identity.handoff_k_str != "1" || identity.handoff_b != 2 || identity.handoff_c != -1 ||
        p < 32 || p > 262144)
        return 0;
    mpz_t carrier;
    mpz_init_set_ui(carrier, 0);
    mpz_setbit(carrier, p);
    mpz_sub_ui(carrier, carrier, 1);
    const bool divides = mpz_cmp(target, carrier) < 0 && mpz_divisible_p(carrier, target);
    mpz_clear(carrier);
    return divides ? p : 0;
}

} // namespace ecm_stage1
