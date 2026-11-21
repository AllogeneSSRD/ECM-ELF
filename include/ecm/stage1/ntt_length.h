#pragma once

namespace ecm_stage1 {

inline constexpr unsigned stage1_ntt_max_curves = 512;
inline constexpr unsigned ntt_default_repeats = 2, ntt_max_repeats = 20;
inline constexpr unsigned naf_tune_span = 64, naf_tune_strata = 8;
inline constexpr unsigned stage1_ntt_min_exponent = 32;
inline constexpr unsigned stage1_ntt_max_exponent = 5242880;
inline constexpr unsigned stage1_ntt_max_length = 524288;
inline constexpr unsigned stage1_ntt_mixed_limit = 163840;
inline constexpr unsigned stage1_ntt_mixed_min = 2560;

struct TuneRange {
    unsigned first = 0, last = 0;
    bool contains(unsigned n) const { return !first || (first <= n && n <= last); }
};

struct NttTuneGrid {
    TuneRange exp, length, naf;
    TuneRange batch{1, stage1_ntt_max_curves};
    bool batch_explicit = false;
    bool find95 = false;
    unsigned group_ms = 100;
    unsigned coarse_repeats = ntt_default_repeats, fine_repeats = ntt_default_repeats;
};

constexpr bool stage1_ntt_power_of_two(unsigned n) {
    return n && !(n & (n - 1));
}

constexpr bool stage1_ntt_radix5(unsigned n) {
    return n >= 5 && n <= stage1_ntt_mixed_limit && n % 5 == 0 &&
           stage1_ntt_power_of_two(n / 5);
}

// Older tune rows still use the full radix-5 shape definition.
constexpr bool stage1_ntt_supported_radix5(unsigned n) {
    return n >= stage1_ntt_mixed_min && stage1_ntt_radix5(n);
}

constexpr unsigned stage1_ntt_radix2_length(unsigned n) {
    return stage1_ntt_radix5(n) ? n / 5 : n;
}

constexpr unsigned stage1_ntt_length_log(unsigned n) {
    unsigned k = 0;
    for (; n > 1; n >>= 1) ++k;
    return k;
}

} // namespace ecm_stage1
