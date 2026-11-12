#pragma once

namespace ecm_stage1 {

inline constexpr unsigned stage1_ntt_max_curves = 128;
inline constexpr unsigned stage1_ntt_mixed_limit = 32768;
inline constexpr unsigned stage1_ntt_mixed_min = 2560;

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
