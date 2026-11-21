#pragma once

#include "ecm/integer.h"
#include "../../curve/edwards.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace ecm_edwards::ntt {

inline bool prefer_device_dictionary(size_t p, unsigned curves, unsigned window, unsigned min_bits) {
    if (min_bits) return p >= min_bits;
    if (p >= 32768 || (p >= 20000 && window >= 6)) return true;
    if (window < 8) return false;
    if (p >= 2000 && curves >= 64) return true;
    if (p >= 6144) return curves >= 8;
    return p >= 3500 && curves >= (window >= 10 ? 16u : 32u);
}

// For the calibration point (1,0), odd multiples alternate between P and -P.
// Their four affine slots are (x,0,0,x) in canonical limb representation.
inline void order_four_calibration_dictionary(std::vector<uint32_t>& table, unsigned words, mpz_srcptr n) {
    Integer minus_one;
    mpz_sub_ui(minus_one.v,n,1);
    std::vector<uint32_t> negative(words);
    mpz_export(negative.data(),nullptr,-1,sizeof(uint32_t),0,0,minus_one.v);
    std::fill(table.begin(),table.end(),0u);
    for (size_t offset = 0, entry = 0; offset < table.size(); offset += 4ull * words, ++entry) {
        auto* x = table.data() + offset;
        if (entry % 2) std::copy(negative.begin(),negative.end(),x);
        else x[0] = 1;
        std::copy_n(x,words,x + 3ull * words);
    }
}

// Entries are packed entry / coordinate / curve / limb, as read by the
// shared NTT packer. One inverse normalizes all odd multiples of a curve.
inline Setup affine_dictionary(std::vector<uint32_t>& table, unsigned curves, unsigned lane,
                               unsigned words, unsigned window, mpz_srcptr px, mpz_srcptr py,
                               mpz_srcptr d, mpz_srcptr n, mpz_ptr factor) {
    const unsigned entries = 1u << (window - 2);
    Array points(4 * entries), prefix(entries + 1);
    std::array<Integer, 4> twice;
    Integer a, b, c, e, f, g, h, inverse_z, iz, sum_twice;
    const auto mul = [n](mpz_ptr r, mpz_srcptr x, mpz_srcptr y) {
        mpz_mul(r, x, y);
        mpz_mod(r, r, n);
    };
    mpz_set(points[0], px);
    mpz_set(points[1], py);
    mpz_set_ui(points[2], 1);
    mul(points[3], px, py);
    mul(a.v, px, px);
    mul(b.v, py, py);
    mpz_mul_ui(e.v, points[3], 2);
    mpz_add(g.v, a.v, b.v);
    mpz_sub_ui(f.v, g.v, 2);
    mpz_sub(h.v, a.v, b.v);
    mul(twice[0].v, e.v, f.v);
    mul(twice[1].v, g.v, h.v);
    mul(twice[2].v, f.v, g.v);
    mul(twice[3].v, e.v, h.v);
    mul(twice[3].v, twice[3].v, d);
    mpz_add(sum_twice.v, twice[0].v, twice[1].v);

    mpz_set_ui(prefix[0], 1);
    for (unsigned i = 0; i < entries; ++i) {
        const unsigned j = 4 * i;
        mul(prefix[i + 1], prefix[i], points[j + 2]);
        if (i + 1 == entries) break;
        mul(a.v, points[j], twice[0].v);
        mul(b.v, points[j + 1], twice[1].v);
        mul(c.v, points[j + 3], twice[3].v);
        mpz_add(e.v, points[j], points[j + 1]);
        mul(e.v, e.v, sum_twice.v);
        mpz_sub(e.v, e.v, a.v);
        mpz_sub(e.v, e.v, b.v);
        mul(g.v, points[j + 2], twice[2].v);
        mpz_sub(f.v, g.v, c.v);
        mpz_add(g.v, g.v, c.v);
        mpz_sub(h.v, b.v, a.v);
        mul(points[j + 4], e.v, f.v);
        mul(points[j + 5], g.v, h.v);
        mul(points[j + 6], f.v, g.v);
        mul(points[j + 7], e.v, h.v);
    }
    const auto rc = inverse(inverse_z.v, prefix[entries], n, factor);
    if (rc != Setup::Ready) {
        if (rc == Setup::Factor) return rc;
        for (unsigned i = 0; i < entries; ++i) {
            mpz_gcd(factor, points[4*i + 2], n);
            if (mpz_cmp_ui(factor, 1) > 0 && mpz_cmp(factor, n) < 0)
                return Setup::Factor;
        }
        return Setup::Degenerate;
    }
    for (unsigned i = entries; i-- > 0;) {
        const unsigned j = 4 * i;
        mul(iz.v, prefix[i], inverse_z.v);
        mul(inverse_z.v, inverse_z.v, points[j + 2]);
        mul(a.v, points[j], iz.v);
        mul(b.v, points[j + 1], iz.v);
        mul(c.v, a.v, b.v);
        mul(c.v, c.v, d);
        mpz_add(e.v, a.v, b.v);
        mpz_mod(e.v, e.v, n);
        const mpz_srcptr coordinates[] = {a.v, b.v, c.v, e.v};
        for (unsigned slot = 0; slot < 4; ++slot) {
            auto* dest = table.data() + ((size_t(j + slot) * curves + lane) * words);
            std::fill_n(dest, words, 0u);
            mpz_export(dest, nullptr, -1, sizeof(uint32_t), 0, 0, coordinates[slot]);
        }
    }
    return Setup::Ready;
}
} // namespace ecm_edwards::ntt
