#pragma once

// Included after Stage in the CPU NTT probe. The field and the twiddle
// representation are shared with the canonical reference implementation.
namespace cpu_ntt {

template<unsigned lanes>
struct Ops {
    static_assert(lanes == 4 || lanes == 8);
    using V = std::conditional_t<lanes == 4, __m256i, __m512i>;

    static V load(const Word* p) {
        if constexpr (lanes == 4) return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        else return _mm512_loadu_si512(p);
    }
    static void store(Word* p, V v) {
        if constexpr (lanes == 4) _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
        else _mm512_storeu_si512(p, v);
    }
    static V splat(Word x) {
        if constexpr (lanes == 4) return _mm256_set1_epi64x(static_cast<int64_t>(x));
        else return _mm512_set1_epi64(static_cast<int64_t>(x));
    }
    static V plus(V a, V b) {
        if constexpr (lanes == 4) return _mm256_add_epi64(a, b);
        else return _mm512_add_epi64(a, b);
    }
    static V minus(V a, V b) {
        if constexpr (lanes == 4) return _mm256_sub_epi64(a, b);
        else return _mm512_sub_epi64(a, b);
    }
    static V mask_low(V a) {
        if constexpr (lanes == 4) return _mm256_and_si256(a, splat(mask52));
        else return _mm512_and_si512(a, splat(mask52));
    }
    static V carry(V a) {
        if constexpr (lanes == 4) return _mm256_srli_epi64(a, 52);
        else return _mm512_srli_epi64(a, 52);
    }
    static V blend(__mmask8 k, V a, V b) {
        if constexpr (lanes == 4) return _mm256_mask_blend_epi64(k, a, b);
        else return _mm512_mask_blend_epi64(k, a, b);
    }
    static V permute(V a, V index) {
        if constexpr (lanes == 4) return _mm256_permutexvar_epi64(index, a);
        else return _mm512_permutexvar_epi64(index, a);
    }
    static V reduce(V a, Word bound = q) {
        const V p = splat(bound);
        if constexpr (lanes == 4)
            return _mm256_mask_sub_epi64(a, _mm256_cmp_epu64_mask(a, p, _MM_CMPINT_NLT), a, p);
        else
            return _mm512_mask_sub_epi64(a, _mm512_cmp_epu64_mask(a, p, _MM_CMPINT_NLT), a, p);
    }
    static V add(V a, V b) { return reduce(plus(a, b)); }
    static V sub(V a, V b) {
        const V d = minus(a, b);
        if constexpr (lanes == 4)
            return _mm256_mask_add_epi64(d, _mm256_cmp_epu64_mask(a, b, _MM_CMPINT_LT), d, splat(q));
        else
            return _mm512_mask_add_epi64(d, _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT), d, splat(q));
    }
    static V maddlo(V acc, V a, V b) {
        if constexpr (lanes == 4) return _mm256_madd52lo_epu64(acc, a, b);
        else return _mm512_madd52lo_epu64(acc, a, b);
    }
    static V maddhi(V acc, V a, V b) {
        if constexpr (lanes == 4) return _mm256_madd52hi_epu64(acc, a, b);
        else return _mm512_madd52hi_epu64(acc, a, b);
    }

    // Shoup multiplication with R=2^52. For a<R and w<q, floor(a*w'/R)
    // differs from floor(a*w/q) by at most one. The result before the last
    // correction lies in [0,2q), so taking the low 52 bits is exact.
    static V fixed_lazy(V a, V w, V precon) {
        const V hi = maddhi(splat(0), a, precon);
        const V lo = maddlo(splat(0), a, w);
        return mask_low(maddlo(lo, hi, splat(0 - q)));
    }
    static V fixed(V a, V w, V precon) { return reduce(fixed_lazy(a, w, precon)); }
    static V fixed(V a, Multiplier m) { return fixed(a, splat(m.value), splat(m.quotient)); }

    static V mul(V a, V b) {
        const V zero = splat(0);
        const V lo = mask_low(maddlo(zero, a, b));
        const V hi = maddhi(zero, a, b);
        const V m = maddlo(zero, lo, splat(montgomery_inverse()));
        const V correction = maddlo(lo, m, splat(q));
        return reduce(maddhi(plus(hi, carry(correction)), m, splat(q)));
    }
};

template<unsigned lanes, bool inverse, bool lazy, bool unit = false>
void butterfly(typename Ops<lanes>::V& a, typename Ops<lanes>::V& b,
               typename Ops<lanes>::V w, typename Ops<lanes>::V precon) {
    using O = Ops<lanes>;
    using V = typename O::V;
    if constexpr (!lazy && inverse) {
        const V v = unit ? b : O::fixed(b, w, precon);
        b = O::sub(a, v);
        a = O::add(a, v);
    } else if constexpr (!lazy) {
        const V d = O::sub(a, b);
        a = O::add(a, b);
        b = unit ? d : O::fixed(d, w, precon);
    } else if constexpr (inverse) {
        // DIT invariant: inputs and outputs <4q. Reduce just the left
        // input to <2q; the twiddle multiplication does that for the right.
        const V u = O::reduce(a, 2 * q);
        const V v = unit ? O::reduce(b, 2 * q) : O::fixed_lazy(b, w, precon);
        a = O::plus(u, v);
        b = O::minus(O::plus(u, O::splat(2 * q)), v);
    } else {
        // DIF invariant: inputs and outputs <2q. The twiddle input is <4q.
        const V d = O::minus(O::plus(a, O::splat(2 * q)), b);
        a = O::reduce(O::plus(a, b), 2 * q);
        b = unit ? O::reduce(d, 2 * q) : O::fixed_lazy(d, w, precon);
    }
}

template<unsigned lanes, bool inverse, bool lazy, bool unit = false>
typename Ops<lanes>::V short_stage(typename Ops<lanes>::V a, typename Ops<lanes>::V index,
                                  __mmask8 lower, typename Ops<lanes>::V w,
                                  typename Ops<lanes>::V precon) {
    using O = Ops<lanes>;
    using V = typename O::V;
    if constexpr (inverse) {
        const V u = [&] {
            if constexpr (lazy) {
                const V lower_values = O::reduce(a, 2 * q);
                if constexpr (unit) return lower_values;
                else return O::blend(lower, O::fixed_lazy(a, w, precon), lower_values);
            } else if constexpr (unit) {
                return a;
            } else {
                return O::blend(lower, O::fixed(a, w, precon), a);
            }
        }();
        const V v = O::permute(u, index);
        if constexpr (lazy)
            return O::blend(lower, O::minus(O::plus(v, O::splat(2 * q)), u), O::plus(u, v));
        else
            return O::blend(lower, O::sub(v, u), O::add(u, v));
    } else {
        const V b = O::permute(a, index);
        if constexpr (lazy) {
            const V d = O::minus(O::plus(b, O::splat(2 * q)), a);
            const V upper = unit ? O::reduce(d, 2 * q) : O::fixed_lazy(d, w, precon);
            return O::blend(lower, upper, O::reduce(O::plus(a, b), 2 * q));
        } else {
            const V d = O::sub(b, a);
            const V upper = unit ? d : O::fixed(d, w, precon);
            return O::blend(lower, upper, O::add(a, b));
        }
    }
}

template<unsigned lanes, bool inverse, bool lazy>
void apply_stage(Word* row, unsigned n, const Stage& s) {
    using O = Ops<lanes>;
    const unsigned half = s.half;
    if (half < lanes) {
        const auto index = O::load(s.peer.data());
        const auto w = O::load(s.weights.data()), wp = O::load(s.quotients.data());
        for (unsigned i = 0; i < n; i += lanes)
            O::store(row + i, short_stage<lanes, inverse, lazy>(O::load(row + i), index,
                                                              s.lower, w, wp));
        return;
    }
    for (unsigned base = 0; base < n; base += 2 * half) {
        for (unsigned j = 0; j < half; j += lanes) {
            auto* x = row + base + j;
            auto* y = x + half;
            auto a = O::load(x), b = O::load(y);
            butterfly<lanes, inverse, lazy>(a, b, O::load(s.weights.data() + j),
                                            O::load(s.quotients.data() + j));
            O::store(x, a);
            O::store(y, b);
        }
    }
}

template<unsigned lanes, bool inverse, bool lazy>
void fused_short(Word* row, unsigned n, const Stage& s1, const Stage& s2, const Stage& s4) {
    using O = Ops<lanes>;
    const auto index1 = O::load(s1.peer.data()), index2 = O::load(s2.peer.data());
    const auto w2 = O::load(s2.weights.data()), wp2 = O::load(s2.quotients.data());
    const auto w4 = O::load(s4.weights.data()), wp4 = O::load(s4.quotients.data());
    const auto unit = O::splat(1), unused = O::splat(0);
    const auto layer1 = [&](typename O::V a) {
        return short_stage<lanes, inverse, lazy, true>(a, index1, s1.lower, unit, unused);
    };
    const auto layer2 = [&](typename O::V a) {
        return short_stage<lanes, inverse, lazy>(a, index2, s2.lower, w2, wp2);
    };

    if constexpr (lanes == 8) {
        const auto index4 = O::load(s4.peer.data());
        const auto layer4 = [&](typename O::V a) {
            return short_stage<lanes, inverse, lazy>(a, index4, s4.lower, w4, wp4);
        };
        for (unsigned i = 0; i < n; i += 8) {
            auto a = O::load(row + i);
            if constexpr (inverse) a = layer4(layer2(layer1(a)));
            else a = layer1(layer2(layer4(a)));
            O::store(row + i, a);
        }
    } else {
        for (unsigned i = 0; i < n; i += 8) {
            auto a = O::load(row + i), b = O::load(row + i + 4);
            if constexpr (inverse) {
                a = layer2(layer1(a));
                b = layer2(layer1(b));
                butterfly<lanes, inverse, lazy>(a, b, w4, wp4);
            } else {
                butterfly<lanes, inverse, lazy>(a, b, w4, wp4);
                a = layer1(layer2(a));
                b = layer1(layer2(b));
            }
            O::store(row + i, a);
            O::store(row + i + 4, b);
        }
    }
}

template<unsigned lanes, bool inverse, bool lazy, bool fused>
void transform_row(Word* row, unsigned n, const std::vector<Stage>& stages) {
    static_assert(4 * q < (Word{1} << 52));
    if constexpr (fused && inverse) {
        fused_short<lanes, inverse, lazy>(row, n, stages[0], stages[1], stages[2]);
        for (size_t i = 3; i < stages.size(); ++i)
            apply_stage<lanes, inverse, lazy>(row, n, stages[i]);
    } else if constexpr (fused) {
        const size_t short_start = stages.size() - 3;
        for (size_t i = 0; i < short_start; ++i)
            apply_stage<lanes, inverse, lazy>(row, n, stages[i]);
        fused_short<lanes, inverse, lazy>(row, n, stages[short_start + 2],
                                         stages[short_start + 1], stages[short_start]);
    } else {
        for (const auto& stage : stages)
            apply_stage<lanes, inverse, lazy>(row, n, stage);
    }

    // Keep the radix-5, pointwise multiply, and unpack phases unchanged.
    if constexpr (lazy) {
        using O = Ops<lanes>;
        for (unsigned i = 0; i < n; i += lanes) {
            auto a = O::load(row + i);
            if constexpr (inverse) a = O::reduce(a, 2 * q);
            O::store(row + i, O::reduce(a));
        }
    }
}

} // namespace cpu_ntt
