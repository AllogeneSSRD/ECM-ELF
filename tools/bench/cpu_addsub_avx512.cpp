// ============================================================================
// cpu_addsub_avx512.cpp — AVX512 fused modular add/sub kernels (K=16).
//
// Compiled in ISOLATION with /arch:AVX512 (MSVC) or -mavx512f -mavx512dq
// (GCC/Clang).  The rest of the benchmark is compiled with an AVX2 baseline
// so the binary runs on AVX2-only CPUs; these kernels are called only after
// detect_cpu_isa() confirms AVX512F + AVX512DQ support at runtime.
//
// Keeping the AVX512 instruction-emitting code in a separate TU is essential:
// /arch:AVX512 lets the compiler emit AVX512 instructions ANYWHERE in a TU
// (including scalar loops), which would crash on an AVX2-only CPU before any
// runtime dispatch could occur.
// ============================================================================

#include <cstdint>
#include <immintrin.h>

// unsigned gt, returns __m512i via vpmovm2d
static inline __m512i _mm512_cmpgt_epu32_m512i(__m512i a, __m512i b) {
    __m512i flip = _mm512_set1_epi32(0x80000000u);
    return _mm512_movm_epi32(
        _mm512_cmpgt_epi32_mask(
            _mm512_xor_si512(a, flip), _mm512_xor_si512(b, flip)));
}
// cmpeq → returns __m512i mask
static inline __m512i _mm512_cmpeq_epi32_m512i(__m512i a, __m512i b) {
    return _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(a, b));
}

void cpu_add_fused_avx512_soa(uint32_t *r, const uint32_t *a,
                              const uint32_t *b, const uint32_t *N,
                              uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vzero = _mm512_setzero_si512();
    __m512i vcarry_add = vzero;
    __m512i vcarry_sub = vones;

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i vnotN = _mm512_xor_si512(vN, vones);

        // Add chain
        __m512i vsum0 = _mm512_add_epi32(va, vb);
        __m512i vovf_ab = _mm512_cmpgt_epu32_m512i(va, vsum0);
        __m512i vsum = _mm512_sub_epi32(vsum0, vcarry_add);
        __m512i vovf_carry = _mm512_and_si512(vcarry_add, _mm512_cmpeq_epi32_m512i(vsum0, vones));
        vcarry_add = _mm512_or_si512(vovf_ab, vovf_carry);

        // Sub chain
        __m512i vtmp0 = _mm512_add_epi32(vsum, vnotN);
        __m512i vovf_sub0 = _mm512_cmpgt_epu32_m512i(vsum, vtmp0);
        __m512i vtmp = _mm512_sub_epi32(vtmp0, vcarry_sub);
        __m512i vovf_carry_sub = _mm512_and_si512(vcarry_sub, _mm512_cmpeq_epi32_m512i(vtmp0, vones));
        vcarry_sub = _mm512_or_si512(vovf_sub0, vovf_carry_sub);

        _mm512_storeu_si512((void*)(r + i * 16), vtmp);
    }

    // Conditional add-N (rare, scalar pass)
    __m512i vmask = _mm512_or_si512(vcarry_add, vcarry_sub);
    alignas(64) uint32_t need[16];
    _mm512_store_si512((void*)need, _mm512_xor_si512(vmask, vones));
    uint32_t carry[16] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 16;
        uint32_t ni = N[i];
        for (int j = 0; j < 16; ++j) {
            if (need[j]) {
                uint64_t s = (uint64_t)rp[j] + (uint64_t)ni + (uint64_t)carry[j];
                rp[j] = (uint32_t)s;
                carry[j] = (uint32_t)(s >> 32);
            }
        }
    }
}

// mask-register variant: carries tracked in __mmask16 (no movm_epi32)
void cpu_add_fused_avx512_soa_mask(uint32_t *r, const uint32_t *a,
                                   const uint32_t *b, const uint32_t *N,
                                   uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __mmask16 k_ca = 0x0000;
    __mmask16 k_cs = 0xFFFF;

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i vnotN = _mm512_xor_si512(vN, vones);

        __m512i vsum0   = _mm512_add_epi32(va, vb);
        __mmask16 k_ovf = _mm512_cmpgt_epu32_mask(va, vsum0);
        __m512i vsum = _mm512_mask_add_epi32(vsum0, k_ca, vsum0, vones);
        __mmask16 k_allones = _mm512_cmpeq_epi32_mask(vsum0, vones);
        k_ca = k_ovf | (k_ca & k_allones);

        __m512i vtmp0   = _mm512_add_epi32(vsum, vnotN);
        __mmask16 k_ovf_s = _mm512_cmpgt_epu32_mask(vsum, vtmp0);
        __m512i vtmp = _mm512_mask_add_epi32(vtmp0, k_cs, vtmp0, vones);
        __mmask16 k_tmp_max = _mm512_cmpeq_epi32_mask(vtmp0, vones);
        k_cs = k_ovf_s | (k_cs & k_tmp_max);

        _mm512_storeu_si512((void*)(r + i * 16), vtmp);
    }

    __mmask16 k_need = ~(k_ca | k_cs) & 0xFFFF;
    uint32_t carry[16] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 16;
        uint32_t ni = N[i];
        for (int j = 0; j < 16; ++j) {
            if (k_need & (1u << j)) {
                uint64_t s = (uint64_t)rp[j] + (uint64_t)ni + (uint64_t)carry[j];
                rp[j] = (uint32_t)s;
                carry[j] = (uint32_t)(s >> 32);
            }
        }
    }
}

void cpu_sub_fused_avx512_soa(uint32_t *r, const uint32_t *a,
                              const uint32_t *b, const uint32_t *N,
                              uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vborrow = vones;

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vnotB = _mm512_xor_si512(vb, vones);

        __m512i vsum0 = _mm512_add_epi32(va, vnotB);
        __m512i vovf = _mm512_cmpgt_epu32_m512i(va, vsum0);
        __m512i vsum = _mm512_sub_epi32(vsum0, vborrow);
        __m512i vovf_b = _mm512_and_si512(vborrow, _mm512_cmpeq_epi32_m512i(vsum0, vones));
        vborrow = _mm512_or_si512(vovf, vovf_b);

        _mm512_storeu_si512((void*)(r + i * 16), vsum);
    }

    // Conditional sub-N
    alignas(64) uint32_t need[16];
    _mm512_store_si512((void*)need, _mm512_xor_si512(vborrow, vones));
    uint32_t borrow[16] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 16;
        uint32_t ni = N[i];
        for (int j = 0; j < 16; ++j) {
            if (need[j]) {
                uint64_t d = (uint64_t)rp[j] - (uint64_t)ni - (uint64_t)borrow[j];
                borrow[j] = ((int64_t)d < 0) ? 1u : 0u;
                rp[j] = (uint32_t)d;
            }
        }
    }
}

void cpu_sub_fused_avx512_soa_mask(uint32_t *r, const uint32_t *a,
                                   const uint32_t *b, const uint32_t *N,
                                   uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __mmask16 k_br = 0xFFFF;

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vnotB = _mm512_xor_si512(vb, vones);

        __m512i vsum0   = _mm512_add_epi32(va, vnotB);
        __mmask16 k_ovf = _mm512_cmpgt_epu32_mask(va, vsum0);
        __m512i vsum = _mm512_mask_add_epi32(vsum0, k_br, vsum0, vones);
        __mmask16 k_allones = _mm512_cmpeq_epi32_mask(vsum0, vones);
        k_br = k_ovf | (k_br & k_allones);

        _mm512_storeu_si512((void*)(r + i * 16), vsum);
    }

    __mmask16 k_need = ~k_br;
    uint32_t borrow[16] = {0};
    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t *rp = r + i * 16;
        uint32_t ni = N[i];
        for (int j = 0; j < 16; ++j) {
            if (k_need & (1u << j)) {
                uint64_t d = (uint64_t)rp[j] - (uint64_t)ni - (uint64_t)borrow[j];
                borrow[j] = ((int64_t)d < 0) ? 1u : 0u;
                rp[j] = (uint32_t)d;
            }
        }
    }
}

// ── AVX512 soa + SIMD blend (add only) ──────────────────────────────
void cpu_add_fused_avx512_soa_blend(uint32_t *r, const uint32_t *a,
                                    const uint32_t *b, const uint32_t *N,
                                    uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vcarry_add = _mm512_setzero_si512();
    __m512i vcarry_sub = vones;

    size_t n = (size_t)limbs * 16;
    uint32_t *tmp_sum = new uint32_t[n];
    uint32_t *tmp_tmp = new uint32_t[n];

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i vnotN = _mm512_xor_si512(vN, vones);

        __m512i vsum0 = _mm512_add_epi32(va, vb);
        __m512i vovf_ab = _mm512_cmpgt_epu32_m512i(va, vsum0);
        __m512i vsum = _mm512_sub_epi32(vsum0, vcarry_add);
        __m512i vovf_carry = _mm512_and_si512(vcarry_add, _mm512_cmpeq_epi32_m512i(vsum0, vones));
        vcarry_add = _mm512_or_si512(vovf_ab, vovf_carry);

        __m512i vtmp0 = _mm512_add_epi32(vsum, vnotN);
        __m512i vovf_sub0 = _mm512_cmpgt_epu32_m512i(vsum, vtmp0);
        __m512i vtmp = _mm512_sub_epi32(vtmp0, vcarry_sub);
        __m512i vovf_carry_sub = _mm512_and_si512(vcarry_sub, _mm512_cmpeq_epi32_m512i(vtmp0, vones));
        vcarry_sub = _mm512_or_si512(vovf_sub0, vovf_carry_sub);

        _mm512_storeu_si512((void*)(tmp_sum + i * 16), vsum);
        _mm512_storeu_si512((void*)(tmp_tmp + i * 16), vtmp);
    }

    __m512i vmask = _mm512_or_si512(vcarry_add, vcarry_sub);
    alignas(64) uint32_t need[16];
    _mm512_store_si512((void*)need, _mm512_xor_si512(vmask, vones));
    __mmask16 k_need = 0;
    for (int j = 0; j < 16; ++j)
        if (need[j]) k_need |= (1u << j);

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i vs = _mm512_loadu_si512((const void*)(tmp_sum + i * 16));
        __m512i vt = _mm512_loadu_si512((const void*)(tmp_tmp + i * 16));
        __m512i res = _mm512_mask_blend_epi32(k_need, vt, vs);
        _mm512_storeu_si512((void*)(r + i * 16), res);
    }

    delete[] tmp_sum; delete[] tmp_tmp;
}

// ── AVX512 mask + blend (add only) ──────────────────────────────────
void cpu_add_fused_avx512_soa_mask_blend(uint32_t *r, const uint32_t *a,
                                         const uint32_t *b, const uint32_t *N,
                                         uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __mmask16 k_ca = 0x0000;
    __mmask16 k_cs = 0xFFFF;

    size_t n = (size_t)limbs * 16;
    uint32_t *tmp_sum = new uint32_t[n];
    uint32_t *tmp_tmp = new uint32_t[n];

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i vnotN = _mm512_xor_si512(vN, vones);

        __m512i vsum0   = _mm512_add_epi32(va, vb);
        __mmask16 k_ovf = _mm512_cmpgt_epu32_mask(va, vsum0);
        __m512i vsum = _mm512_mask_add_epi32(vsum0, k_ca, vsum0, vones);
        __mmask16 k_allones = _mm512_cmpeq_epi32_mask(vsum0, vones);
        k_ca = k_ovf | (k_ca & k_allones);

        __m512i vtmp0   = _mm512_add_epi32(vsum, vnotN);
        __mmask16 k_ovf_s = _mm512_cmpgt_epu32_mask(vsum, vtmp0);
        __m512i vtmp = _mm512_mask_add_epi32(vtmp0, k_cs, vtmp0, vones);
        __mmask16 k_tmp_max = _mm512_cmpeq_epi32_mask(vtmp0, vones);
        k_cs = k_ovf_s | (k_cs & k_tmp_max);

        _mm512_storeu_si512((void*)(tmp_sum + i * 16), vsum);
        _mm512_storeu_si512((void*)(tmp_tmp + i * 16), vtmp);
    }

    __mmask16 k_need = ~(k_ca | k_cs);
    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i vs = _mm512_loadu_si512((const void*)(tmp_sum + i * 16));
        __m512i vt = _mm512_loadu_si512((const void*)(tmp_tmp + i * 16));
        __m512i res = _mm512_mask_blend_epi32(k_need, vt, vs);
        _mm512_storeu_si512((void*)(r + i * 16), res);
    }

    delete[] tmp_sum; delete[] tmp_tmp;
}

// ── AVX512 sub + SIMD blend (vector-carry) ──────────────────────────
// Compute r0 = a-b (loop 1, vector borrow), r1 = r0-N (loop 2, vector
// borrow chain), then blend-select r1 where correction is needed.
void cpu_sub_fused_avx512_soa_blend(uint32_t *r, const uint32_t *a,
                                    const uint32_t *b, const uint32_t *N,
                                    uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vzero = _mm512_setzero_si512();
    __m512i vborrow = vones;

    size_t n = (size_t)limbs * 16;
    uint32_t *tmp_r0 = new uint32_t[n];
    uint32_t *tmp_r1 = new uint32_t[n];

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vnotB = _mm512_xor_si512(vb, vones);

        __m512i vsum0 = _mm512_add_epi32(va, vnotB);
        __m512i vovf = _mm512_cmpgt_epu32_m512i(va, vsum0);
        __m512i vsum = _mm512_sub_epi32(vsum0, vborrow);
        __m512i vovf_b = _mm512_and_si512(vborrow, _mm512_cmpeq_epi32_m512i(vsum0, vones));
        vborrow = _mm512_or_si512(vovf, vovf_b);

        _mm512_storeu_si512((void*)(tmp_r0 + i * 16), vsum);
    }

    __m512i vbN = vzero;
    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i r0 = _mm512_loadu_si512((const void*)(tmp_r0 + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i t  = _mm512_sub_epi32(r0, vN);
        __m512i b1 = _mm512_cmpgt_epu32_m512i(t, r0);
        __m512i r1 = _mm512_add_epi32(t, vbN);
        __m512i b2 = _mm512_and_si512(vbN, _mm512_cmpeq_epi32_m512i(t, vzero));
        vbN = _mm512_or_si512(b1, b2);
        _mm512_storeu_si512((void*)(tmp_r1 + i * 16), r1);
    }

    // need = borrow==0 → select r1.  Build k_need from the vector mask.
    alignas(64) uint32_t need[16];
    _mm512_store_si512((void*)need, _mm512_xor_si512(vborrow, vones));
    __mmask16 k_need = 0;
    for (int j = 0; j < 16; ++j)
        if (need[j]) k_need |= (1u << j);

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i r0 = _mm512_loadu_si512((const void*)(tmp_r0 + i * 16));
        __m512i r1 = _mm512_loadu_si512((const void*)(tmp_r1 + i * 16));
        _mm512_storeu_si512((void*)(r + i * 16),
            _mm512_mask_blend_epi32(k_need, r0, r1));
    }

    delete[] tmp_r0; delete[] tmp_r1;
}

// ── AVX512 sub + mask carry + SIMD blend ────────────────────────────
// Carry tracked in __mmask16 (loop 1); correction subtract in a vector
// borrow chain (loop 2); selector k_need = ~k_br needs no bit loop.
void cpu_sub_fused_avx512_soa_mask_blend(uint32_t *r, const uint32_t *a,
                                         const uint32_t *b, const uint32_t *N,
                                         uint32_t limbs) {
    __m512i vones = _mm512_set1_epi32(0xFFFFFFFFu);
    __m512i vzero = _mm512_setzero_si512();
    __mmask16 k_br = 0xFFFF;

    size_t n = (size_t)limbs * 16;
    uint32_t *tmp_r0 = new uint32_t[n];
    uint32_t *tmp_r1 = new uint32_t[n];

    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i * 16));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i * 16));
        __m512i vnotB = _mm512_xor_si512(vb, vones);

        __m512i vsum0   = _mm512_add_epi32(va, vnotB);
        __mmask16 k_ovf = _mm512_cmpgt_epu32_mask(va, vsum0);
        __m512i vsum = _mm512_mask_add_epi32(vsum0, k_br, vsum0, vones);
        __mmask16 k_allones = _mm512_cmpeq_epi32_mask(vsum0, vones);
        k_br = k_ovf | (k_br & k_allones);

        _mm512_storeu_si512((void*)(tmp_r0 + i * 16), vsum);
    }

    __m512i vbN = vzero;
    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i r0 = _mm512_loadu_si512((const void*)(tmp_r0 + i * 16));
        __m512i vN = _mm512_set1_epi32(N[i]);
        __m512i t  = _mm512_sub_epi32(r0, vN);
        __m512i b1 = _mm512_cmpgt_epu32_m512i(t, r0);
        __m512i r1 = _mm512_add_epi32(t, vbN);
        __m512i b2 = _mm512_and_si512(vbN, _mm512_cmpeq_epi32_m512i(t, vzero));
        vbN = _mm512_or_si512(b1, b2);
        _mm512_storeu_si512((void*)(tmp_r1 + i * 16), r1);
    }

    __mmask16 k_need = ~k_br;  // borrow==0 → select r1
    for (uint32_t i = 0; i < limbs; ++i) {
        __m512i r0 = _mm512_loadu_si512((const void*)(tmp_r0 + i * 16));
        __m512i r1 = _mm512_loadu_si512((const void*)(tmp_r1 + i * 16));
        _mm512_storeu_si512((void*)(r + i * 16),
            _mm512_mask_blend_epi32(k_need, r0, r1));
    }

    delete[] tmp_r0; delete[] tmp_r1;
}
