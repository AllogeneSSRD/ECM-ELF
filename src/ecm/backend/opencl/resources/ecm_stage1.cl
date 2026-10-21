// ECM stage1 ladder — operator-free; implementations bound via ECM_STAGE1_*_IMPL macros.

static inline void double_add_v2(uint *q, uint *u, uint *w, uint *v, uint d, const uint *N,
                                 uint np0, uint limbs) {
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    add_mod(t, v, w, N, limbs);
    (void)sub_mod(v, v, w, N, limbs);

    add_mod(w, u, q, N, limbs);
    (void)sub_mod(u, u, q, N, limbs);

    mont_mul(CB, t, u, N, np0, limbs);
    maybe_mont_normalize(CB, N, limbs);
    mont_mul(DA, v, w, N, np0, limbs);
    maybe_mont_normalize(DA, N, limbs);

    mont_sqr(AA, w, N, np0, limbs);
    mont_sqr(BB, u, N, np0, limbs);
    maybe_mont_normalize(AA, N, limbs);
    maybe_mont_normalize(BB, N, limbs);

    mont_mul(q, AA, BB, N, np0, limbs);
    maybe_mont_normalize(q, N, limbs);

    (void)sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult(dK, d, N, np0, limbs);

    add_mod(u, BB, dK, N, limbs);
    mont_mul(u, K, u, N, np0, limbs);
    maybe_mont_normalize(u, N, limbs);

    add_mod(w, DA, CB, N, limbs);
    (void)sub_mod(v, DA, CB, N, limbs);

    mont_sqr(w, w, N, np0, limbs);
    maybe_mont_normalize(w, N, limbs);
    mont_sqr(v, v, N, np0, limbs);
    maybe_mont_normalize(v, N, limbs);
    mp_shift_left_1_mod(v, v, N, limbs);
}

static inline void swap_limbs(uint *a, uint *b, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        uint tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

static inline void run_double_add_instance(uint instance_i, __global const uint *s_bits,
                                           ulong s_num_bits, ulong s_bits_start,
                                           ulong s_bits_interval, __global uint *data,
                                           uint sigma_0, uint np0, uint limbs) {
    uint base = instance_i * 5u * limbs;
    uint N[MAX_LIMBS];
    uint aX[MAX_LIMBS], aZ[MAX_LIMBS], bX[MAX_LIMBS], bZ[MAX_LIMBS];

    for (uint i = 0u; i < limbs; ++i) {
        N[i] = data[base + i];
        aX[i] = data[base + limbs + i];
        aZ[i] = data[base + 2u * limbs + i];
        bX[i] = data[base + 3u * limbs + i];
        bZ[i] = data[base + 4u * limbs + i];
    }

    uint d = sigma_0 + instance_i;
    int swapped = 0;

    ulong s_end = s_bits_start + s_bits_interval;
    if (s_end > s_num_bits) {
        s_end = s_num_bits;
    }

    for (ulong b = s_bits_start; b < s_end; ++b) {
        ulong nth = s_num_bits - 1ul - b;
        uint limb_idx = (uint)(nth >> 5);
        uint bit_idx = (uint)(nth & 31ul);
        int bit = (int)((s_bits[limb_idx] >> bit_idx) & 1u);

        if (bit != swapped) {
            swapped = !swapped;
            swap_limbs(aX, bX, limbs);
            swap_limbs(aZ, bZ, limbs);
        }
        double_add_v2(aX, aZ, bX, bZ, d, N, np0, limbs);
    }

    if (swapped) {
        swap_limbs(aX, bX, limbs);
        swap_limbs(aZ, bZ, limbs);
    }

    for (uint i = 0u; i < limbs; ++i) {
        data[base + limbs + i] = aX[i];
        data[base + 2u * limbs + i] = aZ[i];
        data[base + 3u * limbs + i] = bX[i];
        data[base + 4u * limbs + i] = bZ[i];
    }
}

#ifndef ECM_STAGE1_USE_COOP_WG
#define ECM_STAGE1_USE_COOP_WG 0
#endif

#if ECM_STAGE1_COOP_WG > 1
__kernel __attribute__((reqd_work_group_size(ECM_STAGE1_COOP_WG, 1, 1)))
#elif defined(ECM_STAGE1_WG_SIZE)
__kernel __attribute__((reqd_work_group_size(ECM_STAGE1_WG_SIZE, 1, 1)))
#else
__kernel
#endif
void kernel_double_add(__global const uint *s_bits, ulong s_num_bits, ulong s_bits_start,
                       ulong s_bits_interval, __global uint *data, uint count,
                       uint sigma_0, uint np0, uint limbs) {
    uint instance_i = get_global_id(0);
    if (instance_i >= count) return;
    if (limbs == 0u || limbs > MAX_LIMBS) return;
    run_double_add_instance(instance_i, s_bits, s_num_bits, s_bits_start, s_bits_interval, data,
                            sigma_0, np0, limbs);
}
