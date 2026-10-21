// ECM stage1 ladder — __local memory variant.
// Only t_local and B_local live in LDS (the two largest arrays in mont_mul/sqr
// that cause scratch-memory spill). Output buffers (CB/DA/AA/BB) stay private
// because add_mod/sub_mod/maybe_normalize don't accept __local pointers.
//
// Scratch layout per work-item (2 × (MAX_LIMBS+2) uints):
//   Block 0  →  t_local   (mont_mul/sqr intermediate workspace)
//   Block 1  →  B_local   (mont_mul/sqr multiplicand copy)
//
// WG_SIZE injected by build plan: 16 for <=4096b, 8 for >4096b.
#ifndef ECM_STAGE1_WG_SIZE
#define ECM_STAGE1_WG_SIZE 8u
#endif
#define ECM_STAGE1_LOCAL_PAD 2u
#define ECM_STAGE1_LOCAL_BLOCKS 2u
// Pre-computed: ECM_STAGE1_WG_SIZE * ECM_STAGE1_LOCAL_BLOCKS * (MAX_LIMBS + ECM_STAGE1_LOCAL_PAD)
#define ECM_STAGE1_LOCAL_SCRATCH_U32 (ECM_STAGE1_WG_SIZE * ECM_STAGE1_LOCAL_BLOCKS * (MAX_LIMBS + ECM_STAGE1_LOCAL_PAD))

#ifndef ECM_STAGE1_MUL_IMPL
#error "ECM_STAGE1_MUL_IMPL not defined"
#endif
#ifndef ECM_STAGE1_SQR_IMPL
#error "ECM_STAGE1_SQR_IMPL not defined"
#endif

// Local mont callers — pass two extra LDS pointers for t_local/B_local.
#define mont_mul_local(out, a, b, N, np0, limbs, tl, bl) \
    ECM_STAGE1_MUL_IMPL((out), (a), (b), (N), (np0), (limbs), (tl), (bl))
#define mont_sqr_local(out, a, N, np0, limbs, tl, bl) \
    ECM_STAGE1_SQR_IMPL((out), (a), (N), (np0), (limbs), (tl), (bl))

// Standard operator macros (add/sub/special_mult use private memory).
#ifndef ECM_STAGE1_ADD_IMPL
#error "ECM_STAGE1_ADD_IMPL not defined"
#endif
#ifndef ECM_STAGE1_SUB_IMPL
#error "ECM_STAGE1_SUB_IMPL not defined"
#endif
#ifndef ECM_STAGE1_SPECIAL_MULT_IMPL
#error "ECM_STAGE1_SPECIAL_MULT_IMPL not defined"
#endif
#define add_mod(r, a, b, N, limbs) ECM_STAGE1_ADD_IMPL((r), (a), (b), (N), (limbs))
#define sub_mod(r, a, b, N, limbs) ECM_STAGE1_SUB_IMPL((r), (a), (b), (N), (limbs))
#define special_mult(r, m, N, np0, limbs) ECM_STAGE1_SPECIAL_MULT_IMPL((r), (m), (N), (np0), (limbs))

static inline void double_add_v2_local(uint *q, uint *u, uint *w, uint *v, uint d, const uint *N,
                                       uint np0, uint limbs, __local uint *s) {
    // LDS blocks (each = limbs+2 uints, only for mont_mul/sqr temporary workspace)
    __local uint *const t_local = s;                       // Block 0
    __local uint *const b_local = s + (limbs + 2u);       // Block 1

    // All output buffers stay private (compatible with add_mod/sub_mod/maybe_normalize)
    uint t[MAX_LIMBS], CB[MAX_LIMBS], DA[MAX_LIMBS], AA[MAX_LIMBS], BB[MAX_LIMBS];
    uint K[MAX_LIMBS], dK[MAX_LIMBS];

    add_mod(t, v, w, N, limbs);
    (void)sub_mod(v, v, w, N, limbs);

    add_mod(w, u, q, N, limbs);
    (void)sub_mod(u, u, q, N, limbs);

    mont_mul_local(CB, t, u, N, np0, limbs, t_local, b_local);
    maybe_mont_normalize(CB, N, limbs);
    mont_mul_local(DA, v, w, N, np0, limbs, t_local, b_local);
    maybe_mont_normalize(DA, N, limbs);

    mont_sqr_local(AA, w, N, np0, limbs, t_local, b_local);
    mont_sqr_local(BB, u, N, np0, limbs, t_local, b_local);
    maybe_mont_normalize(AA, N, limbs);
    maybe_mont_normalize(BB, N, limbs);

    mont_mul_local(q, AA, BB, N, np0, limbs, t_local, b_local);
    maybe_mont_normalize(q, N, limbs);

    (void)sub_mod(K, AA, BB, N, limbs);

    mp_copy(dK, K, limbs);
    special_mult(dK, d, N, np0, limbs);

    add_mod(u, BB, dK, N, limbs);
    mont_mul_local(u, K, u, N, np0, limbs, t_local, b_local);
    maybe_mont_normalize(u, N, limbs);

    add_mod(w, DA, CB, N, limbs);
    (void)sub_mod(v, DA, CB, N, limbs);

    mont_sqr_local(w, w, N, np0, limbs, t_local, b_local);
    maybe_mont_normalize(w, N, limbs);
    mont_sqr_local(v, v, N, np0, limbs, t_local, b_local);
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

static inline void run_double_add_instance_local(uint instance_i, __global const uint *s_bits,
                                                  ulong s_num_bits, ulong s_bits_start,
                                                  ulong s_bits_interval, __global uint *data,
                                                  uint sigma_0, uint np0, uint limbs,
                                                  __local uint *wi_scratch) {
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
        double_add_v2_local(aX, aZ, bX, bZ, d, N, np0, limbs, wi_scratch);
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

__kernel __attribute__((reqd_work_group_size(ECM_STAGE1_WG_SIZE, 1, 1)))
void kernel_double_add_local(__global const uint *s_bits, ulong s_num_bits, ulong s_bits_start,
                              ulong s_bits_interval, __global uint *data, uint count,
                              uint sigma_0, uint np0, uint limbs) {
    uint instance_i = get_global_id(0);
    if (instance_i >= count) return;
    if (limbs == 0u || limbs > MAX_LIMBS) return;

    __local uint scratch[ECM_STAGE1_LOCAL_SCRATCH_U32];
    uint lid = get_local_id(0);
    __local uint *wi_scratch = scratch + lid * 2u * (MAX_LIMBS + 2u);

    run_double_add_instance_local(instance_i, s_bits, s_num_bits, s_bits_start,
                                   s_bits_interval, data, sigma_0, np0, limbs, wi_scratch);
}
