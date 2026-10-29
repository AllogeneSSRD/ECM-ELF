// Forward DIF: natural input -> bit-reversed spectrum.
// Inverse DIT: that spectrum -> natural output. No permutation buffer.
// Kernels use a real second NDRange dimension for independent batch members.
#ifndef ECM_NTT_TILE_BITS
#define ECM_NTT_TILE_BITS 8
#endif
#define ECM_NTT_TILE ((ulong)1 << ECM_NTT_TILE_BITS)

inline ulong ecm_ntt_root(__global const ulong *low, __global const ulong *high,
                         ulong exponent, uint inverse, uint low_bits,
                         ulong low_count, ulong high_count) {
    const ulong a = low[(ulong)inverse*low_count + (exponent & (low_count-1))];
    const ulong index = exponent >> low_bits;
    return index ? ecm_gl_mul(a,high[(ulong)inverse*high_count + index]) : a;
}

__kernel void ecm_ntt_global(__global ulong *data, ulong offset, ulong stride,
                            ulong n, ulong span_half, uint inverse,
                            __global const ulong *low, __global const ulong *high,
                            uint low_bits, ulong low_count, ulong high_count) {
    const ulong tid = (ulong)get_global_id(0);
    if (tid >= n/2) return;
    const ulong j = tid % span_half;
    const ulong i = (tid/span_half)*(2*span_half) + j;
    const ulong base = offset + (ulong)get_global_id(1)*stride;
    const ulong w = ecm_ntt_root(low,high,j*(n/(2*span_half)),inverse,
                                low_bits,low_count,high_count);
    const ulong u = data[base+i], v = data[base+i+span_half];
    if (inverse) {
        const ulong t = ecm_gl_mul(v,w);
        data[base+i] = ecm_gl_add(u,t);
        data[base+i+span_half] = ecm_gl_sub(u,t);
    } else {
        data[base+i] = ecm_gl_add(u,v);
        data[base+i+span_half] = ecm_gl_mul(ecm_gl_sub(u,v),w);
    }
}

__kernel void ecm_ntt_tile(__global ulong *data, __global const ulong *other,
                          ulong offset, ulong stride, ulong n, uint inverse,
                          uint product, ulong scale,
                          __global const ulong *low, __global const ulong *high,
                          uint low_bits, ulong low_count, ulong high_count) {
    __local ulong values[ECM_NTT_TILE];
    const ulong lane = (ulong)get_local_id(0), lanes = (ulong)get_local_size(0);
    const ulong base = offset + (ulong)get_group_id(1)*stride
                        + (ulong)get_group_id(0)*ECM_NTT_TILE;
    for (ulong i=lane; i<ECM_NTT_TILE; i+=lanes) {
        ulong v = data[base+i];
        if (inverse) {
            if (product) v = ecm_gl_mul(v,other[base+i]);
            v = ecm_gl_mul(v,scale);
        }
        values[i] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // The loop bound and direction are uniform for the whole work-group.
    for (ulong span_half = inverse ? (ulong)1 : ECM_NTT_TILE/2;
         span_half && span_half<ECM_NTT_TILE;
         span_half = inverse ? span_half*2 : span_half/2) {
        for (ulong tid=lane; tid<ECM_NTT_TILE/2; tid+=lanes) {
            const ulong j = tid % span_half, i = (tid/span_half)*(2*span_half)+j;
            const ulong w = ecm_ntt_root(low,high,j*(n/(2*span_half)),inverse,
                                        low_bits,low_count,high_count);
            const ulong u = values[i], v = values[i+span_half];
            if (inverse) {
                const ulong t = ecm_gl_mul(v,w);
                values[i] = ecm_gl_add(u,t);
                values[i+span_half] = ecm_gl_sub(u,t);
            } else {
                values[i] = ecm_gl_add(u,v);
                values[i+span_half] = ecm_gl_mul(ecm_gl_sub(u,v),w);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (ulong i=lane; i<ECM_NTT_TILE; i+=lanes) data[base+i] = values[i];
}
