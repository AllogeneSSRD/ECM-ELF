// OpenCL kernels for Montgomery multiplication/squaring (CIOS style).
// One work-item handles one instance.

#ifndef MAX_LIMBS
#define MAX_LIMBS 128
#endif

__kernel void cgbn_mont_mul(
    __global const uint *a,
    __global const uint *b,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs)
{
    uint idx = get_global_id(0);
    uint base = idx * limbs;

    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    // Cache b and n into private arrays to reduce global memory traffic
    uint B[MAX_LIMBS];
    uint N[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = b[base + j];
        N[j] = n[base + j];
    }

    // Classic CIOS (Column Operand Scanning) Montgomery multiplication
    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[base + i];

        // t += ai * B (use cached B)
        ulong carry = 0ul;
        #pragma unroll 4
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);

        // t = (t + m*N) / 2^32 (use cached N)
        carry = 0ul;
        #pragma unroll 4
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }

        ulong top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        ulong top2 = (ulong)t_hi + (top >> 32);
        t[limbs] = (uint)top2;
        t_hi = (uint)(top2 >> 32);
    }

    // Conditional subtraction: if t >= n then t -= n
    int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
    if (!ge) {
        for (int i = (int)limbs - 1; i >= 0; --i) {
            uint tv = t[(uint)i];
            uint nv = N[(uint)i];
            if (tv > nv) {
                ge = 1;
                break;
            }
            if (tv < nv) {
                ge = 0;
                break;
            }
        }
    }

    if (ge) {
        ulong borrow = 0ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    }

    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = t[i];
    }
}



__kernel void cgbn_mont_sqr(
    __global const uint *a,
    __global const uint *n,
    __global uint *out,
    uint np0,
    uint limbs)
{
    uint idx = get_global_id(0);
    uint base = idx * limbs;

    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 1];
    for (uint i = 0u; i <= limbs; ++i) {
        t[i] = 0u;
    }
    uint t_hi = 0u;

    // Cache a and n into private arrays to reduce global memory traffic (sqr uses a twice)
    uint A[MAX_LIMBS];
    uint N[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        A[j] = a[base + j];
        N[j] = n[base + j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = A[i];

        // t += ai * a (use cached A)
        ulong carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)A[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong uvh = (ulong)t[limbs] + carry;
        t[limbs] = (uint)uvh;
        t_hi += (uint)(uvh >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);

        // t = (t + m*n) / 2^32 (use cached N)
        carry = 0ul;
        #pragma unroll
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }

        ulong top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        ulong top2 = (ulong)t_hi + (top >> 32);
        t[limbs] = (uint)top2;
        t_hi = (uint)(top2 >> 32);
    }

    int ge = (t_hi != 0u || t[limbs] != 0u) ? 1 : 0;
    if (!ge) {
        for (int i = (int)limbs - 1; i >= 0; --i) {
            uint tv = t[(uint)i];
            uint nv = n[base + (uint)i];
            if (tv > nv) {
                ge = 1;
                break;
            }
            if (tv < nv) {
                ge = 0;
                break;
            }
        }
    }

    if (ge) {
        ulong borrow = 0ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)n[base + i];
            ulong w = tv - nv - borrow;
            t[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
    }

    for (uint i = 0u; i < limbs; ++i) {
        out[base + i] = t[i];
    }
}
