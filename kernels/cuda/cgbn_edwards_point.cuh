#pragma once

// Extended Edwards arithmetic for a=1: x^2+y^2=1+d*x^2*y^2.
// All coordinates and d are canonical CGBN Montgomery residues modulo odd n.
// The caller must leave one spare container bit so modular additions cannot wrap.
// Input and output points may alias. This header is shared by the GPU Stage1
// correctness harness and the eventual batched worker.

template<class Env>
struct cgbn_edwards_point {
    using bn = typename Env::cgbn_t;
    bn x, y, z, t;
};

template<class Env>
struct cgbn_edwards_field {
    using bn = typename Env::cgbn_t;
    Env &env;
    const bn &n;
    uint32_t np0;

    __device__ __forceinline__ void add(bn &r, const bn &a, const bn &b) const {
        cgbn_add(env, r, a, b);
        if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
    }
    __device__ __forceinline__ void sub(bn &r, const bn &a, const bn &b) const {
        if (cgbn_sub(env, r, a, b)) cgbn_add(env, r, r, n);
    }
    __device__ __forceinline__ void mul(bn &r, const bn &a, const bn &b) const {
        cgbn_mont_mul(env, r, a, b, n, np0);
        if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
    }
    __device__ __forceinline__ void sqr(bn &r, const bn &a) const {
        cgbn_mont_sqr(env, r, a, n, np0);
        if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
    }

    __device__ __forceinline__ void dbl(cgbn_edwards_point<Env> &out,
                                         const cgbn_edwards_point<Env> &p) const {
        bn A, B, C, E, F, G, H, tmp;
        sqr(A, p.x);
        sqr(B, p.y);
        sqr(C, p.z);
        add(C, C, C);
        add(tmp, p.x, p.y);
        sqr(E, tmp);
        sub(E, E, A);
        sub(E, E, B);
        add(G, A, B);
        sub(F, G, C);
        sub(H, A, B);
        mul(out.x, E, F);
        mul(out.y, G, H);
        mul(out.t, E, H);
        mul(out.z, F, G);
    }

    __device__ __forceinline__ void add_point(cgbn_edwards_point<Env> &out,
                                               const cgbn_edwards_point<Env> &p,
                                               const cgbn_edwards_point<Env> &q,
                                               const bn &d) const {
        bn A, B, C, D, E, F, G, H, tmp, tmp2;
        mul(A, p.x, q.x);
        mul(B, p.y, q.y);
        mul(C, p.t, q.t);
        mul(C, C, d);
        mul(D, p.z, q.z);
        add(tmp, p.x, p.y);
        add(tmp2, q.x, q.y);
        mul(E, tmp, tmp2);
        sub(E, E, A);
        sub(E, E, B);
        sub(F, D, C);
        add(G, D, C);
        sub(H, B, A);
        mul(out.x, E, F);
        mul(out.y, G, H);
        mul(out.t, E, H);
        mul(out.z, F, G);
    }
};
