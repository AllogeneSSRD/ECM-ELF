#pragma once
#include "options.h"
#include "../dispatch/tiers.def"
#define ECM_S2G_CAPACITY_ROW(NW,FN,...) if (_nwd <= NW) { FN<NW>(__VA_ARGS__); } else
/* the word-count dispatch: seven instantiations cover every N up to 16384 bits */
#define S2G_DISPATCH(NWD, FN, ...)                                                      \
    do {                                                                                \
        const int _nwd = (NWD);                                                         \
        ECM_CUDA_STAGE2_CAPACITIES(ECM_S2G_CAPACITY_ROW,FN,__VA_ARGS__)                   \
        {                                                                               \
            std::fprintf(stderr, "%s: N has %d words (%d bits); the ladder supports up " \
                                 "to 256 words (16384 bits)\n",                          \
                         NTT_PROBE_NAME, _nwd, _nwd * 64);                              \
            std::exit(2);                                                               \
        }                                                                               \
    } while (0)

template <int NW>
static void s2g_launch_mont_test(int nw, int cases, const unsigned long long *da,
                                 const unsigned long long *db, const unsigned long long *dn,
                                 unsigned long long ninv, unsigned long long *dout)
{
    const unsigned int th = 128;
    const unsigned int bl = (unsigned int)((cases + th - 1) / th);
    ecm_launch_s2g_mont_test_kernel<NW>(EcmCudaLaunch{bl, th}, da, db, dn, ninv, nw, cases, dout);
}

template <int NW>
static void s2g_launch_xadd_test(int nw,int cases,const unsigned long long *in,
                                 const unsigned long long *n,unsigned long long ninv,
                                 unsigned long long *out,bool candidate)
{
    const unsigned int th=64,bl=(cases*5+th-1)/th;
    if(candidate) ecm_launch_s2g_xadd_test_kernel<NW,true>(EcmCudaLaunch{bl,th}, in,n,ninv,nw,cases,out);
    else ecm_launch_s2g_xadd_test_kernel<NW,false>(EcmCudaLaunch{bl,th}, in,n,ninv,nw,cases,out);
}

template <int NW>
static void s2g_launch_ladder(int nw, int npts, const unsigned long long *dn,
                              unsigned long long ninv, const unsigned long long *dqx,
                              const unsigned long long *dqz, const unsigned long long *da24,
                              const unsigned long long *dmone, const unsigned long long *djs,
                              unsigned long long *dx, unsigned long long *dz, bool normal_output=true)
{
    const unsigned int th = 64;
    /* the same kernel, launched once per <= g_ladder_cap points (grid-stride inside).
       THE GRID IS SIZED TO THE CHUNK, NOT TO THE CAP.  This one line was worth 100x: the launch
       used to be `ceil(g_ladder_cap/64)` blocks whatever the chunk was, i.e. 8192 threads for a
       ONE-point chain, and this kernel's arrays are indexed with the runtime `nw` and therefore
       live in LOCAL memory (see the note on s2g_mont_mul) -- so every launch committed
       ~8192 x 5 KB = 40 MB of local memory for nothing.  Measured before the fix: a 1-point
       launch (the setup chain, 25 sequential prime powers) cost 0.51 s and a 2768-point batch
       cost 1.29 s, i.e. almost all of it launch overhead; the naming loop's 664388 candidate
       ladders cost 309 s of a 318 s run (section 26).  The thread->point mapping and the
       grid-stride arithmetic are unchanged, so the result is bit-identical. */
    for (int p0 = 0; p0 < npts; p0 += g_ladder_cap) {
        const int m = ((npts - p0) < g_ladder_cap) ? (npts - p0) : g_ladder_cap;
        const unsigned int bl = (unsigned int)((m + th - 1) / th);
        if constexpr(g_xadd6)
            ecm_launch_s2g_ladder_kernel<NW,true>(EcmCudaLaunch{bl,th}, dn,ninv,nw,dqx,dqz,da24,dmone,djs+p0,m,
                                                dx+(size_t)p0*nw,dz+(size_t)p0*nw,normal_output);
        else ecm_launch_s2g_ladder_kernel<NW>(EcmCudaLaunch{bl, th}, dn, ninv, nw, dqx, dqz, da24, dmone, djs + p0, m,
                                          dx + (size_t)p0 * nw, dz + (size_t)p0 * nw, normal_output);
    }
}

template <int NW>
static void s2g_launch_seed_pair(int nw,unsigned long long clo,unsigned long long npts,
                               unsigned long long per_block,unsigned long long blocks,
                               const unsigned long long *dn,unsigned long long ninv,
                               const unsigned long long *hx,const unsigned long long *hz,
                               const unsigned long long *a24,const unsigned long long *mone,
                               unsigned long long *ox,unsigned long long *oz)
{
    for(unsigned long long offset=0;offset<blocks;offset+=g_ladder_cap) {
        const unsigned long long count=std::min(blocks-offset,(unsigned long long)g_ladder_cap);
        const unsigned int th=nw>128?32:64,bl=(unsigned int)((count+th-1)/th);
        if constexpr(g_xadd6)ecm_launch_s2g_seed_pair_kernel<NW,true>(EcmCudaLaunch{bl,th}, dn,ninv,nw,hx,hz,a24,mone,
            clo,npts,per_block,blocks,offset,count,ox,oz);
        else ecm_launch_s2g_seed_pair_kernel<NW,false>(EcmCudaLaunch{bl,th}, dn,ninv,nw,hx,hz,a24,mone,
            clo,npts,per_block,blocks,offset,count,ox,oz);
    }
}

template <int NW>
static void s2g_launch_chain(int nw, unsigned long long blocks, unsigned long long npts,
                             unsigned long long per_block, unsigned long long ninv,
                             const unsigned long long *dn, const unsigned long long *ddx,
                             const unsigned long long *ddz, const unsigned long long *dsx,
                             const unsigned long long *dsz, const unsigned long long *esx,
                             const unsigned long long *esz, unsigned long long *ox,
                             unsigned long long *oz, size_t seed_stride=1)
{
    const unsigned int th = nw>128?32:64;
    const unsigned int bl = (unsigned int)((blocks + th - 1) / th);
    if constexpr(g_xadd6)
        ecm_launch_s2g_chain_kernel<NW,true>(EcmCudaLaunch{bl,th}, dn,ninv,nw,ddx,ddz,dsx,dsz,esx,esz,npts,per_block,
                                           blocks,ox,oz,seed_stride);
    else ecm_launch_s2g_chain_kernel<NW>(EcmCudaLaunch{bl, th}, dn, ninv, nw, ddx, ddz, dsx, dsz, esx, esz, npts, per_block,
                                    blocks, ox, oz, seed_stride);
}

template <int NW>
static void s2g_launch_segprod(int nw, unsigned long long npts, unsigned long long seg,
                               unsigned long long ninv, const unsigned long long *dn,
                               const unsigned long long *dz, unsigned long long *out, const unsigned long long *fix=nullptr)
{
    if(!npts)return;
    const unsigned long long nseg = (npts + seg - 1) / seg;
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((nseg + th - 1) / th);
    ecm_launch_s2g_segprod_kernel<NW>(EcmCudaLaunch{bl, th}, dn, ninv, nw, dz, npts, seg, nseg, out, fix);
}

template <int NW>
static void s4_launch_reduce(int nw, int L, unsigned long long nbatch,
                             unsigned long long out_slots, unsigned long long total,
                             const unsigned long long *ddig, unsigned long long n, int bpw,
                             unsigned long long slot_words, const unsigned long long *dn,
                             unsigned long long ninv, const unsigned long long *dy,
                             unsigned long long w, unsigned long long *dout,
                             unsigned long long slot_bits, unsigned long long *dbad,
                             unsigned long long *s4_dbg = nullptr,
                             unsigned long long first = 0, unsigned long long count = ~0ull, int mersenne_bits=0)
{
    const unsigned int th = 128;
    const unsigned int bl = (unsigned int)((total + th - 1) / th);
    if(mersenne_bits && !s4_tail_mont_mode() && s4_dbg==nullptr)
        ecm_launch_s4_reduce_kernel<NW,true>(EcmCudaLaunch{bl,th}, ddig,n,bpw,slot_words,out_slots,nbatch,dn,ninv,nw,
            L,dy,w,dout,slot_bits,dbad,s4_dbg,0,g_div_shift,g_div_recip,first,count==~0ull ? out_slots : count,mersenne_bits);
    else
        ecm_launch_s4_reduce_kernel<NW>(EcmCudaLaunch{bl, th}, ddig, n, bpw, slot_words, out_slots, nbatch, dn, ninv, nw,
            L, dy, w, dout, slot_bits, dbad, s4_dbg, s4_tail_mont_mode(),
            g_div_shift, g_div_recip, first, count==~0ull ? out_slots : count,0);
}

template <int NW>
static void s2g_launch_block_prod(int nw, int nvals, int per_block,
                                  const unsigned long long *dv, const unsigned long long *dn,
                                  unsigned long long ninv, unsigned long long *dout)
{
    const int nb = (nvals + per_block - 1) / per_block;
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((nb + th - 1) / th);
    ecm_launch_s2g_block_prod_kernel<NW>(EcmCudaLaunch{bl, th}, dv, nvals, nw, per_block, dn, ninv, dout);
}
