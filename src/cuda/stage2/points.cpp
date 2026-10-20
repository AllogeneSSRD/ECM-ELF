// Baby/giant generation, normalization, segment inverses and nonunit recovery.
#include "points.h"
#include "product.h"
#include "options.h"
#include "checks.h"
#include "point_launch.h"
#include "stage2_baby_launch.h"

bool affine_x_gmp_checked(mpz_t out,const mpz_t X,const mpz_t Z,const mpz_t N)
{
    if (mpz_cmp_ui(Z, 0) == 0 || (mpz_cmp(Z,N)>=0 && mpz_divisible_p(Z,N))) {
        mpz_set_ui(out, 0); return true;
    }
    mpz_t inv;
    mpz_init(inv);
    const bool ok = (mpz_invert(inv, Z, N) != 0);
    if (!ok) {
        mpz_mod(out, X, N);
    } else {
        mpz_mul(out, X, inv);
        mpz_mod(out, out, N);
    }
    mpz_clear(inv);
    return ok;
}

void affine_x_gmp(mpz_t out,const mpz_t X,const mpz_t Z,const mpz_t N)
{
    (void)affine_x_gmp_checked(out, X, Z, N);
}

GfinvBatch::GfinvBatch(
    const std::vector<unsigned long long> &p,
    size_t w,
    mpz_srcptr n,
    bool on,
    GfinvStats &s):products(p),W(w),N(n),enabled(on),stats(s)
{
    if(enabled) {
        for(size_t j=0;j<=GROUP;++j)mpz_init(prefix[j]);
        for(size_t j=0;j<GROUP;++j)mpz_init(inverse[j]);
        mpz_inits(acc,value,nullptr);
    }
}

void GfinvBatch::prepare(size_t index)
{
    const double t0=now_s();base=index/GROUP*GROUP;
    count=std::min(GROUP,products.size()/W-base);
    ++stats.groups;stats.segments+=count;
    mpz_set_ui(prefix[0],1);
    for(size_t j=0;j<count;++j) {
        words_to_mpz(value,products.data()+(base+j)*W,W);
        mpz_mul(prefix[j+1],prefix[j],value);mpz_mod(prefix[j+1],prefix[j+1],N);
    }
    ++stats.group_attempts;
    if(mpz_invert(acc,prefix[count],N)) {
        // Product is a unit iff every factor is a unit, including composite N.
        for(size_t j=count;j--;) {
            mpz_mul(inverse[j],acc,prefix[j]);mpz_mod(inverse[j],inverse[j],N);
            words_to_mpz(value,products.data()+(base+j)*W,W);
            mpz_mul(acc,acc,value);mpz_mod(acc,acc,N);unit[j]=true;
        }
        stats.good+=count;
    } else {
        // Recover the EXACT original per-segment classification, not a group-wide failure.
        ++stats.group_failures;
        for(size_t j=0;j<count;++j) {
            words_to_mpz(value,products.data()+(base+j)*W,W);
            ++stats.individual_attempts;unit[j]=mpz_invert(inverse[j],value,N)!=0;
            if(unit[j])++stats.good;else {++stats.nonunits;mpz_set_ui(inverse[j],0);}
        }
    }
    unsigned long long bytes=sizeof(mp_limb_t)*(acc[0]._mp_alloc+value[0]._mp_alloc);
    for(size_t j=0;j<=GROUP;++j)bytes+=sizeof(mp_limb_t)*prefix[j][0]._mp_alloc;
    for(size_t j=0;j<GROUP;++j)bytes+=sizeof(mp_limb_t)*inverse[j][0]._mp_alloc;
    stats.scratch_peak_bytes=std::max(stats.scratch_peak_bytes,bytes);
    stats.t_prepare+=now_s()-t0;
}

bool GfinvBatch::get(mpz_t out,size_t index)
{
    if(!enabled || !W || products.size()%W || index>=products.size()/W) {
        std::fprintf(stderr,"%s: FATAL: segment inverse cache index outside grid\n",NTT_PROBE_NAME);std::exit(3);
    }
    ++stats.requests;
    if(base==(size_t)-1 || index<base || index>=base+count)prepare(index);else ++stats.cache_hits;
    const size_t j=index-base;mpz_set(out,inverse[j]);
    static bool poisoned=false;
    const char *fault=std::getenv("NTT_GFINV_BATCH_TEST_BAD");
    if(unit[j] && fault && std::atoi(fault) && !poisoned) {mpz_add_ui(out,out,1);poisoned=true;}
    return unit[j];
}

void words_neg_mod_n(
    std::vector<unsigned long long> &dst,
    size_t off,
    const unsigned long long *x,
    const unsigned long long *n,
    size_t W)
{
    bool zero = true;
    for (size_t i = 0; i < W; ++i)
        if (x[i]) { zero = false; break; }
    if (zero) {
        for (size_t i = 0; i < W; ++i) dst[off + i] = 0ull;
        return;
    }
    unsigned long long borrow = 0;
    for (size_t i = 0; i < W; ++i) {
        const unsigned long long t = n[i] - x[i];
        const unsigned long long b1 = (n[i] < x[i]) ? 1ull : 0ull;
        const unsigned long long d = t - borrow;
        const unsigned long long b2 = (t < borrow) ? 1ull : 0ull;
        dst[off + i] = d;
        borrow = b1 | b2;
    }
}

void gfinv_segprod_host(
    std::vector<unsigned long long> &out,
    size_t npts,
    size_t nw,
    const std::vector<unsigned long long> &gz,
    const mpz_t N)
{
    const size_t nseg = (npts + S2G_GFINV_SEG - 1) / S2G_GFINV_SEG;
    out.assign(nseg * nw, 0ull);
    mpz_t p, t;
    mpz_inits(p, t, nullptr);
    std::vector<unsigned long long> tmp(nw, 0ull);
    for (size_t s = 0; s < nseg; ++s) {
        const size_t a = s * S2G_GFINV_SEG;
        const size_t b = ((a + S2G_GFINV_SEG) < npts) ? (a + S2G_GFINV_SEG) : npts;
        words_to_mpz(p, &gz[a * nw], nw);
        for (size_t i = a + 1; i < b; ++i) {
            words_to_mpz(t, &gz[i * nw], nw);
            mpz_mul(p, p, t);
            mpz_mod(p, p, N);
        }
        mpz_to_words(tmp, nw, p);
        std::copy(tmp.begin(), tmp.end(), out.begin() + (long)(s * nw));
    }
    mpz_clears(p, t, nullptr);
}

void ladder_points(
    const LadderCtx &C,
    const std::vector<unsigned long long> &js,
    std::vector<unsigned long long> &outx,
    std::vector<unsigned long long> &outz)
{
    stage2_memory::PersistentScope persistent_ladder_cache;
    const size_t nw = C.nw, n = js.size();
    outx.assign(n * nw, 0ull);
    outz.assign(n * nw, 0ull);
    if (n == 0) return;
    /* ONE set of device buffers for every ladder call in the process.  This used to be eight
       cudaMalloc + eight cudaFree per CALL, and at nw=83 (5261 bits) the driver overhead
       dominated the arithmetic completely: the setup chain is 25 calls with ONE point each and
       it measured 12.693 s of every real run (~0.5 s per call), while each of those ladders is a
       <=10-bit chain, i.e. a few hundred Montgomery multiplications (section 26.6).  The buffers
       grow on demand and are keyed by nw, so the only thing that changes is when memory is
       requested: the kernel, the packing, the values and the results are bit-identical. */
    static unsigned long long *dn = nullptr, *dqx = nullptr, *dqz = nullptr, *da24 = nullptr,
                              *dmone = nullptr, *djs = nullptr, *dx = nullptr, *dz = nullptr;
    static size_t cap_nw = 0, cap_n = 0;
    if (cap_nw != nw) {
        cudaFree(dn); cudaFree(dqx); cudaFree(dqz); cudaFree(da24); cudaFree(dmone);
        cudaFree(djs); cudaFree(dx); cudaFree(dz);
        dn = dqx = dqz = da24 = dmone = djs = dx = dz = nullptr;
        cap_nw = nw;
        cap_n = 0;
        CK(cudaMalloc(&dn, nw * 8));
        CK(cudaMalloc(&dqx, nw * 8));
        CK(cudaMalloc(&dqz, nw * 8));
        CK(cudaMalloc(&da24, nw * 8));
        CK(cudaMalloc(&dmone, nw * 8));
    }
    if (n > cap_n) {
        cudaFree(djs); cudaFree(dx); cudaFree(dz);
        djs = dx = dz = nullptr;
        CK(cudaMalloc(&djs, n * 8));
        CK(cudaMalloc(&dx, n * nw * 8));
        CK(cudaMalloc(&dz, n * nw * 8));
        cap_n = n;
    }
    CK(cudaMemcpy(dn, C.hn.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqx, C.hqx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqz, C.hqz.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da24, C.ha24.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmone, C.hmone.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(djs, js.data(), n * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_ladder, (int)nw, (int)n, dn, C.ninv, dqx, dqz, da24, dmone,
                 djs, dx, dz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(outx.data(), dx, outx.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outz.data(), dz, outz.size() * 8, cudaMemcpyDeviceToHost));
}

bool S3Workspace::need_giant_base(unsigned long long d,mpz_srcptr modulus)
{
    if(giant_base_d==d)return giant_base_unit;
    const double begin=now_s();const size_t nw=C->nw;
    if(!giant_base){CK(cudaMalloc(&giant_base,16*nw));bytes+=16*nw;}
    {
        need_pts(1);CK(cudaMemcpy(djs,&d,8,cudaMemcpyHostToDevice));
        S2G_DISPATCH((int)nw,s2g_launch_ladder,(int)nw,1,dn,C->ninv,dqx,dqz,da24,dmone,
                     djs,giant_base,giant_base+nw,false);
        CK(cudaGetLastError());++ladder_calls;++ladder_points_total;
        std::vector<unsigned long long> z(nw);
        CK(cudaMemcpy(z.data(),giant_base+nw,8*nw,cudaMemcpyDeviceToHost));
        mpz_t value,gcd;mpz_inits(value,gcd,nullptr);
        words_to_mpz(value,z.data(),nw);mpz_gcd(gcd,value,modulus);
        giant_base_unit=mpz_cmp_ui(gcd,1)==0;mpz_clears(value,gcd,nullptr);
        ++g_giant_seed.base_gpu_builds;g_giant_seed.base_d2h_bytes+=8*nw;
    }
    giant_base_d=d;++g_giant_seed.base_builds;
    g_giant_seed.base_build_seconds+=now_s()-begin;
    if(!giant_base_unit)++g_giant_seed.base_nonunits;
    return giant_base_unit; // Preserve the original factor path for nonunits/infinity.
}

void ladder_points_ws(
    S3Workspace &W,
    const std::vector<unsigned long long> &js,
    std::vector<unsigned long long> &outx,
    std::vector<unsigned long long> &outz)
{
    stage2_trace::Range trace("Giant.ladder");
    const LadderCtx &C = *W.C;
    const size_t nw = C.nw, n = js.size();
    outx.assign(n * nw, 0ull);
    outz.assign(n * nw, 0ull);
    if (n == 0) return;
    W.need_pts(n);
    CK(cudaMemcpy(W.djs, js.data(), n * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_ladder, (int)nw, (int)n, W.dn, C.ninv, W.dqx, W.dqz,
                 W.da24, W.dmone, W.djs, W.dx, W.dz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(outx.data(), W.dx, outx.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outz.data(), W.dz, outz.size() * 8, cudaMemcpyDeviceToHost));
    ++W.ladder_calls;
    W.ladder_points_total += n;
}

const unsigned long long *ResidentGiant::coord(size_t q,bool use_z) const
{
    const size_t group=q/(GROUP*SEG),local=q%(GROUP*SEG);
    const auto &v=(use_z?bad_z:bad_x)[group];
    if(local*w+w>v.size()){std::fprintf(stderr,"FATAL: sparse giant coordinate outside bad group\n");std::exit(3);}
    return v.data()+local*w;
}

bool ResidentGiant::good(size_t segment) const
{return group_good[segment/GROUP]!=0;}

void ResidentGiant::prepare(
    S3Workspace &ws,
    PolyLayer &L,
    std::vector<unsigned long long> &gseg,
    mpz_t Ginv,
    unsigned long long &gamma_points,
    unsigned long long &projective_segments,
    DeviceGLeafStats &st)
{
    const double begin=now_s();const size_t ns=(n+SEG-1)/SEG,ng=(ns+GROUP-1)/GROUP;
    const auto bad_before=st.bad_point_d2h_bytes,seg_before=st.bad_segment_d2h_bytes;
    ++st.chunks;st.groups+=ng;st.coord_peak_bytes=std::max(st.coord_peak_bytes,16ull*n*w);
    group_good.assign(ng,0);bad_x.resize(ng);bad_z.resize(ng);
    if(!groups_device)CK(cudaMalloc(&groups_device,ng*w*8));
    if(!group_fix)CK(cudaMalloc(&group_fix,(GROUP+1)*w*8));
    std::vector<unsigned long long> fix((GROUP+1)*w),word(w),products(ng*w);
    mpz_t R,p,v,inv;mpz_inits(R,p,v,inv,nullptr);words_to_mpz(R,ws.C->hmone.data(),w);mpz_set_ui(p,1);
    for(size_t j=0;j<=GROUP;++j) {
        mpz_to_words(word,w,p);std::copy(word.begin(),word.end(),fix.begin()+j*w);
        mpz_mul(p,p,R);mpz_mod(p,p,L.N);
    }
    CK(cudaMemcpy(group_fix,fix.data(),fix.size()*8,cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)w,s2g_launch_segprod,(int)w,ns,GROUP,ws.C->ninv,ws.dn,segments,groups_device,group_fix);
    CK(cudaGetLastError());CK(cudaMemcpy(products.data(),groups_device,products.size()*8,cudaMemcpyDeviceToHost));
    st.group_d2h_bytes+=products.size()*8;st.group_peak_bytes=std::max(st.group_peak_bytes,8ull*(ng+GROUP+1)*w);
    std::vector<unsigned long long> all_segments;
    if(device_gleaf_flag("NTT_DEVICE_GLEAF_CHECK")) {
        all_segments.resize(ns*w);CK(cudaMemcpy(all_segments.data(),segments,all_segments.size()*8,cudaMemcpyDeviceToHost));
    }
    for(size_t h=0;h<ng;++h) {
        const size_t lo=h*GROUP,hi=std::min(ns,lo+GROUP);
        words_to_mpz(p,products.data()+h*w,w);
        if(!all_segments.empty()) {
            mpz_set_ui(v,1);
            for(size_t k=lo;k<hi;++k){words_to_mpz(inv,all_segments.data()+k*w,w);mpz_mul(v,v,inv);mpz_mod(v,v,L.N);}
            if(mpz_cmp(p,v)){std::fprintf(stderr,"FATAL: GPU Gamma group GMP mismatch\n");std::exit(3);}++st.checked_groups;
        }
        const double ti=now_s();const bool unit=mpz_invert(inv,p,L.target())!=0;st.t_invert+=now_s()-ti;
        if(unit) {
            group_good[h]=1;mpz_mul(Ginv,Ginv,inv);mpz_mod(Ginv,Ginv,L.N);
            gamma_points+=std::min(n,hi*SEG)-lo*SEG;projective_segments+=hi-lo;st.good_segments+=hi-lo;
        } else {
            ++st.bad_groups;CK(cudaMemcpy(gseg.data()+lo*w,segments+lo*w,(hi-lo)*w*8,cudaMemcpyDeviceToHost));
            st.bad_segment_d2h_bytes+=(hi-lo)*w*8;
            const size_t first=lo*SEG,count=std::min(n,hi*SEG)-first;
            bad_x[h].resize(count*w);bad_z[h].resize(count*w);
            CK(cudaMemcpy(bad_x[h].data(),x+first*w,count*w*8,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(bad_z[h].data(),z+first*w,count*w*8,cudaMemcpyDeviceToHost));
            st.bad_point_d2h_bytes+=count*w*16;
        }
    }
    st.avoided_point_d2h_bytes+=16ull*n*w-(st.bad_point_d2h_bytes-bad_before);
    st.avoided_segment_d2h_bytes+=8ull*ns*w-(st.bad_segment_d2h_bytes-seg_before);
    mpz_clears(R,p,v,inv,nullptr);st.t_prepare+=now_s()-begin;
}

void giant_ladder_into(
    S3Workspace &ws,
    const LadderCtx &C,
    unsigned long long D,
    unsigned long long first,
    size_t points,
    unsigned long long *x,
    unsigned long long *z)
{
    const auto w=C.nw;
    std::vector<unsigned long long> scalars(std::min(points,ws.pt_cap));
    for(size_t offset=0;offset<points;offset+=scalars.size()) {
        const auto take=std::min(scalars.size(),points-offset);
        for(size_t i=0;i<take;++i)scalars[i]=(first+offset+i)*D;
        CK(cudaMemcpy(ws.djs,scalars.data(),take*8,cudaMemcpyHostToDevice));
        S2G_DISPATCH((int)w,s2g_launch_ladder,(int)w,(int)take,ws.dn,C.ninv,
            ws.dqx,ws.dqz,ws.da24,ws.dmone,ws.djs,x+offset*w,z+offset*w);
        CK(cudaGetLastError());++ws.ladder_calls;ws.ladder_points_total+=take;
    }
}

bool giant_chunk_chain(
    PolyLayer &L,
    const LadderCtx &C,
    S3Workspace &W,
    unsigned long long D,
    unsigned long long clo,
    unsigned long long chi,
    unsigned long long per_block,
    std::vector<unsigned long long> &gx,
    std::vector<unsigned long long> &gz,
    bool check,
    unsigned long long &seed_points,
    std::vector<unsigned long long> *gseg,
    unsigned long long seg,
    ResidentGiant *resident,
    bool wide_tail)
{
    stage2_trace::Range trace("Giant.chain");
    const size_t nw = C.nw;
    const unsigned long long npts = chi - clo + 1;
    const unsigned long long blocks = (npts + per_block - 1) / per_block;
    mpz_t Rm, X, Z;
    mpz_inits(Rm, X, Z, nullptr);
    /* R = the Montgomery constant (the image of 1) is exactly what the ladder context holds as
       `hmone`, so the normal->image conversion needs no extra constant */
    words_to_mpz(Rm, C.hmone.data(), nw);
    /* the ladder points we need: for every block its first two multiples, plus x_D itself */
    std::vector<unsigned long long> js;
    js.reserve((size_t)(2 * blocks + 1));
    for (unsigned long long b = 0; b < blocks; ++b) {
        const unsigned long long i0 = clo + b * per_block;
        js.push_back(i0 * D);
        js.push_back((i0 + 1 <= chi ? (i0 + 1) : i0) * D);
    }
    js.push_back(D);                                    /* the difference point x_D */
    seed_points = (unsigned long long)js.size();
    std::vector<unsigned long long> lx, lz;
    bool ladder_fallback=false;
    if(g_giant_seed_device) {
        W.need_pts(ecm_stage2::giant_seed_reserve(npts,js.size(),wide_tail));
        const bool paired=g_giant_seed_pair && W.need_giant_base(D,L.target());
        ladder_fallback=wide_tail && !paired;
        if(!ladder_fallback) {
        if(paired) {
            S2G_DISPATCH((int)nw,s2g_launch_seed_pair,(int)nw,clo,npts,per_block,blocks,
                         W.dn,C.ninv,W.giant_base,W.giant_base+nw,W.da24,W.dmone,W.dx,W.dz);
            ++g_giant_seed.paired_chunks;g_giant_seed.paired_ladders+=blocks;
            g_giant_seed.scalar_h2d_avoided+=js.size()*8;
        } else {
            CK(cudaMemcpy(W.djs,js.data(),js.size()*8,cudaMemcpyHostToDevice));
            S2G_DISPATCH((int)nw,s2g_launch_ladder,(int)nw,(int)js.size(),W.dn,C.ninv,
                         W.dqx,W.dqz,W.da24,W.dmone,W.djs,W.dx,W.dz,false);
        }
        CK(cudaGetLastError());
        ++W.ladder_calls;W.ladder_points_total+=paired?blocks:js.size();
        ++g_giant_seed.chunks;g_giant_seed.points+=js.size();
        g_giant_seed.avoided_d2h_bytes+=16ull*nw*js.size();
        g_giant_seed.avoided_h2d_bytes+=16ull*nw*js.size();
        g_giant_seed.avoided_cpu_modmuls+=2*js.size();
        g_giant_seed.avoided_montmuls+=2*js.size();
        if(g_giant_seed_check) {
            lx.resize(js.size()*nw);lz.resize(js.size()*nw);
            CK(cudaMemcpy(lx.data(),W.dx,lx.size()*8,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(lz.data(),W.dz,lz.size()*8,cudaMemcpyDeviceToHost));
            std::vector<unsigned long long> cx,cz,expect(nw);
            ladder_points(C,js,cx,cz); // independent normal-output endpoint, not W's lease
            mpz_t old_z,new_z;mpz_inits(old_z,new_z,nullptr);
            for(size_t i=0;i<js.size();++i)for(int z=0;z<2;++z) {
                if(paired) {
                    if(z)continue; // Projective scale differs; compare x by cross products.
                    words_to_mpz(X,cx.data()+i*nw,nw);words_to_mpz(new_z,lz.data()+i*nw,nw);
                    mpz_mul(X,X,new_z);mpz_mod(X,X,L.target());
                    words_to_mpz(Z,lx.data()+i*nw,nw);words_to_mpz(old_z,cz.data()+i*nw,nw);
                    mpz_mul(Z,Z,old_z);mpz_mod(Z,Z,L.target());
                    if(mpz_cmp(X,Z)!=0){std::fprintf(stderr,"FATAL: paired seed GMP projective mismatch point=%llu\n",(unsigned long long)i);std::exit(3);}
                    g_giant_seed.checked_words+=2*nw;continue;
                }
                words_to_mpz(X,(z?cz:cx).data()+i*nw,nw);mpz_mul(X,X,Rm);mpz_mod(X,X,L.N);
                mpz_to_words(expect,nw,X);
                if(!std::equal(expect.begin(),expect.end(),(z?lz:lx).begin()+i*nw)) {
                    std::fprintf(stderr,"%s: FATAL: device seed GMP image mismatch point=%llu\n",NTT_PROBE_NAME,(unsigned long long)i);std::exit(3);
                }
                g_giant_seed.checked_words+=nw;
            }
            mpz_clears(old_z,new_z,nullptr);
        }
        }
    } else ladder_points_ws(W, js, lx, lz);
    /* the seeds and the difference point as MONTGOMERY IMAGES (x*R mod N) */
    std::vector<unsigned long long> hdsx,hdsz,hesx,hesz,hdx,hdz;
    if(!g_giant_seed_device) {
        hdsx.resize((size_t)blocks*nw);hdsz.resize((size_t)blocks*nw);
        hesx.resize((size_t)blocks*nw);hesz.resize((size_t)blocks*nw);hdx.resize(nw);hdz.resize(nw);
        std::vector<unsigned long long> tmpw(nw, 0ull);
        auto img = [&](std::vector<unsigned long long> &dst, size_t off, size_t e, bool use_z) {
            words_to_mpz(X, use_z ? &lz[e * nw] : &lx[e * nw], nw);
            mpz_mul(X, X, Rm);
            mpz_mod(X, X, L.N);
            mpz_to_words(tmpw, nw, X);
            std::copy(tmpw.begin(), tmpw.end(), dst.begin() + (long)off);
        };
        for (unsigned long long b = 0; b < blocks; ++b) {
            img(hdsx, (size_t)b * nw, (size_t)(2 * b), false);
            img(hdsz, (size_t)b * nw, (size_t)(2 * b), true);
            img(hesx, (size_t)b * nw, (size_t)(2 * b + 1), false);
            img(hesz, (size_t)b * nw, (size_t)(2 * b + 1), true);
        }
        img(hdx, 0, (size_t)(2 * blocks), false);
        img(hdz, 0, (size_t)(2 * blocks), true);
    }
    unsigned long long *ddsx = nullptr, *ddsz = nullptr, *desx = nullptr, *desz = nullptr,
                       *ddx = nullptr, *ddz = nullptr, *ox = nullptr, *oz = nullptr;
    const size_t nseed = (size_t)blocks * nw;
    if(g_giant_seed_device) {
        ddsx=W.dx;ddsz=W.dz;desx=W.dx+nw;desz=W.dz+nw;
        ddx=W.dx+2*nseed;ddz=W.dz+2*nseed;
    } else {
    CK(cudaMalloc(&ddsx, nseed * 8));
    CK(cudaMalloc(&ddsz, nseed * 8));
    CK(cudaMalloc(&desx, nseed * 8));
    CK(cudaMalloc(&desz, nseed * 8));
    CK(cudaMalloc(&ddx, nw * 8));
    CK(cudaMalloc(&ddz, nw * 8));

    CK(cudaMemcpy(ddsx, hdsx.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddsz, hdsz.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(desx, hesx.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(desz, hesz.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddx, hdx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddz, hdz.data(), nw * 8, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&ox, (size_t)npts * nw * 8));
    CK(cudaMalloc(&oz, (size_t)npts * nw * 8));
    if(ladder_fallback) {
        // A nonunit [D]Q must retain the old individual-ladder factor path.
        // Reuse the reserved index tile and the normal chain output arrays;
        // no fallback-only device allocation or seed-capacity growth occurs.
        giant_ladder_into(W,C,D,clo,(size_t)npts,ox,oz);
        seed_points=0;
        stage2_log::print(stage2_log::debug,"wide_tail_ladder_fallback: points=%llu reason=nonunit_base index_capacity=%llu\n",
            npts,(unsigned long long)W.pt_cap);
    } else {
        S2G_DISPATCH((int)nw, s2g_launch_chain, (int)nw, blocks, npts, per_block, C.ninv, W.dn, ddx,
                     ddz, ddsx, ddsz, desx, desz, ox, oz,g_giant_seed_device?2:1);
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    if(!resident || check || g_gfinv_seg_check) {
        gx.assign((size_t)npts*nw,0);gz.assign((size_t)npts*nw,0);
        CK(cudaMemcpy(gx.data(),ox,gx.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gz.data(),oz,gz.size()*8,cudaMemcpyDeviceToHost));
    } else {std::vector<unsigned long long>().swap(gx);std::vector<unsigned long long>().swap(gz);}
    /* the per-segment z-products, while the point buffers are still on the device (section 42) */
    unsigned long long *dsp = nullptr;
    if (gseg) {
        const unsigned long long nseg = (npts + seg - 1) / seg;
        gseg->assign((size_t)nseg * nw, 0ull);
        CK(cudaMalloc(&dsp, (size_t)nseg * nw * 8));
        if(g_gfinv_seg_exact)W.need_segfix((size_t)seg);
        S2G_DISPATCH((int)nw, s2g_launch_segprod, (int)nw, npts, seg, C.ninv, W.dn, oz, dsp,
                    g_gfinv_seg_exact?W.dsegfix:nullptr);
        g_giant_seed.segments+=nseg;
        if(g_gfinv_seg_exact) {
            g_giant_seed.segment_fix_muls+=nseg-((npts%seg)==1?1:0);
            g_giant_seed.fix_table_peak_bytes=std::max(g_giant_seed.fix_table_peak_bytes,8ull*(W.segfix_cap+1)*nw);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        if(!resident || g_gfinv_seg_check)CK(cudaMemcpy(gseg->data(),dsp,gseg->size()*8,cudaMemcpyDeviceToHost));
        if(!resident)cudaFree(dsp);
        if(g_gfinv_seg_check) {
            if(seg!=S2G_GFINV_SEG){std::fprintf(stderr,"FATAL: segment check grid mismatch\n");std::exit(3);}
            std::vector<unsigned long long> expected;
            gfinv_segprod_host(expected,(size_t)npts,nw,gz,L.N);
            if(expected!=*gseg){std::fprintf(stderr,"%s: FATAL: segment product GMP mismatch\n",NTT_PROBE_NAME);std::exit(3);}
            g_giant_seed.segment_checks+=nseg;
        }
    }
    if(!g_giant_seed_device) {
        cudaFree(ddsx);cudaFree(ddsz);cudaFree(desx);cudaFree(desz);cudaFree(ddx);cudaFree(ddz);
    }
    if(resident){resident->x=ox;resident->z=oz;resident->segments=dsp;resident->n=npts;resident->w=nw;resident->own=true;}
    else {cudaFree(ox);cudaFree(oz);}
    /* THE CHECK: the same chunk through the ladder, compared on the AFFINE x value (the chain and
       the ladder hold different projective representatives of the same point, so X and Z cannot
       be compared -- X/Z mod N can, and that is the only quantity the caller uses). */
    if (check && !ladder_fallback) {
        std::vector<unsigned long long> gjs((size_t)npts, 0ull);
        for (unsigned long long i = 0; i < npts; ++i) gjs[(size_t)i] = (clo + i) * D;
        std::vector<unsigned long long> cx, cz;
        ladder_points_ws(W, gjs, cx, cz);
        std::vector<unsigned long long> a1(nw, 0ull), a2(nw, 0ull);
        unsigned long long bad = 0, first = 0;
        for (unsigned long long i = 0; i < npts; ++i) {
            mpz_t X1, Z1;
            mpz_inits(X1, Z1, nullptr);
            words_to_mpz(X1, &gx[(size_t)i * nw], nw);
            words_to_mpz(Z1, &gz[(size_t)i * nw], nw);
            affine_x_gmp(X, X1, Z1, L.target());
            mpz_to_words(a1, nw, X);
            words_to_mpz(X1, &cx[(size_t)i * nw], nw);
            words_to_mpz(Z1, &cz[(size_t)i * nw], nw);
            affine_x_gmp(X, X1, Z1, L.target());
            mpz_to_words(a2, nw, X);
            if (a1 != a2) {
                if (!bad) first = i;
                /* the PATTERN matters: sparse mismatches inside blocks mean an arithmetic edge
                   case, a whole block means a seed problem, and everything after one index means
                   a truncated transfer.  Print the first eight. */
                if (bad < 8)
                    std::fprintf(stderr, "giant_chain_mismatch: i=%llu block=%llu off=%llu\n",
                                 i, i / per_block, i % per_block);
                ++bad;
            }
            /* THE DEGENERACY WINDOW: a giant point whose Z is not invertible mod N is a point
               where stage 2 has effectively FOUND a factor (gcd(Z, N) > 1), and affine_x_gmp
               deliberately falls back to X for it.  Such a point cannot be compared between two
               different projective representatives -- and, more importantly, the CHAIN must not
               run through it (the xADD formula is undefined with Z = 0 mod q).  Print the window
               around the first mismatch so the two explanations can be told apart by inspection. */
            if (bad && i <= first + 12) {
                mpz_t Zc, Zl, g;
                mpz_inits(Zc, Zl, g, nullptr);
                words_to_mpz(Zc, &gz[(size_t)i * nw], nw);
                words_to_mpz(Zl, &cz[(size_t)i * nw], nw);
                mpz_gcd(g, Zc, L.target());
                const bool zc_bad = (mpz_cmp_ui(g, 1) > 0);
                mpz_gcd(g, Zl, L.target());
                const bool zl_bad = (mpz_cmp_ui(g, 1) > 0);
                std::fprintf(stderr, "giant_chain_window: i=%llu equal=%d chain_Z_noninvertible=%d "
                                     "ladder_Z_noninvertible=%d\n", i, (a1 == a2) ? 1 : 0,
                             zc_bad ? 1 : 0, zl_bad ? 1 : 0);
                mpz_clears(Zc, Zl, g, nullptr);
            }
            mpz_clears(X1, Z1, nullptr);
        }
        stage2_log::print(stage2_log::debug, "giant_chain_check: points=%llu blocks=%llu per_block=%llu seed_points=%llu "
                    "mismatches=%llu first=%llu\n", npts, blocks, per_block, seed_points, bad,
                    first);
    }
    mpz_clears(Rm, X, Z, nullptr);
    return !ladder_fallback;
}

int suyama_curve(mpz_t a24,mpz_t px,mpz_t pz,unsigned long long sigma,const mpz_t N)
{
    mpz_t sig, u, v, num, den, inv, A, t;
    mpz_inits(sig, u, v, num, den, inv, A, t, nullptr);
    set_u64_mpz(sig, sigma);
    mpz_mul(u, sig, sig);
    mpz_sub_ui(u, u, 5);                        /* u = sigma^2 - 5 */
    mpz_mul_ui(v, sig, 4);                      /* v = 4 sigma */
    mpz_sub(num, v, u);
    mpz_powm_ui(num, num, 3, N);                /* (v-u)^3 */
    mpz_mul_ui(t, u, 3);
    mpz_add(t, t, v);                           /* 3u+v */
    mpz_mul(num, num, t);
    mpz_powm_ui(den, u, 3, N);
    mpz_mul(den, den, v);
    mpz_mul_ui(den, den, 4);                    /* 4 u^3 v */
    int rc = 0;
    if (mpz_invert(inv, den, N) == 0) { rc = 1; goto done; }
    mpz_mul(A, num, inv);
    mpz_sub_ui(A, A, 2);                        /* A = (v-u)^3(3u+v)/(4u^3v) - 2 */
    mpz_add_ui(t, A, 2);
    mpz_set_ui(den, 4);
    if (mpz_invert(inv, den, N) == 0) { rc = 1; goto done; }
    mpz_mul(a24, t, inv);
    mpz_mod(a24, a24, N);                       /* a24 = (A+2)/4 */
    mpz_powm_ui(px, u, 3, N);                   /* X0 = u^3 */
    mpz_powm_ui(pz, v, 3, N);                   /* Z0 = v^3 */
done:
    mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
    return rc;
}

bool device_baby_generate(
    const LadderCtx &C,
    const mpz_t N,
    const std::vector<unsigned long long> &indices,
    std::vector<std::vector<unsigned long long>> &leaf,
    SmallPrimeBabyCache *cache,
    std::vector<std::string> &factors,
    unsigned long long &noninv,
    DeviceBabyStats &st,
    bool force_check)
{
    stage2_trace::Range trace("Baby.generate");
    const double begin=now_s();const size_t n=indices.size(),w=C.nw;
    if(!n)return false;
    ecm_stage2::BabyMemoryLayout layout;
    if(!ecm_stage2::baby_memory_layout(n,w,layout)) {
        std::fprintf(stderr,"FATAL: baby device layout overflow\n");std::exit(3);
    }
    const auto &counts=layout.counts;const auto &offset=layout.offset;const auto words=layout.tree_words;
    const size_t groups=counts[8];
    const size_t bytes=(size_t)layout.bytes;
    st.points=n;st.groups=groups;st.payload_bytes=bytes;
    size_t free_bytes=0,total_bytes=0;CK(cudaMemGetInfo(&free_bytes,&total_bytes));
    const auto limit=fuse_env_ull("NTT_BABY_DEVICE_MAX_MB",512)*1024*1024;
    if(bytes>limit || free_bytes<bytes+64ull*1024*1024 ||
       fuse_env_ull("NTT_BABY_DEVICE_ALLOC_FAIL",0)) {
        stage2_log::print(stage2_log::debug, "baby_device_fallback: payload_bytes=%llu limit_bytes=%llu reason=budget_or_injection\n",
            (unsigned long long)bytes,(unsigned long long)limit);return false;
    }
    BabyBuffer dn,dqx,dqz,da,dm,dj,dx,dz,tree,output,mask;
    for(auto *b:{&dn,&dqx,&dqz,&da,&dm})if(!b->allocate(w*8))return false;
    if(!dj.allocate(n*8) || !dx.allocate(n*w*8) || !dz.allocate(n*w*8) ||
       !tree.allocate(words*8) || !output.allocate(n*w*8) || !mask.allocate(groups))return false;
    CK(cudaMemcpy(dn.ptr,C.hn.data(),w*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqx.ptr,C.hqx.data(),w*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqz.ptr,C.hqz.data(),w*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da.ptr,C.ha24.data(),w*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm.ptr,C.hmone.data(),w*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dj.ptr,indices.data(),n*8,cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)w,s2g_launch_ladder,(int)w,(int)n,dn.words(),C.ninv,dqx.words(),dqz.words(),
                 da.words(),dm.words(),dj.words(),dx.words(),dz.words(),false);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());st.ladder_seconds=now_s()-begin;
    for(int l=1;l<=8;++l) {
        const auto *in=l==1?dz.words():tree.words()+offset[l-1];
        S2G_DISPATCH((int)w,s2g_launch_baby_product,(int)w,counts[l-1],in,tree.words()+offset[l],dn.words(),C.ninv);
        CK(cudaGetLastError());
    }
    std::vector<unsigned long long> roots(groups*w),negative(n*w),word(w);
    std::vector<unsigned char> good(groups,0);
    CK(cudaMemcpy(roots.data(),tree.words()+offset[8],roots.size()*8,cudaMemcpyDeviceToHost));
    st.root_d2h_bytes=roots.size()*8;
    mpz_t R,ri,p,inv,X,Z,x,neg,g;mpz_inits(R,ri,p,inv,X,Z,x,neg,g,nullptr);
    words_to_mpz(R,C.hmone.data(),w);
    if(!mpz_invert(ri,R,N)){std::fprintf(stderr,"FATAL: baby Montgomery radix not a unit\n");std::exit(3);}
    const double ti=now_s();
    for(size_t k=0;k<groups;++k) {
        words_to_mpz(p,roots.data()+k*w,w);
        if(mpz_invert(inv,p,N)) {
            good[k]=1;mpz_mul(inv,inv,R);mpz_mod(inv,inv,N); // ordinary inverse of the root product
            mpz_to_words(word,w,inv);std::copy(word.begin(),word.end(),roots.begin()+k*w);
        } else {++st.bad_groups;std::fill(roots.begin()+k*w,roots.begin()+(k+1)*w,0);}
    }
    st.invert_seconds=now_s()-ti;
    CK(cudaMemcpy(tree.words()+offset[8],roots.data(),roots.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mask.ptr,good.data(),groups,cudaMemcpyHostToDevice));st.seed_h2d_bytes=roots.size()*8+groups;
    for(int l=8;l>=2;--l) {
        S2G_DISPATCH((int)w,s2g_launch_baby_inverse,(int)w,counts[l-1],tree.words()+offset[l],
                     tree.words()+offset[l-1],(const unsigned char*)mask.ptr,size_t(256)>>l,dn.words(),C.ninv);
        CK(cudaGetLastError());
    }
    S2G_DISPATCH((int)w,s2g_launch_baby_leaf,(int)w,n,tree.words()+offset[1],dx.words(),dz.words(),
                 output.words(),(const unsigned char*)mask.ptr,dn.words(),C.ninv);
    CK(cudaGetLastError());CK(cudaMemcpy(negative.data(),output.ptr,negative.size()*8,cudaMemcpyDeviceToHost));
    st.leaf_d2h_bytes=negative.size()*8;
    // Restore exactly the ordinary-coordinate fallback, including X for a nonunit
    // Z and x=0 for Z=0. GCD/cache order follows the original sorted baby indices.
    for(size_t k=0;k<groups;++k)if(!good[k]) {
        const size_t lo=k*256,count=std::min(size_t(256),n-lo);
        std::vector<unsigned long long> bx(count*w),bz(count*w);
        CK(cudaMemcpy(bx.data(),dx.words()+lo*w,bx.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bz.data(),dz.words()+lo*w,bz.size()*8,cudaMemcpyDeviceToHost));
        st.coordinate_d2h_bytes+=16*count*w;
        for(size_t j=0;j<count;++j) {
            words_to_mpz(X,bx.data()+j*w,w);mpz_mul(X,X,ri);mpz_mod(X,X,N);
            words_to_mpz(Z,bz.data()+j*w,w);mpz_mul(Z,Z,ri);mpz_mod(Z,Z,N);
            if(cache && !mpz_sgn(Z))cache->record(lo+j,N);
            if(!affine_x_gmp_checked(x,X,Z,N)) {
                ++noninv;mpz_gcd(g,Z,N);if(cache)cache->record(lo+j,g);
                char *v=mpz_get_str(nullptr,10,g);factors.emplace_back(v);
                void (*release)(void*,size_t)=nullptr;mp_get_memory_functions(nullptr,nullptr,&release);
                release(v,std::strlen(v)+1);
            }
            mpz_neg(neg,x);mpz_mod(neg,neg,N);mpz_to_words(word,w,neg);
            std::copy(word.begin(),word.end(),negative.begin()+(lo+j)*w);
        }
    }
    if(fuse_env_ull("NTT_BABY_DEVICE_TEST_BAD",0))negative[0]^=1;
    if(force_check || fuse_env_ull("NTT_BABY_DEVICE_CHECK",0)) {
        std::vector<unsigned long long> bx,bz;ladder_points(C,indices,bx,bz);
        unsigned long long check_noninv=0;
        for(size_t j=0;j<n;++j) {
            words_to_mpz(X,bx.data()+j*w,w);words_to_mpz(Z,bz.data()+j*w,w);
            const bool unit=affine_x_gmp_checked(x,X,Z,N);if(!unit)++check_noninv;
            mpz_neg(neg,x);mpz_mod(neg,neg,N);
            // Device leaves are canonical in the carrier, not necessarily N.
            words_to_mpz(p,negative.data()+j*w,w);mpz_mod(p,p,N);
            if(mpz_cmp(neg,p)) {
                std::fprintf(stderr,"FATAL: baby device affine GMP mismatch point=%llu\n",(unsigned long long)j);std::exit(3);
            }
            if(cache){mpz_gcd(g,Z,N);cache->gcd_at(j,p);if(mpz_cmp(g,p)){
                std::fprintf(stderr,"FATAL: baby device cache GCD mismatch\n");std::exit(3);}}
            st.checked_words+=w;
        }
        if(check_noninv!=noninv){std::fprintf(stderr,"FATAL: baby device nonunit count mismatch\n");std::exit(3);}
    }
    for(size_t j=0;j<n;++j){leaf[j].assign(2*w,0);std::copy(negative.begin()+j*w,negative.begin()+(j+1)*w,leaf[j].begin());leaf[j][w]=1;}
    mpz_clears(R,ri,p,inv,X,Z,x,neg,g,nullptr);st.total_seconds=now_s()-begin;
    return true;
}

void SmallPrimeBabyCache::begin(
    const LadderCtx &C,
    unsigned long long d,
    unsigned long long b1,
    unsigned long long b2,
    const std::vector<unsigned long long> &indices)
{
    key=C;D=d;B1=b1;B2=b2;js=indices;nonunit.clear();ready=false;
}

void SmallPrimeBabyCache::record(size_t index,const mpz_t gcd)
{
    char *v=mpz_get_str(nullptr,10,gcd);nonunit.emplace_back(index,v);
    void (*release)(void*,size_t)=nullptr;mp_get_memory_functions(nullptr,nullptr,&release);
    release(v,std::strlen(v)+1);
}

bool SmallPrimeBabyCache::matches(const LadderCtx &C,const Stage2Params &SP) const
{
    return ready && D==SP.D && B1==SP.B1 && B2==SP.B2 && js==SP.baby_j &&
        key.nw==C.nw && key.ninv==C.ninv && key.hn==C.hn && key.hqx==C.hqx &&
        key.hqz==C.hqz && key.ha24==C.ha24 && key.hmone==C.hmone && key.target_n==C.target_n;
}

void SmallPrimeBabyCache::gcd_at(size_t index,mpz_t g) const
{
    const auto it=std::lower_bound(nonunit.begin(),nonunit.end(),index,
        [](const auto &v,size_t i){return v.first<i;});
    if(it!=nonunit.end() && it->first==index)mpz_set_str(g,it->second.c_str(),10);
    else mpz_set_ui(g,1);
}

size_t SmallPrimeBabyCache::payload_bytes() const
{
    size_t v=js.capacity()*8+(key.hn.capacity()+key.hqx.capacity()+key.hqz.capacity()+
        key.ha24.capacity()+key.hmone.capacity()+key.target_n.capacity())*8+nonunit.capacity()*sizeof(nonunit[0]);
    for(const auto &e:nonunit)v+=e.second.capacity()+1;
    return v; // capacity ledger, not process private bytes or allocator overhead
}
