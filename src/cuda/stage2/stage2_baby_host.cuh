#pragma once
#include "stage2_baby_device.cuh"

struct DeviceBabyStats {
    size_t points=0,groups=0,bad_groups=0,payload_bytes=0,coordinate_d2h_bytes=0,
        root_d2h_bytes=0,leaf_d2h_bytes=0,seed_h2d_bytes=0,checked_words=0;
    double ladder_seconds=0,invert_seconds=0,total_seconds=0;
};
struct BabyBuffer {
    void *ptr=nullptr;
    ~BabyBuffer(){if(ptr)CK(cudaFree(ptr));}
    bool allocate(size_t bytes) {
        const auto code=cudaMalloc(&ptr,bytes);
        if(code==cudaErrorMemoryAllocation){cudaGetLastError();ptr=nullptr;return false;}
        CK(code);return true;
    }
    unsigned long long *words(){return static_cast<unsigned long long*>(ptr);}
};

// All buffers end their lifetime before F-tree construction. An allocation failure
// returns to the original ladder/normalizer before modifying leaves or factor state.
static bool device_baby_generate(const LadderCtx &C,const mpz_t N,
    const std::vector<unsigned long long> &indices,
    std::vector<std::vector<unsigned long long>> &leaf,SmallPrimeBabyCache *cache,
    std::vector<std::string> &factors,unsigned long long &noninv,DeviceBabyStats &st,bool force_check=false)
{
    const double begin=now_s();const size_t n=indices.size(),w=C.nw;
    if(!n)return false;
    size_t counts[9]={n},offset[9]={},words=0;
    for(int l=1;l<=8;++l){counts[l]=(counts[l-1]+1)/2;offset[l]=words;words+=counts[l]*w;}
    const size_t groups=counts[8];
    const size_t bytes=8*((3*n+5)*w+n+words)+groups;
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

// Exercise the actual host caller with zero and nonunit coordinates, not merely
// the arithmetic kernels. Synthetic composite and curve states stay in the fixture.
static void device_baby_fixture(size_t w)
{
    mpz_t N,R,v;mpz_inits(N,R,v,nullptr);
    mpz_set_ui(N,1);mpz_mul_2exp(N,N,64*w-4);mpz_sub_ui(N,N,1);mpz_mul_ui(N,N,3);
    mpz_set_ui(R,1);mpz_mul_2exp(R,R,64*w);mpz_mod(R,R,N);
    LadderCtx c;c.nw=w;mpz_to_words(c.hn,w,N);mpz_to_words(c.hmone,w,R);
    unsigned long long inverse=1;for(int i=0;i<6;++i)inverse*=2-c.hn[0]*inverse;c.ninv=0-inverse;
    for(unsigned long long qz:{1ull,3ull}) {
        auto image=[&](std::vector<unsigned long long> &out,unsigned long long x){
            mpz_mul_ui(v,R,x);mpz_mod(v,v,N);mpz_to_words(out,w,v);};
        image(c.hqx,5);image(c.hqz,qz);image(c.ha24,3);
        std::vector<unsigned long long> js(513);for(size_t i=0;i<js.size();++i)js[i]=i;
        SmallPrimeBabyCache cache;cache.begin(c,210,20,1000,js);
        std::vector<std::vector<unsigned long long>> leaves(js.size());
        std::vector<std::string> factors;unsigned long long noninv=0;DeviceBabyStats st;
        if(!device_baby_generate(c,N,js,leaves,&cache,factors,noninv,st,true) ||
           st.checked_words!=js.size()*w || factors.size()!=noninv || !st.bad_groups) {
            std::fprintf(stderr,"FATAL: baby device host fixture incomplete\n");std::exit(3);
        }
        for(const auto &leaf:leaves) {
            if(leaf.size()!=2*w || leaf[w]!=1 ||
               std::any_of(leaf.begin()+w+1,leaf.end(),[](auto x){return x!=0;})) {
                std::fprintf(stderr,"FATAL: baby leaf is not monic\n");std::exit(3);
            }
        }
        stage2_log::print(stage2_log::debug, "baby_device_fixture: words=%llu qz=%llu checked_words=%llu noninv=%llu bad_groups=%llu bad=0\n",
            (unsigned long long)w,qz,(unsigned long long)st.checked_words,noninv,(unsigned long long)st.bad_groups);
    }
    mpz_clears(N,R,v,nullptr);
}
