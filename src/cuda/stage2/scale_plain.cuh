#pragma once

// Included after s2g_mont_mul and S2G_DISPATCH. Coefficients stay in the plain
// domain: Mont(h, factor*R mod N) = h*factor mod N, R = 2^(64*nw).
template<int NW>
__global__ void s2g_plain_scale_kernel(unsigned long long *h, size_t count,
    const unsigned long long *factor_mont, const unsigned long long *n,
    unsigned long long ninv, int nw)
{
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)s2g_mont_mul<NW>(h+i*nw,h+i*nw,factor_mont,n,ninv,nw);
}

template<int NW>
static void s2g_launch_plain_scale(int nw,size_t count,unsigned long long *h,
    const unsigned long long *factor,const unsigned long long *n,unsigned long long ninv)
{
    s2g_plain_scale_kernel<NW><<<(unsigned)((count+127)/128),128>>>(h,count,factor,n,ninv,nw);
}

struct PlainScaleStats {
    bool requested=false,enabled=false;
    unsigned long long coefficients=0,h2d_bytes=0,check_d2h_bytes=0,checked_words=0;
    double seconds=0;
    const char *fallback="none";
};

// scratch is a dead W-word slot in a separate owner allocation. No allocation
// on the normal path; diagnostic snapshots are host vectors, counted separately.
static void s2g_plain_scale(size_t W,size_t count,unsigned long long *h,
    unsigned long long *scratch,const unsigned long long *dn,mpz_srcptr N,
    mpz_srcptr factor,PlainScaleStats &st,bool check,bool poison)
{
    if(!count)return;
    if(poison&&!check){std::fprintf(stderr,"FATAL: Gamma poison requires full GMP check\n");std::exit(3);}
    const double begin=now_s();
    std::vector<unsigned long long> before,after,scalar(W),hn(W);
    if(check){before.resize(count*W);CK(cudaMemcpy(before.data(),h,before.size()*8,cudaMemcpyDeviceToHost));st.check_d2h_bytes+=before.size()*8;}
    mpz_t z;mpz_init(z);mpz_mul_2exp(z,factor,(mp_bitcnt_t)(64*W));mpz_mod(z,z,N);
    mpz_to_words(scalar,W,z);mpz_to_words(hn,W,N);
    unsigned long long inverse=1;
    for(int i=0;i<6;++i)inverse*=2-hn[0]*inverse;
    CK(cudaMemcpy(scratch,scalar.data(),W*8,cudaMemcpyHostToDevice));st.h2d_bytes+=W*8;
    S2G_DISPATCH((int)W,s2g_launch_plain_scale,(int)W,count,h,scratch,dn,0ull-inverse);
    CK(cudaGetLastError());
    if(poison){
        unsigned long long word=0;CK(cudaMemcpy(&word,h,8,cudaMemcpyDeviceToHost));word^=1;
        CK(cudaMemcpy(h,&word,8,cudaMemcpyHostToDevice));st.check_d2h_bytes+=8;st.h2d_bytes+=8;
    }
    // Finish before the owner can be released or its scratch reused. Diagnostic
    // copies are deliberately outside formal performance samples.
    CK(cudaStreamSynchronize(0));
    if(check){
        after.resize(count*W);CK(cudaMemcpy(after.data(),h,after.size()*8,cudaMemcpyDeviceToHost));st.check_d2h_bytes+=after.size()*8;
        for(size_t i=0;i<count;++i){
            words_to_mpz(z,before.data()+i*W,W);mpz_mul(z,z,factor);mpz_mod(z,z,N);mpz_to_words(scalar,W,z);
            if(!std::equal(scalar.begin(),scalar.end(),after.data()+i*W)){
                std::fprintf(stderr,"FATAL: Gamma device GMP mismatch coefficient=%llu\n",(unsigned long long)i);std::exit(3);
            }
        }
        st.checked_words+=count*W;
    }
    mpz_clear(z);st.enabled=true;st.coefficients+=count;st.seconds+=now_s()-begin;
}

static void s2g_plain_scale_fixture(mpz_srcptr N,size_t W)
{
    unsigned long long *h=nullptr,*scalar=nullptr,*dn=nullptr;
    CK(cudaMalloc(&h,129*W*8));CK(cudaMalloc(&scalar,W*8));CK(cudaMalloc(&dn,W*8));
    std::vector<unsigned long long> data(129*W),word(W);
    mpz_to_words(word,W,N);CK(cudaMemcpy(dn,word.data(),W*8,cudaMemcpyHostToDevice));
    mpz_t z,factor;mpz_inits(z,factor,nullptr);
    unsigned long long seed=0x35b94ae670dc182full,cases=0,words=0;
    for(size_t count:{1u,2u,129u})for(unsigned mode=0;mode<4;++mode){
        for(size_t i=0;i<count;++i){
            if(i==0)mpz_set_ui(z,0);
            else if(i==1)mpz_set_ui(z,1);
            else if(i==2)mpz_sub_ui(z,N,1);
            else{
                for(size_t j=0;j<W;++j){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;word[j]=seed;}
                words_to_mpz(z,word.data(),W);mpz_mod(z,z,N);
            }
            mpz_to_words(word,W,z);std::copy(word.begin(),word.end(),data.data()+i*W);
        }
        if(mode==2)mpz_sub_ui(factor,N,1);else mpz_set_ui(factor,mode==3?7:mode);
        CK(cudaMemcpy(h,data.data(),count*W*8,cudaMemcpyHostToDevice));
        PlainScaleStats st;s2g_plain_scale(W,count,h,scalar,dn,N,factor,st,true,false);
        ++cases;words+=st.checked_words;
    }
    mpz_clears(z,factor,nullptr);CK(cudaFree(dn));CK(cudaFree(scalar));CK(cudaFree(h));
    std::printf("gscale_device_fixture: cases=%llu words=%llu bad=0 (plain GMP, factors 0/1/N-1/7, counts 1/2/129)\n",cases,words);
}
