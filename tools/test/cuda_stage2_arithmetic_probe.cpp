// Independent GMP oracle; links the actual production arithmetic object.
#include "../../src/ecm/stage2/cuda/launch.h"
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using Word=unsigned long long;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool v){if(!v)throw std::runtime_error("production arithmetic differs from GMP");}
static void pack(Word *out,int w,mpz_srcptr x){std::fill(out,out+w,0);mpz_export(out,nullptr,-1,8,0,0,x);}
static void unpack(mpz_ptr x,const Word *in,int w){mpz_import(x,w,-1,8,0,0,in);}
struct Buffer {
    Word *p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*8));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(const std::vector<Word> &v){check(cudaMemcpy(p,v.data(),v.size()*8,cudaMemcpyHostToDevice));}
    void get(std::vector<Word> &v){check(cudaMemcpy(v.data(),p,v.size()*8,cudaMemcpyDeviceToHost));}
};
static unsigned long long cases=0,comparisons=0;
template<int NW>static void run(int w,int bits,bool mersenne,gmp_randstate_t random) {
    constexpr int count=17,group=4;
    mpz_t n,r,rinv,x,y,want,got,inv,radix;
    mpz_inits(n,r,rinv,x,y,want,got,inv,radix,nullptr);
    if(mersenne){mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,1);}
    else {mpz_urandomb(n,random,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    mpz_set_ui(r,1);mpz_mul_2exp(r,r,64*w);require(mpz_invert(rinv,r,n)!=0);
    mpz_set_ui(radix,1);mpz_mul_2exp(radix,radix,64);require(mpz_invert(inv,n,radix)!=0);
    mpz_neg(inv,inv);mpz_fdiv_r_2exp(inv,inv,64);Word ninv=0;mpz_export(&ninv,nullptr,-1,8,0,0,inv);
    std::vector<Word> hn(w),a(count*w),b(a.size()),out(a.size()),expected(a.size()),factor(w);
    pack(hn.data(),w,n);
    for(int k=0;k<count;++k){
        if(k<3){mpz_set_ui(x,k);if(k==2)mpz_sub_ui(x,n,1);}else mpz_urandomm(x,random,n);
        if(k==0)mpz_sub_ui(y,n,1);else mpz_urandomm(y,random,n);
        pack(a.data()+k*w,w,x);pack(b.data()+k*w,w,y);
        mpz_mul(want,x,y);mpz_mul(want,want,rinv);mpz_mod(want,want,n);pack(expected.data()+k*w,w,want);
    }
    Buffer dn(w),da(a.size()),db(b.size()),doo(out.size()),df(w);
    dn.put(hn);da.put(a);db.put(b);
    const int point_bits=mersenne?bits:0;
    check(ecm_stage2_copy_constants(&point_bits,sizeof(point_bits),offsetof(S2GDeviceConstants,point_mersenne_bits)));
    ecm_launch_s2g_mont_test_kernel<NW>({1,128},da.p,db.p,dn.p,ninv,w,count,doo.p);check(cudaGetLastError());doo.get(out);
    require(out==expected);comparisons+=count;
    for(int mode=0;mode<4;++mode){
        if(mode==2)mpz_sub_ui(y,n,1);else mpz_set_ui(y,mode==3?7:mode);
        mpz_mod(y,y,n);mpz_mul(x,y,r);mpz_mod(x,x,n);pack(factor.data(),w,x);df.put(factor);da.put(a);
        for(int k=0;k<count;++k){unpack(x,a.data()+k*w,w);mpz_mul(want,x,y);mpz_mod(want,want,n);pack(expected.data()+k*w,w,want);}
        ecm_launch_s2g_plain_scale_kernel<NW>({1,128},da.p,count,df.p,dn.p,ninv,w);check(cudaGetLastError());da.get(out);
        require(out==expected);comparisons+=count;
    }
    da.put(a);
    ecm_launch_s2g_baby_product_kernel<NW>({1,64},da.p,count,doo.p,dn.p,ninv,w);check(cudaGetLastError());doo.get(out);
    for(int k=0;k<(count+1)/2;++k){
        unpack(x,a.data()+2*k*w,w);
        if(2*k+1<count){unpack(y,a.data()+(2*k+1)*w,w);mpz_mul(x,x,y);mpz_mul(x,x,rinv);mpz_mod(x,x,n);}
        unpack(got,out.data()+k*w,w);require(mpz_cmp(got,x)==0);++comparisons;
    }
    ecm_launch_s2g_block_prod_kernel<NW>({1,128},da.p,count,w,group,dn.p,ninv,doo.p);check(cudaGetLastError());doo.get(out);
    for(int k=0;k<(count+group-1)/group;++k){
        unpack(want,a.data()+k*group*w,w);
        for(int j=k*group+1;j<std::min(count,(k+1)*group);++j){unpack(x,a.data()+j*w,w);mpz_mul(want,want,x);mpz_mul(want,want,rinv);mpz_mod(want,want,n);}
        unpack(got,out.data()+k*w,w);require(mpz_cmp(got,want)==0);++comparisons;
    }
    ++cases;std::printf("PASS bits=%d words=%d capacity=%d mersenne=%d\n",bits,w,NW,int(mersenne));
    mpz_clears(n,r,rinv,x,y,want,got,inv,radix,nullptr);
}
int main(int argc,char **argv){
    try{
        check(cudaSetDevice(argc>1?std::atoi(argv[1]):1));
        gmp_randstate_t random;gmp_randinit_mt(random);gmp_randseed_ui(random,20261020);
        std::vector<int> widths={1,2,3,4,5,7,8,9,16,17,32,33,64,65,127,128,129,255,256};
        if(argc>2){
            if(argc!=3 || std::strcmp(argv[2],"--all-words"))throw std::runtime_error("usage: probe [DEVICE [--all-words]]");
            widths.clear();for(int w=1;w<=256;++w)widths.push_back(w);
        }
        for(int w:widths)
            for(int tail:{1,63,64})for(bool mersenne:{false,true}){
                const int bits=std::max(2,(w-1)*64+tail);
#define ECM_PROBE_ROW(NW,FN,...) if(w<=NW){run<NW>(w,bits,mersenne,random);}else
                ECM_CUDA_STAGE2_CAPACITIES(ECM_PROBE_ROW,unused) {throw std::runtime_error("capacity");}
#undef ECM_PROBE_ROW
            }
        gmp_randclear(random);
        std::printf("{\"complete\":true,\"moduli\":%llu,\"comparisons\":%llu,\"bad\":0}\n",cases,comparisons);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
