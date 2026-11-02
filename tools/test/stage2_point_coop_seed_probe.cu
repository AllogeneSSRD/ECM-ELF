// Actual scalar/cooperative ladder headers; independent GMP polynomial oracle.
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib,"bcrypt.lib")
#include "../../src/ecm/stage2/cuda/device/point.cuh"
#include "../../src/ecm/stage2/cuda/device/point_coop_ladder.cuh"
using Word=unsigned long long;
#ifndef ECM_SEED_WORDS
#define ECM_SEED_WORDS 256
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}while(0)
void pack(Word *p,int w,mpz_srcptr x){std::fill(p,p+w,0);size_t z=0;mpz_export(p,&z,-1,8,0,0,x);}
void unpack(mpz_ptr x,const Word *p,int w){mpz_import(x,w,-1,8,0,0,p);}
std::string digest(const std::vector<Word> &v){
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE h=nullptr;unsigned char b[32];
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0||BCryptCreateHash(alg,&h,nullptr,0,nullptr,0,0)<0||
       BCryptHashData(h,(PUCHAR)v.data(),(ULONG)(v.size()*8),0)<0||BCryptFinishHash(h,b,32,0)<0)exit(2);
    BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(alg,0);char s[65];for(int i=0;i<32;++i)std::sprintf(s+2*i,"%02x",b[i]);return s;
}
int main(int argc,char **argv){
    if(argc!=12){fprintf(stderr,"bits chains per_block first npts kind tpi samples device fault chunk\n");return 2;}
    const int bits=atoi(argv[1]),chains=atoi(argv[2]),per=atoi(argv[3]),kind=atoi(argv[6]),tpi=atoi(argv[7]),
        samples=atoi(argv[8]),device=atoi(argv[9]),fault=atoi(argv[10]),chunk=atoi(argv[11]);
    const Word first=std::strtoull(argv[4],nullptr,10),npts=std::strtoull(argv[5],nullptr,10);
    if(bits<8193||bits>64*ECM_SEED_WORDS||chains<1||chains>8192||per<1||per>64||
       npts<=Word(chains-1)*per||npts>Word(chains)*per||first>~Word(0)-Word(chains-1)*per||
       kind<0||kind>2||(tpi!=0&&tpi!=32)||samples<1||samples>5||fault<0||fault>1||chunk<1||chunk>8192)return 2;
    const int w=(bits+63)/64;CK(cudaSetDevice(device));
    mpz_t n,R,ri,ri3,ri4,a24,a,b,c,d,x,z,px,pz,qx,qz,tx,tz,got;
    mpz_inits(n,R,ri,ri3,ri4,a24,a,b,c,d,x,z,px,pz,qx,qz,tx,tz,got,nullptr);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x571100+bits+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,kind?1:189);
    if(kind==2){mpz_urandomb(n,rng,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    mpz_set_ui(R,1);mpz_mul_2exp(R,R,64*w);mpz_invert(ri,R,n);mpz_powm_ui(ri3,ri,3,n);mpz_powm_ui(ri4,ri,4,n);
    const size_t entries=2*(size_t)chains+1;std::vector<Word> input(5*w),out(2*entries*w);pack(input.data(),w,n);
    for(int j=1;j<4;++j){mpz_urandomm(a,rng,n);pack(input.data()+j*w,w,a);}
    mpz_mod(a,R,n);pack(input.data()+4*w,w,a);
    Word inv=1;for(int j=0;j<6;++j)inv*=2-input[0]*inv;Word ninv=0-inv;
    Word *di,*doo;CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&doo,out.size()*8));
    CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    S2GDeviceConstants constants{};constants.point_square=1;constants.point_mersenne_bits=kind==1?bits:0;
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&constants,sizeof(constants)));
    cudaFuncAttributes attr{};if(tpi)CK(cudaFuncGetAttributes(&attr,stage2_coop_seed_pair<ECM_SEED_WORDS,32>));
    else CK(cudaFuncGetAttributes(&attr,s2g_seed_pair_kernel<256,true>));
    auto run=[&](){for(int offset=0;offset<chains;offset+=chunk){const int count=std::min(chunk,chains-offset);
        if(tpi)stage2_coop_seed_pair<ECM_SEED_WORDS,32><<<(count*32+63)/64,64>>>(di,ninv,w,di+w,di+2*w,di+3*w,di+4*w,
            first,npts,per,chains,offset,count,kind==1?bits:0,doo,doo+entries*w);
        else s2g_seed_pair_kernel<256,true><<<(count+31)/32,32>>>(di,ninv,w,di+w,di+2*w,di+3*w,di+4*w,
            first,npts,per,chains,offset,count,doo,doo+entries*w);CK(cudaGetLastError());}};
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));std::vector<float> times;
    for(int j=0;j<samples;++j){float ms;CK(cudaEventRecord(begin));run();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,begin,end));times.push_back(ms);}
    CK(cudaMemcpy(out.data(),doo,out.size()*8,cudaMemcpyDeviceToHost));if(fault)out[0]^=1;
    unpack(px,input.data()+w,w);unpack(pz,input.data()+2*w,w);unpack(a24,input.data()+3*w,w);mpz_mul(a24,a24,ri);mpz_mod(a24,a24,n);
    // Independent GMP polynomial formulas retain the original Montgomery scale.
    auto dbl=[&](mpz_ptr ox,mpz_ptr oz,mpz_srcptr ix,mpz_srcptr iz){mpz_add(a,ix,iz);mpz_mul(a,a,a);mpz_mod(a,a,n);
        mpz_sub(b,ix,iz);mpz_mul(b,b,b);mpz_mod(b,b,n);mpz_sub(c,a,b);mpz_mul(tx,a,b);mpz_mul(tx,tx,ri3);mpz_mod(tx,tx,n);
        mpz_mul(d,a24,c);mpz_add(d,d,b);mpz_mul(tz,c,d);mpz_mul(tz,tz,ri3);mpz_mod(tz,tz,n);mpz_set(ox,tx);mpz_set(oz,tz);};
    auto add=[&](mpz_ptr ox,mpz_ptr oz){mpz_mul(a,x,qx);mpz_mul(b,z,qz);mpz_sub(a,a,b);mpz_mul(a,a,a);
        mpz_mul(tx,pz,a);mpz_mul(tx,tx,ri4);mpz_mod(tx,tx,n);mpz_mul(a,x,qz);mpz_mul(b,z,qx);mpz_sub(a,a,b);mpz_mul(a,a,a);
        mpz_mul(tz,px,a);mpz_mul(tz,tz,ri4);mpz_mod(tz,tz,n);mpz_set(ox,tx);mpz_set(oz,tz);};
    const int checks=chains>37?17:chains;size_t bad=0;
    for(int sample=0;sample<checks;++sample){const int i=checks==chains?sample:(int)((size_t)(chains-1)*sample/(checks-1));const Word k=first+Word(i)*per;
        if(!k){mpz_mod(x,R,n);mpz_set_ui(z,0);mpz_set(qx,px);mpz_set(qz,pz);}
        else{mpz_set(x,px);mpz_set(z,pz);dbl(qx,qz,px,pz);int top=63;while(!(k>>top&1))--top;
            for(int bit=top-1;bit>=0;--bit){if(k>>bit&1){add(x,z);dbl(qx,qz,qx,qz);}else{add(qx,qz);dbl(x,z,x,z);}}}
        const bool single=Word(i)*per+1>=npts;
        for(int j=0;j<4;++j){const auto expected=j==0?x:j==1?z:j==2?(single?x:qx):(single?z:qz);
            const size_t at=((j&1)?entries:0)+2*(size_t)i+(j>=2);unpack(got,out.data()+at*w,w);bad+=mpz_cmp(got,expected)!=0;}}
    unpack(got,out.data()+2*(size_t)chains*w,w);bad+=mpz_cmp(got,px)!=0;
    unpack(got,out.data()+(entries+2*(size_t)chains)*w,w);bad+=mpz_cmp(got,pz)!=0;
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    printf("point_coop_seed: bits=%d nw=%d capacity_bits=%d chains=%d per_block=%d first=%llu npts=%llu kind=%d tpi=%d chunk=%d bad=%zu checked_words=%zu output_words=%zu median_ms=%.6f registers=%d local_bytes=%zu sha256=%s samples_ms=",
        bits,w,tpi?64*ECM_SEED_WORDS:16384,chains,per,first,npts,kind,tpi,chunk,bad,(size_t)(checks*4+2)*w,out.size(),sorted[sorted.size()/2],attr.numRegs,attr.localSizeBytes,digest(out).c_str());
    for(size_t j=0;j<times.size();++j)printf("%s%.6f",j?",":"",times[j]);puts("");
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));CK(cudaFree(di));CK(cudaFree(doo));
    mpz_clears(n,R,ri,ri3,ri4,a24,a,b,c,d,x,z,px,pz,qx,qz,tx,tz,got,nullptr);gmp_randclear(rng);return bad?1:0;
}
