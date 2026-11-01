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
#ifndef ECM_LADDER_WORDS
#define ECM_LADDER_WORDS 256
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
    if(argc!=10){fprintf(stderr,"bits points tpi scalar_bits kind normal samples device fault\n");return 2;}
    const int bits=atoi(argv[1]),points=atoi(argv[2]),tpi=atoi(argv[3]),scalar_bits=atoi(argv[4]),kind=atoi(argv[5]),
        normal=atoi(argv[6]),samples=atoi(argv[7]),device=atoi(argv[8]),fault=atoi(argv[9]);
    if(bits<8193||bits>64*ECM_LADDER_WORDS||points<1||points>8192||(tpi!=0&&tpi!=32)||scalar_bits<0||scalar_bits>64||
       kind<0||kind>2||normal<0||normal>1||samples<1||samples>5||fault<0||fault>1)return 2;
    const int w=(bits+63)/64;CK(cudaSetDevice(device));
    mpz_t n,R,ri,ri3,ri4,a24,a,b,c,d,x,z,xx,zz,px,pz,qx,qz,tx,tz,got;
    mpz_inits(n,R,ri,ri3,ri4,a24,a,b,c,d,x,z,xx,zz,px,pz,qx,qz,tx,tz,got,nullptr);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x579500+bits+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,kind?1:189);
    if(kind==2){mpz_urandomb(n,rng,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    mpz_set_ui(R,1);mpz_mul_2exp(R,R,64*w);mpz_invert(ri,R,n);mpz_powm_ui(ri3,ri,3,n);mpz_powm_ui(ri4,ri,4,n);
    std::vector<Word> input(5*w),out(2*(size_t)points*w),scalars(points);pack(input.data(),w,n);
    for(int j=1;j<4;++j){mpz_urandomm(a,rng,n);pack(input.data()+j*w,w,a);}
    mpz_mod(a,R,n);pack(input.data()+4*w,w,a);
    Word state=0x9513572468abcdefull;auto next=[&](){state^=state<<13;state^=state>>7;state^=state<<17;return state;};
    const Word edges[]={0,1,2,3,4,31,32,33,63,64,65,65535,65536,65537,0xffffffffull,0x100000000ull,
        0x100000001ull,0x7fffffffffffffffull,0x8000000000000000ull,~0ull};
    for(int i=0;i<points;++i){scalars[i]=next();if(!scalar_bits&&i<20)scalars[i]=edges[i];
        else if(scalar_bits&&scalar_bits<64)scalars[i]=(scalars[i]&((1ull<<scalar_bits)-1))|(1ull<<(scalar_bits-1));}
    Word inv=1;for(int j=0;j<6;++j)inv*=2-input[0]*inv;Word ninv=0-inv;
    Word *di,*ds,*doo;CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&ds,points*8));CK(cudaMalloc(&doo,out.size()*8));
    CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,scalars.data(),points*8,cudaMemcpyHostToDevice));
    S2GDeviceConstants constants{};constants.point_square=1;constants.point_mersenne_bits=kind==1?bits:0;
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&constants,sizeof(constants)));
    cudaFuncAttributes attr{};if(tpi)CK(cudaFuncGetAttributes(&attr,stage2_coop_ladder<ECM_LADDER_WORDS,32>));
    else CK(cudaFuncGetAttributes(&attr,s2g_ladder_kernel<256,true>));
    auto run=[&](){if(tpi)stage2_coop_ladder<ECM_LADDER_WORDS,32><<<(points*32+63)/64,64>>>(di,ninv,w,di+w,di+2*w,di+3*w,di+4*w,ds,points,kind==1?bits:0,normal!=0,doo,doo+(size_t)points*w);
        else s2g_ladder_kernel<256,true><<<(points+63)/64,64>>>(di,ninv,w,di+w,di+2*w,di+3*w,di+4*w,ds,points,doo,doo+(size_t)points*w,normal!=0);CK(cudaGetLastError());};
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));std::vector<float> times;
    for(int j=0;j<samples;++j){float ms;CK(cudaEventRecord(begin));run();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,begin,end));times.push_back(ms);}
    CK(cudaMemcpy(out.data(),doo,out.size()*8,cudaMemcpyDeviceToHost));if(fault)out[0]^=1;
    unpack(px,input.data()+w,w);unpack(pz,input.data()+2*w,w);unpack(a24,input.data()+3*w,w);mpz_mul(a24,a24,ri);mpz_mod(a24,a24,n);
    auto dbl=[&](mpz_ptr ox,mpz_ptr oz,mpz_srcptr ix,mpz_srcptr iz){mpz_add(a,ix,iz);mpz_mul(a,a,a);mpz_mod(a,a,n);
        mpz_sub(b,ix,iz);mpz_mul(b,b,b);mpz_mod(b,b,n);mpz_sub(c,a,b);mpz_mul(tx,a,b);mpz_mul(tx,tx,ri3);mpz_mod(tx,tx,n);
        mpz_mul(d,a24,c);mpz_add(d,d,b);mpz_mul(tz,c,d);mpz_mul(tz,tz,ri3);mpz_mod(tz,tz,n);mpz_set(ox,tx);mpz_set(oz,tz);};
    auto add=[&](mpz_ptr ox,mpz_ptr oz){mpz_mul(a,x,qx);mpz_mul(b,z,qz);mpz_sub(a,a,b);mpz_mul(a,a,a);
        mpz_mul(tx,pz,a);mpz_mul(tx,tx,ri4);mpz_mod(tx,tx,n);mpz_mul(a,x,qz);mpz_mul(b,z,qx);mpz_sub(a,a,b);mpz_mul(a,a,a);
        mpz_mul(tz,px,a);mpz_mul(tz,tz,ri4);mpz_mod(tz,tz,n);mpz_set(ox,tx);mpz_set(oz,tz);};
    // Correctness checks every coordinate. Large throughput calls use fixed
    // distributed samples; the complete output SHA is still compared across
    // backends, without repeating an expensive full CPU ladder for each call.
    const int checks=scalar_bits&&points>37?17:points;size_t bad=0;
    for(int sample=0;sample<checks;++sample){const int i=checks==points?sample:(int)((size_t)(points-1)*sample/(checks-1));const Word k=scalars[i];
        if(!k){mpz_mod(x,R,n);mpz_set_ui(z,0);}else{mpz_set(x,px);mpz_set(z,pz);dbl(qx,qz,px,pz);
            int top=63;while(!(k>>top&1))--top;for(int bit=top-1;bit>=0;--bit){if(k>>bit&1){add(x,z);dbl(qx,qz,qx,qz);}else{add(qx,qz);dbl(x,z,x,z);}}}
        if(normal){mpz_mul(x,x,ri);mpz_mod(x,x,n);mpz_mul(z,z,ri);mpz_mod(z,z,n);}
        unpack(got,out.data()+(size_t)i*w,w);bad+=mpz_cmp(got,x)!=0;unpack(got,out.data()+((size_t)points+i)*w,w);bad+=mpz_cmp(got,z)!=0;}
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    printf("point_coop_ladder: bits=%d nw=%d capacity_bits=%d points=%d scalar_bits=%d kind=%d normal=%d tpi=%d bad=%zu checked_words=%zu output_words=%zu median_ms=%.6f registers=%d local_bytes=%zu sha256=%s samples_ms=",
        bits,w,tpi?64*ECM_LADDER_WORDS:16384,points,scalar_bits,kind,normal,tpi,bad,(size_t)checks*2*w,out.size(),sorted[sorted.size()/2],attr.numRegs,attr.localSizeBytes,digest(out).c_str());
    for(size_t j=0;j<times.size();++j)printf("%s%.6f",j?",":"",times[j]);puts("");
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));CK(cudaFree(di));CK(cudaFree(ds));CK(cudaFree(doo));
    mpz_clears(n,R,ri,ri3,ri4,a24,a,b,c,d,x,z,xx,zz,px,pz,qx,qz,tx,tz,got,nullptr);gmp_randclear(rng);return bad?1:0;
}
