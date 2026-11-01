// Full xADD chains, exact scalar coordinate scale, fixed 16k CGBN capacity.
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
#include "../../src/ecm/stage2/cuda/device/giant_chain.cuh"
#include "../../src/ecm/stage2/cuda/device/point_coop_chain.cuh"
using Word=unsigned long long;
#define CK(x) do{const auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}while(0)
void pack(Word *p,int nw,mpz_srcptr x){std::fill(p,p+nw,0);size_t z=0;mpz_export(p,&z,-1,8,0,0,x);}
void unpack(mpz_ptr x,const Word *p,int nw){mpz_import(x,nw,-1,8,0,0,p);}
std::string digest(const std::vector<Word> &v) {
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;unsigned char bytes[32];
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0 ||
       BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0)<0 ||
       BCryptHashData(hash,(PUCHAR)v.data(),(ULONG)(v.size()*8),0)<0 ||
       BCryptFinishHash(hash,bytes,32,0)<0)exit(2);
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
    char text[65];for(int i=0;i<32;++i)std::sprintf(text+2*i,"%02x",bytes[i]);return text;
}
int main(int argc,char **argv) {
    if(argc!=9&&argc!=11){fprintf(stderr,"bits chains length tpi threads device kind samples [tail stride]\n");return 2;}
    const int bits=atoi(argv[1]),chains=atoi(argv[2]),length=atoi(argv[3]),tpi=atoi(argv[4]),
        threads=atoi(argv[5]),device=atoi(argv[6]),kind=atoi(argv[7]),samples=atoi(argv[8]);
    if(bits<8193||bits>16384||chains<1||chains>8192||length<2||length>64||
       (tpi!=0&&tpi!=16)||(threads!=32&&threads!=64&&threads!=128)||kind<0||kind>2||samples<1||samples>9)return 2;
    const int tail=argc==11?atoi(argv[9]):length,stride=argc==11?atoi(argv[10]):1;
    if(tail<1||tail>length||(stride!=1&&stride!=2))return 2;
    const int nw=(bits+63)/64;const size_t points=(size_t)(chains-1)*length+tail;
    if(2*points*nw*8>1024ull*1024*1024)return 2;
    CK(cudaSetDevice(device));mpz_t n,a,rinv,r,px[6],next[2],u,v,got;
    mpz_inits(n,a,rinv,r,u,v,got,nullptr);for(auto &x:px)mpz_init(x);for(auto &x:next)mpz_init(x);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x987654u+bits+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,kind==0?189:1);
    if(kind==2){mpz_urandomb(n,rng,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    mpz_set_ui(r,1);mpz_mul_2exp(r,r,64*nw);if(!mpz_invert(rinv,r,n))return 2;
    std::vector<Word> input((size_t)(3+4*chains)*nw),output(2*points*nw);pack(input.data(),nw,n);
    for(size_t i=nw;i<input.size();i+=nw){mpz_urandomm(a,rng,n);pack(input.data()+i,nw,a);}
    Word inv=1;for(int i=0;i<6;++i)inv*=2-input[0]*inv;const Word ninv=0-inv;
    Word *di,*doo;CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&doo,output.size()*8));CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    S2GDeviceConstants constants{};constants.point_square=1;constants.point_mersenne_bits=kind==1?bits:0;
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&constants,sizeof(constants)));
    auto *dn=di,*dx=di+nw,*dz=di+2*nw,*sx=di+3*nw;
    const size_t szoff=3+(size_t)chains*stride,exoff=stride==2?4:3+(size_t)chains*2,
        ezoff=stride==2?szoff+1:3+(size_t)chains*3;
    auto *sz=di+szoff*nw,*ex=di+exoff*nw,*ez=di+ezoff*nw;
    cudaFuncAttributes attr{};
    if(tpi)CK(cudaFuncGetAttributes(&attr,stage2_coop_chain<256,16>));
    else CK(cudaFuncGetAttributes(&attr,s2g_chain_kernel<256,true>));
    auto run=[&](){
        if(tpi)stage2_coop_chain<256,16><<<(chains*tpi+threads-1)/threads,threads>>>(dn,ninv,nw,dx,dz,sx,sz,ex,ez,chains,points,length,stride,kind==1?bits:0,doo,doo+points*nw);
        else s2g_chain_kernel<256,true><<<(chains+threads-1)/threads,threads>>>(dn,ninv,nw,dx,dz,sx,sz,ex,ez,points,length,chains,doo,doo+points*nw,stride);
        CK(cudaGetLastError());
    };
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));std::vector<float> timings;
    for(int i=0;i<samples;++i){float ms;CK(cudaEventRecord(begin));run();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,begin,end));timings.push_back(ms);}
    CK(cudaMemcpy(output.data(),doo,output.size()*8,cudaMemcpyDeviceToHost));size_t checked=0,bad=0;
    auto mul=[&](mpz_ptr dst,mpz_srcptr x,mpz_srcptr y){mpz_mul(dst,x,y);mpz_mul(dst,dst,rinv);mpz_mod(dst,dst,n);};
    for(int c=0;c<chains;++c) {
        if(chains>37 && c!=0 && c!=chains-1)continue;
        unpack(px[0],input.data()+(3+(size_t)c*stride)*nw,nw);unpack(px[1],input.data()+(szoff+(size_t)c*stride)*nw,nw);
        unpack(px[2],input.data()+(exoff+(size_t)c*stride)*nw,nw);unpack(px[3],input.data()+(ezoff+(size_t)c*stride)*nw,nw);
        unpack(px[4],input.data()+nw,nw);unpack(px[5],input.data()+2*nw,nw);
        for(int k=0;k<(c==chains-1?tail:length);++k) {
            if(k>=2) {
                mul(u,px[2],px[4]);mul(v,px[3],px[5]);mpz_sub(u,u,v);mul(u,u,u);mul(next[0],px[1],u);
                mul(u,px[2],px[5]);mul(v,px[3],px[4]);mpz_sub(u,u,v);mul(u,u,u);mul(next[1],px[0],u);
                mpz_set(px[0],px[2]);mpz_set(px[1],px[3]);mpz_set(px[2],next[0]);mpz_set(px[3],next[1]);
            }
            for(int z=0;z<2;++z){unpack(got,output.data()+((size_t)z*points+(size_t)c*length+k)*nw,nw);bad+=mpz_cmp(got,px[(k==0?0:2)+z])!=0;checked+=nw;}
        }
    }
    auto sorted=timings;std::sort(sorted.begin(),sorted.end());
    printf("point_coop_chain: bits=%d nw=%d chains=%d length=%d tail=%d seed_stride=%d tpi=%d threads=%d kind=%d bad=%zu checked_words=%zu output_words=%zu median_ms=%.6f registers=%d local_bytes=%zu sha256=%s samples_ms=",
        bits,nw,chains,length,tail,stride,tpi,threads,kind,bad,checked,output.size(),sorted[sorted.size()/2],attr.numRegs,attr.localSizeBytes,digest(output).c_str());
    for(size_t i=0;i<timings.size();++i)printf("%s%.6f",i?",":"",timings[i]);puts("");
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));CK(cudaFree(di));CK(cudaFree(doo));
    for(auto &x:px)mpz_clear(x);for(auto &x:next)mpz_clear(x);mpz_clears(n,a,rinv,r,u,v,got,nullptr);gmp_randclear(rng);return bad?1:0;
}
