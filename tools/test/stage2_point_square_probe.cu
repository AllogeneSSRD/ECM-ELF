// Standalone probe of the real production primitives, not a copied arithmetic
// implementation. Recurrences measure both many-instance throughput and serial
// dependencies; GMP checks every returned word, including in-place results.
#include <cuda_runtime.h>
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../src/ecm/stage2/cuda/device/point.cuh"
using Word=unsigned long long;
#define CK(x) do{const auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<int NW,int MODE>
__global__ void square_probe(const Word *input,const Word *n,Word *out,int nw,
    int count,int repeats,int alias,int fault,Word ninv)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Word x[NW],v[NW];
    for(int j=0;j<nw;++j)x[j]=input[(size_t)i*nw+j];
    for(int k=0;k<repeats;++k) {
        auto *dst=alias?x:v;
        if constexpr(MODE==1)s2g_mont_sqr<NW>(dst,x,n,ninv,nw);
        else if constexpr(MODE==2)s2g_point_square<NW>(dst,x,n,ninv,nw);
        else s2g_mont_mul<NW>(dst,x,x,n,ninv,nw);
        if(!alias)for(int j=0;j<nw;++j)x[j]=v[j];
    }
    for(int j=0;j<nw;++j)out[(size_t)i*nw+j]=x[j];
    if(MODE && fault && i==0)out[0]^=1;
}
template<int NW>void launch(int mode,int threads,const Word *in,const Word *n,Word *out,
    int nw,int count,int repeats,int alias,int fault,Word ninv,cudaFuncAttributes &attr,bool prepare)
{
    if(mode==1) {
        if(prepare){CK(cudaFuncGetAttributes(&attr,square_probe<NW,1>));return;}
        square_probe<NW,1><<<(count+threads-1)/threads,threads>>>(in,n,out,nw,count,repeats,alias,fault,ninv);
    } else if(mode==2) {
        if(prepare){CK(cudaFuncGetAttributes(&attr,square_probe<NW,2>));return;}
        square_probe<NW,2><<<(count+threads-1)/threads,threads>>>(in,n,out,nw,count,repeats,alias,fault,ninv);
    } else {
        if(prepare){CK(cudaFuncGetAttributes(&attr,square_probe<NW,0>));return;}
        square_probe<NW,0><<<(count+threads-1)/threads,threads>>>(in,n,out,nw,count,repeats,alias,fault,ninv);
    }
}
void pack(Word *p,int w,mpz_srcptr v){std::fill(p,p+w,0);size_t z=0;mpz_export(p,&z,-1,8,0,0,v);}
int main(int argc,char **argv)
{
    if(argc!=10){fprintf(stderr,"bits count repeats mode alias threads device modulus_kind fault\n");return 2;}
    const int bits=atoi(argv[1]),count=atoi(argv[2]),repeats=atoi(argv[3]),mode=atoi(argv[4]),alias=atoi(argv[5]),
        threads=atoi(argv[6]),device=atoi(argv[7]),kind=atoi(argv[8]),fault=atoi(argv[9]);
    if(bits<2||bits>16384||count<1||count>131072||repeats<1||repeats>64||mode<0||mode>2||
       alias<0||alias>1||(threads!=32&&threads!=64&&threads!=128)||kind<0||kind>2||fault<0||fault>1)return 2;
    CK(cudaSetDevice(device));const int nw=(bits+63)/64;const size_t size=(size_t)count*nw;
    mpz_t n,a,rinv,t;mpz_inits(n,a,rinv,t,nullptr);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x20261101u+bits+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,bits>=9&&kind==0?189:1);
    if(kind==2){mpz_urandomb(n,rng,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    mpz_set_ui(t,1);mpz_mul_2exp(t,t,64*nw);if(!mpz_invert(rinv,t,n))return 2;
    std::vector<Word> hn(nw),input(size),expected(size),got(size);pack(hn.data(),nw,n);
    Word inv=1;for(int j=0;j<6;++j)inv*=2-hn[0]*inv;const Word ninv=0-inv;
    for(int i=0;i<count;++i) {
        mpz_urandomm(a,rng,n);
        if(i<4){if(i<2)mpz_set_ui(a,i);else mpz_sub_ui(a,n,i-1);}
        else if(i<20){mpz_set_ui(a,1);mpz_mul_2exp(a,a,(bits-1)-(i-4)%(bits-1));mpz_mod(a,a,n);}
        pack(input.data()+(size_t)i*nw,nw,a);
        for(int k=0;k<repeats;++k){mpz_mul(a,a,a);mpz_mul(a,a,rinv);mpz_mod(a,a,n);}
        pack(expected.data()+(size_t)i*nw,nw,a);
    }
    S2GDeviceConstants constants{};constants.point_mersenne_bits=kind==1?bits:0;constants.point_square=mode==2;
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&constants,sizeof(constants)));
    Word *di,*dn,*dout;CK(cudaMalloc(&di,size*8));CK(cudaMalloc(&dn,nw*8));CK(cudaMalloc(&dout,size*8));
    CK(cudaMemcpy(di,input.data(),size*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dn,hn.data(),nw*8,cudaMemcpyHostToDevice));
    cudaFuncAttributes attr{};
    auto run=[&](bool prepare=false){
#define ROW(NW) if(nw<=NW)launch<NW>(mode,threads,di,dn,dout,nw,count,repeats,alias,fault,ninv,attr,prepare);else
        ROW(4) ROW(8) ROW(16) ROW(32) ROW(64) ROW(128) ROW(256) {exit(2);}
#undef ROW
        CK(cudaGetLastError());
    };
    run(true); // Query/load kernel metadata before timing, without executing it.
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));std::vector<float> samples;
    // Retain all five measurements, including the first call: no dropped warmup.
    for(int k=0;k<5;++k){float ms=0;CK(cudaEventRecord(begin));run();CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms);}
    CK(cudaMemcpy(got.data(),dout,size*8,cudaMemcpyDeviceToHost));size_t bad=0,first=0;
    for(size_t j=0;j<size;++j)if(got[j]!=expected[j]){if(!bad)first=j;++bad;}
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    printf("point_square: bits=%d nw=%d kind=%d count=%d repeats=%d mode=%d alias=%d threads=%d device=%d "
           "bad=%zu first_bad=%zu median_ms=%.6f min_ms=%.6f max_ms=%.6f registers=%d local_bytes=%zu samples_ms=",
           bits,nw,kind,count,repeats,mode,alias,threads,device,bad,first,sorted[2],sorted.front(),sorted.back(),attr.numRegs,attr.localSizeBytes);
    for(size_t j=0;j<samples.size();++j)printf("%s%.6f",j?",":"",samples[j]);puts("");
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));CK(cudaFree(di));CK(cudaFree(dn));CK(cudaFree(dout));
    mpz_clears(n,a,rinv,t,nullptr);gmp_randclear(rng);return bad?1:0;
}
