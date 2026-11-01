// Cooperative arithmetic against the actual scalar production header. No
// curve engine replacement: first qualify the field and its latency/throughput.
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../src/ecm/stage2/cuda/device/point.cuh"
#include "../../src/ecm/stage2/cuda/device/point_coop_field.cuh"
using Word=unsigned long long;
#ifndef ECM_COOP_TPI
#define ECM_COOP_TPI 16
#endif
#ifndef ECM_COOP_MAX_WORDS
#define ECM_COOP_MAX_WORDS 256
#endif
#define CK(x) do{const auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}while(0)

template<int NW>
__global__ void scalar_probe(const Word *input,const Word *other,const Word *n,Word *out,
    int nw,int count,int repeats,int operation,int alias,int fault,Word ninv)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Word x[NW],y[NW],v[NW];
    for(int j=0;j<nw;++j){x[j]=input[(size_t)i*nw+j];y[j]=other[(size_t)i*nw+j];}
    for(int k=0;k<repeats;++k) {
        auto *dst=alias?x:v;
        if(operation)s2g_point_square<NW>(dst,x,n,ninv,nw);
        else s2g_mont_mul<NW>(dst,x,y,n,ninv,nw);
        if(!alias)for(int j=0;j<nw;++j)x[j]=v[j];
    }
    for(int j=0;j<nw;++j)out[(size_t)i*nw+j]=x[j];
    if(fault && i==0)out[0]^=1;
}

template<int BITS,int TPI>
__global__ void cooperative_probe(cgbn_mem_t<BITS> *input,cgbn_mem_t<BITS> *other,
    cgbn_mem_t<BITS> *modulus,cgbn_mem_t<BITS> *out,int count,int repeats,
    int operation,int alias,int mersenne_bits,int fault,uint32_t np0)
{
    const int i=(blockIdx.x*blockDim.x+threadIdx.x)/TPI;if(i>=count)return;
    using Context=cgbn_context_t<TPI>;using Env=cgbn_env_t<Context,BITS>;using Bn=typename Env::cgbn_t;
    Context ctx(cgbn_no_checks);Env e(ctx);Bn x,y,n,v;
    // The upstream load API takes mutable pointers; these inputs are read only.
    cgbn_load(e,x,input+i);cgbn_load(e,y,other+i);cgbn_load(e,n,modulus);
    const Stage2CoopField<Env> f{e,n,np0,mersenne_bits};
    for(int k=0;k<repeats;++k) {
        if(alias){if(operation)f.square(x,x);else f.mul(x,x,y);}
        else {if(operation)f.square(v,x);else f.mul(v,x,y);cgbn_set(e,x,v);}
    }
    cgbn_store(e,out+i,x);
    // The selected group has finished its store before the injected mutation.
    __syncwarp(((1u<<TPI)-1)<<(threadIdx.x%32 & ~(TPI-1)));
    if(fault && i==0 && threadIdx.x%TPI==0)out[0]._limbs[0]^=1;
}

struct Buffers {Word *a,*b,*n,*out;};
template<int BITS,int TPI>
void coop_launch(bool prepare,cudaFuncAttributes &attr,int threads,const Buffers &b,
    int count,int repeats,int operation,int alias,int mersenne_bits,int fault,Word ninv)
{
    if(prepare){CK(cudaFuncGetAttributes(&attr,cooperative_probe<BITS,TPI>));return;}
    cooperative_probe<BITS,TPI><<<(count*TPI+threads-1)/threads,threads>>>(
        (cgbn_mem_t<BITS>*)b.a,(cgbn_mem_t<BITS>*)b.b,(cgbn_mem_t<BITS>*)b.n,
        (cgbn_mem_t<BITS>*)b.out,count,repeats,operation,alias,mersenne_bits,fault,(uint32_t)ninv);
}
template<int NW>
void launch(bool prepare,cudaFuncAttributes &attr,int tpi,int threads,const Buffers &b,
    int nw,int count,int repeats,int operation,int alias,int bits,int kind,int fault,Word ninv)
{
    if(!tpi) {
        if(prepare){CK(cudaFuncGetAttributes(&attr,scalar_probe<NW>));return;}
        scalar_probe<NW><<<(count+threads-1)/threads,threads>>>(b.a,b.b,b.n,b.out,nw,count,repeats,operation,alias,fault,ninv);
        return;
    }
    // Exact radix widths: no hidden conversion to a wider Montgomery domain.
    if(nw!=NW){fprintf(stderr,"unsupported exact CGBN width %d\n",nw);exit(2);}
    const int p=kind==1?bits:0;
    if(tpi!=ECM_COOP_TPI){fprintf(stderr,"binary compiled for TPI=%d\n",ECM_COOP_TPI);exit(2);}
    coop_launch<64*NW,ECM_COOP_TPI>(prepare,attr,threads,b,count,repeats,operation,alias,p,fault,ninv);
}
void pack(Word *p,int nw,mpz_srcptr a){std::fill(p,p+nw,0);size_t z=0;mpz_export(p,&z,-1,8,0,0,a);}
int main(int argc,char **argv)
{
    if(argc!=12){fprintf(stderr,"bits count repeats tpi operation alias threads device kind fault samples\n");return 2;}
    const int bits=atoi(argv[1]),count=atoi(argv[2]),repeats=atoi(argv[3]),tpi=atoi(argv[4]),
        operation=atoi(argv[5]),alias=atoi(argv[6]),threads=atoi(argv[7]),device=atoi(argv[8]),
        kind=atoi(argv[9]),fault=atoi(argv[10]),nsamples=atoi(argv[11]);
    if(bits<2||bits>16384||count<1||count>131072||repeats<1||repeats>128||
       (tpi!=0&&tpi!=4&&tpi!=8&&tpi!=16)||operation<0||operation>1||alias<0||alias>1||
       (threads!=64&&threads!=128&&threads!=256)||kind<0||kind>2||fault<0||fault>1||nsamples<1||nsamples>9)return 2;
    const int nw=(bits+63)/64;const size_t words=(size_t)count*nw;
    CK(cudaSetDevice(device));mpz_t n,a,b,rinv,tmp;mpz_inits(n,a,b,rinv,tmp,nullptr);
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x20261102u+bits+kind);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,bits>=9&&kind==0?189:1);
    if(kind==2){mpz_urandomb(n,rng,bits);mpz_setbit(n,bits-1);mpz_setbit(n,0);}
    mpz_set_ui(tmp,1);mpz_mul_2exp(tmp,tmp,64*nw);if(!mpz_invert(rinv,tmp,n))return 2;
    std::vector<Word> hn(nw),ha(words),hb(words),expected(words),got(words);pack(hn.data(),nw,n);
    Word inv=1;for(int k=0;k<6;++k)inv*=2-hn[0]*inv;const Word ninv=0-inv;
    for(int i=0;i<count;++i) {
        mpz_urandomm(a,rng,n);mpz_urandomm(b,rng,n);
        if(i<4){if(i<2)mpz_set_ui(a,i);else mpz_sub_ui(a,n,i-1);}
        else if(i<20){mpz_set_ui(a,1);mpz_mul_2exp(a,a,(bits-1)-(i-4)%(bits-1));mpz_mod(a,a,n);}
        if(i<6)mpz_sub_ui(b,n,1);
        pack(ha.data()+(size_t)i*nw,nw,a);pack(hb.data()+(size_t)i*nw,nw,b);
        for(int k=0;k<repeats;++k){mpz_mul(a,a,operation?a:b);mpz_mul(a,a,rinv);mpz_mod(a,a,n);}
        pack(expected.data()+(size_t)i*nw,nw,a);
    }
    S2GDeviceConstants constants{};constants.point_mersenne_bits=kind==1?bits:0;constants.point_square=1;
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&constants,sizeof(constants)));
    Buffers d{};CK(cudaMalloc(&d.a,words*8));CK(cudaMalloc(&d.b,words*8));CK(cudaMalloc(&d.n,nw*8));CK(cudaMalloc(&d.out,words*8));
    CK(cudaMemcpy(d.a,ha.data(),words*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(d.b,hb.data(),words*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(d.n,hn.data(),nw*8,cudaMemcpyHostToDevice));
    cudaFuncAttributes attr{};
    auto run=[&](bool prepare=false) {
#define ROW(NW) if(nw<=NW)launch<NW>(prepare,attr,tpi,threads,d,nw,count,repeats,operation,alias,bits,kind,fault,ninv);else
        ROW(64)
#if ECM_COOP_MAX_WORDS >= 128
        ROW(128)
#endif
#if ECM_COOP_MAX_WORDS >= 256
        ROW(256)
#endif
        {fprintf(stderr,"unsupported width\n");exit(2);}
#undef ROW
        CK(cudaGetLastError());
    };
    run(true);cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));std::vector<float> samples;
    for(int k=0;k<nsamples;++k){float ms;CK(cudaEventRecord(begin));run();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms);}
    CK(cudaMemcpy(got.data(),d.out,words*8,cudaMemcpyDeviceToHost));size_t bad=0;
    for(size_t i=0;i<words;++i)bad+=got[i]!=expected[i];
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    printf("point_coop: bits=%d nw=%d count=%d repeats=%d tpi=%d operation=%d alias=%d threads=%d kind=%d bad=%zu median_ms=%.6f registers=%d local_bytes=%zu samples_ms=",
        bits,nw,count,repeats,tpi,operation,alias,threads,kind,bad,sorted[sorted.size()/2],attr.numRegs,attr.localSizeBytes);
    for(size_t i=0;i<samples.size();++i)printf("%s%.6f",i?",":"",samples[i]);puts("");
    CK(cudaFree(d.a));CK(cudaFree(d.b));CK(cudaFree(d.n));CK(cudaFree(d.out));CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
    gmp_randclear(rng);mpz_clears(n,a,b,rinv,tmp,nullptr);return bad?1:0;
}
