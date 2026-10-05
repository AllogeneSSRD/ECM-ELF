#include <cuda_runtime.h>
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "stage2_point_reference.cuh"
#include "../bench/stage2_point_mersenne.cuh"
using Word=unsigned long long;
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<int NW,bool NEW>
__global__ void probe(const Word*a,const Word*b,const Word*n,Word*out,int nw,int bits,int count,int repeats,int alias,int fault,Word ninv){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Word x[NW],y[NW],v[NW];
    for(int j=0;j<nw;++j){x[j]=a[(size_t)i*nw+j];y[j]=b[(size_t)i*nw+j];}
    for(int k=0;k<repeats;++k){
        Word *dst=alias==1?x:alias==2?y:v;
        if(NEW)s2g_mersenne_mont_mul<NW>(dst,x,y,n,nw,bits);
        else s2g_mont_mul<NW>(dst,x,y,n,ninv,nw);
        // Keep the same recurrence regardless of the output alias.
        if(alias!=1)for(int j=0;j<nw;++j)x[j]=dst[j];
        if(alias==2)for(int j=0;j<nw;++j)y[j]=b[(size_t)i*nw+j];
    }
    for(int j=0;j<nw;++j)out[(size_t)i*nw+j]=x[j];
    if(NEW && fault && i==0)out[0]^=1;
}
template<int NW>void launch(int mode,const Word*a,const Word*b,const Word*n,Word*out,int nw,int bits,int count,int repeats,int alias,int fault,int threads,Word ninv){
    if(mode)probe<NW,true><<<(count+threads-1)/threads,threads>>>(a,b,n,out,nw,bits,count,repeats,alias,fault,ninv);
    else probe<NW,false><<<(count+threads-1)/threads,threads>>>(a,b,n,out,nw,bits,count,repeats,alias,fault,ninv);
}
void words(Word*p,int nw,const mpz_t v){std::fill(p,p+nw,0);size_t z;mpz_export(p,&z,-1,8,0,0,v);}
int main(int argc,char**argv){
    if(argc!=9){fprintf(stderr,"bits count repeats mode alias threads device fault\n");return 2;}
    int bits=atoi(argv[1]),count=atoi(argv[2]),repeats=atoi(argv[3]),mode=atoi(argv[4]),alias=atoi(argv[5]),threads=atoi(argv[6]),device=atoi(argv[7]),fault=atoi(argv[8]);
    if(bits<2||bits>8192||count<1||count>131072||repeats<1||repeats>64||mode<0||mode>1||alias<0||alias>2||threads<32||threads>256)return 2;
    int nw=(bits+63)/64;size_t bytes=(size_t)count*nw*8;
    CK(cudaSetDevice(device));
    mpz_t n,a,b,rinv,t,ref;mpz_inits(n,a,b,rinv,t,ref,nullptr);
    mpz_set_ui(n,1);mpz_mul_2exp(n,n,bits);mpz_sub_ui(n,n,1);
    mpz_set_ui(t,1);mpz_mul_2exp(t,t,64*nw);if(!mpz_invert(rinv,t,n))return 2;
    gmp_randstate_t rng;gmp_randinit_mt(rng);gmp_randseed_ui(rng,0x20261005);
    std::vector<Word> ha(count*nw),hb(count*nw),hn(nw),got(count*nw),expected(count*nw);
    words(hn.data(),nw,n);
    Word inv=1;for(int i=0;i<6;++i)inv*=2-hn[0]*inv;const Word ninv=0-inv;
    for(int i=0;i<count;++i){
        mpz_urandomb(a,rng,bits);mpz_mod(a,a,n);mpz_urandomb(b,rng,bits);mpz_mod(b,b,n);
        if(i<16){
            int ai=i/4,bi=i%4;
            if(ai<2)mpz_set_ui(a,ai);else mpz_sub_ui(a,n,ai-1);
            if(bi<2)mpz_set_ui(b,bi);else mpz_sub_ui(b,n,bi-1);
        }
        // Include large powers and the partially used high limb.
        if(i>=16 && i<32){mpz_set_ui(a,1);mpz_mul_2exp(a,a,(bits-1)-(i-16)%(bits-1));}
        words(ha.data()+(size_t)i*nw,nw,a);words(hb.data()+(size_t)i*nw,nw,b);
        for(int k=0;k<repeats;++k){mpz_mul(a,a,b);mpz_mul(a,a,rinv);mpz_mod(a,a,n);}
        words(expected.data()+(size_t)i*nw,nw,a);
    }
    Word *da,*db,*dn,*dout;CK(cudaMalloc(&da,bytes));CK(cudaMalloc(&db,bytes));CK(cudaMalloc(&dn,nw*8));CK(cudaMalloc(&dout,bytes));
    CK(cudaMemcpy(da,ha.data(),bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(db,hb.data(),bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(dn,hn.data(),nw*8,cudaMemcpyHostToDevice));
    auto run=[&](){
        if(nw<=1)launch<1>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else if(nw<=2)launch<2>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else if(nw<=4)launch<4>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else if(nw<=8)launch<8>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else if(nw<=16)launch<16>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else if(nw<=32)launch<32>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else if(nw<=64)launch<64>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        else launch<128>(mode,da,db,dn,dout,nw,bits,count,repeats,alias,fault,threads,ninv);
        CK(cudaGetLastError());
    };
    for(int i=0;i<3;++i)run();CK(cudaDeviceSynchronize());
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));float ms=0;
    CK(cudaEventRecord(start));for(int i=0;i<3;++i)run();CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));CK(cudaEventElapsedTime(&ms,start,end));
    CK(cudaMemcpy(got.data(),dout,bytes,cudaMemcpyDeviceToHost));size_t bad=0,first=0;
    for(size_t i=0;i<got.size();++i)if(got[i]!=expected[i]){if(!bad)first=i;++bad;}
    printf("point_mersenne: bits=%d nw=%d count=%d repeats=%d mode=%d alias=%d threads=%d device=%d bad=%zu first_bad=%zu event_ms=%.6f payload_bytes=%zu\n",bits,nw,count,repeats,mode,alias,threads,device,bad,first,ms/3,(3*bytes+nw*8));
    CK(cudaFree(da));CK(cudaFree(db));CK(cudaFree(dn));CK(cudaFree(dout));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));gmp_randclear(rng);mpz_clears(n,a,b,rinv,t,ref,nullptr);
    return bad?1:0;
}
