#include <cuda_runtime.h>
#include <gmp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define GL_P 0xFFFFFFFF00000001ull
#include "../bench/ntt_goldilocks_reduce.cuh"
#include "../bench/ntt_goldilocks_ptx.cuh"
using Word=unsigned long long;
#define CK(call) do{const auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)
static Word live=0,peak=0;
struct Buffer{
    Word *p=nullptr;size_t bytes;
    explicit Buffer(size_t size):bytes(size){CK(cudaMalloc(&p,size));live+=bytes;peak=std::max(live,peak);}
    ~Buffer(){CK(cudaFree(p));live-=bytes;}
};
template<int METHOD>__device__ __forceinline__ Word mul(Word a,Word b)
{
    if constexpr(METHOD==0)return gl_reduce128_short(a*b,__umul64hi(a,b));
    if constexpr(METHOD==1)return gl_reduce128_ptx(a*b,__umul64hi(a,b));
    return gl_mul_ptx(a,b);
}
template<int METHOD>__global__ void primitive(const Word *a,const Word *b,Word *out,size_t count,bool reduce)
{
    const auto i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=reduce ? (METHOD==0 ? gl_reduce128_short(a[i],b[i]) : gl_reduce128_ptx(a[i],b[i])) : mul<METHOD>(a[i],b[i]);
}
template<int METHOD>__global__ void chain(Word *out,size_t count,int iterations)
{
    const auto i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){Word v=(i+1)*0x9e3779b97f4a7c15ull;for(int j=0;j<iterations;++j)v=mul<METHOD>(v,0xfedcba9876543211ull);out[i]=v;}
}
__global__ void compare_words(const Word *a,const Word *b,size_t count,Word *bad)
{
    const auto i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && a[i]!=b[i])atomicAdd(bad,1ull);
}
__global__ void corrupt(Word *out){out[0]^=1;}
static Word random64(Word &seed){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static void put(mpz_t z,Word v){mpz_import(z,1,-1,8,0,0,&v);}
static Word get(mpz_t z){Word v=0;size_t count=0;mpz_export(&v,&count,-1,8,0,0,z);return v;}
static int check(bool fault)
{
    std::vector<Word> a,b;const Word edges[]={0,1,2,GL_P-1,GL_P,GL_P+1,~0ull,0xffffffffull,1ull<<32,1ull<<63,0xffffffff00000000ull,0xfffffffe00000001ull};
    for(auto lo:edges)for(auto hi:edges){a.push_back(lo);b.push_back(hi);}
    Word seed=0x351987;for(int i=0;i<200000;++i){a.push_back(random64(seed));b.push_back(random64(seed));}
    Buffer da(a.size()*8),db(b.size()*8),out(a.size()*8);
    CK(cudaMemcpy(da.p,a.data(),da.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(db.p,b.data(),db.bytes,cudaMemcpyHostToDevice));
    mpz_t p,x,y,z;mpz_inits(p,x,y,z,nullptr);put(p,GL_P);Word bad=0;
    std::vector<Word> got(a.size()),want(a.size());
    for(bool reduce:{true,false}){
        for(size_t i=0;i<a.size();++i){put(x,a[i]);put(y,b[i]);
            if(reduce){mpz_mul_2exp(z,y,64);mpz_add(z,z,x);}else mpz_mul(z,x,y);
            mpz_mod(z,z,p);want[i]=get(z);}
        for(int method:{0,1,2}){
            const auto grid=(unsigned int)((a.size()+255)/256);
            if(method==0)primitive<0><<<grid,256>>>(da.p,db.p,out.p,a.size(),reduce);
            else if(method==1)primitive<1><<<grid,256>>>(da.p,db.p,out.p,a.size(),reduce);
            else primitive<2><<<grid,256>>>(da.p,db.p,out.p,a.size(),reduce);
            if(fault && method==1 && reduce)corrupt<<<1,1>>>(out.p);
            CK(cudaMemcpy(got.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));Word wrong=0;
            for(size_t i=0;i<a.size();++i)wrong+=got[i]!=want[i];bad+=wrong;
            std::printf("gl_ptx_check: method=%d reduce=%d words=%zu bad=%llu\n",method,(int)reduce,a.size(),wrong);
        }
    }
    // Independent GMP reference for a dependent sequence as well as primitives.
    for(int method:{0,1,2}){
        if(method==0)chain<0><<<1,256>>>(out.p,256,256);
        else if(method==1)chain<1><<<1,256>>>(out.p,256,256);
        else chain<2><<<1,256>>>(out.p,256,256);
        CK(cudaMemcpy(got.data(),out.p,256*8,cudaMemcpyDeviceToHost));Word wrong=0;
        put(y,0xfedcba9876543211ull);
        for(size_t i=0;i<256;++i){put(x,(i+1)*0x9e3779b97f4a7c15ull);
            for(int j=0;j<256;++j){mpz_mul(x,x,y);mpz_mod(x,x,p);}wrong+=got[i]!=get(x);}
        bad+=wrong;std::printf("gl_ptx_chain_check: method=%d words=256 iterations=256 bad=%llu\n",method,wrong);
    }
    mpz_clears(p,x,y,z,nullptr);std::printf("gl_ptx_gate: fault=%d bad=%llu\n",(int)fault,bad);return bad ? 3 : 0;
}
static int benchmark(int candidate)
{
    constexpr size_t count=1ull<<20;constexpr int iterations=512;
    Buffer reference(count*8),out(count*8),bad(8);
    chain<0><<<(unsigned int)(count/256),256>>>(reference.p,count,iterations);
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));int run=0;
    for(int mode:{0,1,1,0,1,0,0,1}){
        double seconds=0;Word wrong=0;
        for(int repeat=0;repeat<4;++repeat){
            CK(cudaEventRecord(start));
            if(!mode)chain<0><<<(unsigned int)(count/256),256>>>(out.p,count,iterations);
            else if(candidate==1)chain<1><<<(unsigned int)(count/256),256>>>(out.p,count,iterations);
            else chain<2><<<(unsigned int)(count/256),256>>>(out.p,count,iterations);
            CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));if(repeat)seconds+=ms/1000.0;
            CK(cudaMemset(bad.p,0,8));compare_words<<<(unsigned int)(count/256),256>>>(out.p,reference.p,count,bad.p);
            CK(cudaMemcpy(&wrong,bad.p,8,cudaMemcpyDeviceToHost));if(wrong)return 3;
        }
        std::printf("gl_ptx_bench: run=%d mode=%d candidate=%d N=%zu iterations=%d seconds=%.9f bad=%llu\n",++run,mode,candidate,count,iterations,seconds/3,wrong);
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));return 0;
}
template<int M>static void resources(){cudaFuncAttributes a{};CK(cudaFuncGetAttributes(&a,chain<M>));std::printf("gl_ptx_resources: method=%d regs=%d local=%zu\n",M,a.numRegs,a.localSizeBytes);}
int main(int argc,char **argv)
{
    CK(cudaSetDevice(argc>1 ? std::atoi(argv[1]) : 1));std::setvbuf(stdout,nullptr,_IONBF,0);
    resources<0>();resources<1>();resources<2>();int code;
    if(argc>2 && !std::strcmp(argv[2],"--bench"))code=benchmark(argc>3 ? std::atoi(argv[3]) : 1);
    else code=check(argc>2 && !std::strcmp(argv[2],"--fault"));
    CK(cudaDeviceSynchronize());std::printf("gl_ptx_memory: peak_bytes=%llu live_bytes=%llu\n",peak,live);return code ? code : live ? 4 : 0;
}
