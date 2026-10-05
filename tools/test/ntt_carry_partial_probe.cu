#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#define CK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s: %s\n",#call,cudaGetErrorString(e));std::exit(2);}}while(0)
#include "carry_reference.cuh"
#include "carry_cone_reference.cuh"
#include "../bench/ntt_carry_partial.cuh"
using U=unsigned long long;
static void launch(int mode,const U *c,U n,int bpw,unsigned int *partial,U *out,U stride,U batch){
    const auto blocks=(unsigned int)((n+255)/256);
    if(mode){carry_partial_kernel<<<dim3(blocks,(unsigned int)batch),256>>>(c,n,bpw,partial,stride);
        carry_partial_finish_kernel<<<(unsigned int)batch,256>>>(partial,blocks,out);}
    else carry_residual_kernel<<<dim3(blocks,(unsigned int)batch),256>>>(c,n,bpw,out,stride);
    CK(cudaGetLastError());
}
static unsigned height(U v){unsigned n=0;while(v){++n;v>>=1;}return n;}
template<int R> static void cone(int mode,const U *c,U *digits,U n,int bpw,unsigned int *partial,U *out,U stride,U batch){
    const auto blocks=(unsigned int)((n+255)/256);const dim3 grid(blocks,(unsigned int)batch);
    if(mode){carry_cone_check_kernel<R><<<grid,256>>>(c,digits,n,bpw,stride,partial);
        carry_partial_finish_kernel<<<(unsigned int)batch,256>>>(partial,blocks,out);}
    else {carry_cone_kernel<R><<<grid,256>>>(c,digits,n,bpw,stride);carry_residual_kernel<<<grid,256>>>(digits,n,bpw,out,stride);}
    CK(cudaGetLastError());
}
template<int R> static U cone_gate(int bpw){
    U bad=0;std::mt19937_64 rng(0x11830797+R);
    for(U n:{1ull,31ull,32ull,33ull,255ull,256ull,257ull,513ull,4097ull})for(int pattern:{0,1}){
        const U batch=3,stride=n+19,blocks=(n+255)/256,mask=(1ull<<bpw)-1;const U guard=0xfeedcafebee;
        std::vector<U> input(stride*batch,guard),want(input),got(input.size()),verdict(2*batch),expected(2*batch);
        for(U s=0;s<batch;++s){U carry=0;for(U i=0;i<n;++i){
            const U v=pattern ? (i==s ? 1ull<<bpw : mask) : rng();input[s*stride+i]=v;
            const U sum=v+carry,overflow=sum<v;want[s*stride+i]=sum&mask;carry=(sum>>bpw)+(overflow ? 1ull<<(64-bpw) : 0);}}
        U *dc=nullptr,*digits=nullptr,*out=nullptr;unsigned int *part=nullptr;
        CK(cudaMalloc(&dc,input.size()*8));CK(cudaMalloc(&digits,input.size()*8));CK(cudaMalloc(&out,verdict.size()*8));CK(cudaMalloc(&part,(2*blocks*batch+4)*4));
        CK(cudaMemcpy(dc,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        for(int mode:{0,1}){
            std::vector<U> initial(input.size(),guard);CK(cudaMemcpy(digits,initial.data(),initial.size()*8,cudaMemcpyHostToDevice));
            for(U s=0;s<batch;++s){expected[2*s]=9+s;expected[2*s+1]=s ? 0 : 70;}
            CK(cudaMemcpy(out,expected.data(),expected.size()*8,cudaMemcpyHostToDevice));CK(cudaMemset(part,0xda,(2*blocks*batch+4)*4));
            for(int repeat=0;repeat<3;++repeat)cone<R>(mode,dc,digits,n,bpw,part,out,stride,batch);
            CK(cudaMemcpy(got.data(),digits,got.size()*8,cudaMemcpyDeviceToHost));for(size_t i=0;i<got.size();++i)bad+=got[i]!=want[i];
            for(U s=0;s<batch;++s)for(U i=0;i<n;++i)expected[2*s+1]=std::max(expected[2*s+1],(U)height(want[s*stride+i]));
            CK(cudaMemcpy(verdict.data(),out,verdict.size()*8,cudaMemcpyDeviceToHost));bad+=verdict!=expected;
            unsigned int tail[4];CK(cudaMemcpy(tail,part+2*blocks*batch,16,cudaMemcpyDeviceToHost));for(auto v:tail)bad+=v!=0xdadadadau;
        }
        CK(cudaFree(dc));CK(cudaFree(digits));CK(cudaFree(out));CK(cudaFree(part));
    }
    return bad;
}
static int gate(){
    std::mt19937_64 rng(0x50cb7937);U cases=0,words=0,bad=0;
    for(int bpw:{1,7,26,32,62})for(U n:{1ull,31ull,32ull,33ull,255ull,256ull,257ull,513ull,4097ull})for(U batch:{1ull,3ull}){
        const U stride=n+19,blocks=(n+255)/256;std::vector<U> c(stride*batch,0xfee1deadbeef);
        std::vector<U> want(2*batch),got(want.size()),seed(want.size());
        for(U s=0;s<batch;++s){seed[2*s]=s+11;seed[2*s+1]=s==0 ? 70 : 0;
            for(U i=0;i<n;++i)c[s*stride+i]=(i%3==0 ? rng() : rng()&((1ull<<bpw)-1));}
        U *dc=nullptr,*out=nullptr;unsigned int *part=nullptr;
        CK(cudaMalloc(&dc,c.size()*8));CK(cudaMalloc(&out,want.size()*8));CK(cudaMalloc(&part,(2*blocks*batch+4)*4));
        CK(cudaMemcpy(dc,c.data(),c.size()*8,cudaMemcpyHostToDevice));
        for(int mode:{0,1}){
            want=seed;CK(cudaMemcpy(out,seed.data(),seed.size()*8,cudaMemcpyHostToDevice));CK(cudaMemset(part,0xda,(2*blocks*batch+4)*4));
            for(int repeat=0;repeat<3;++repeat){launch(mode,dc,n,bpw,part,out,stride,batch);
                for(U s=0;s<batch;++s)for(U i=0;i<n;++i){const U v=c[s*stride+i];want[2*s]+=v>=(1ull<<bpw);want[2*s+1]=std::max(want[2*s+1],(U)height(v));}}
            CK(cudaMemcpy(got.data(),out,got.size()*8,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<got.size();++i)bad+=got[i]!=want[i];
            std::vector<unsigned int> tail(4);CK(cudaMemcpy(tail.data(),part+2*blocks*batch,16,cudaMemcpyDeviceToHost));
            for(auto v:tail)bad+=v!=0xdadadadau;++cases;words+=3*n*batch;
        }
        std::vector<U> after(c.size());CK(cudaMemcpy(after.data(),dc,after.size()*8,cudaMemcpyDeviceToHost));bad+=after!=c;
        CK(cudaFree(dc));CK(cudaFree(out));CK(cudaFree(part));
    }
    // A corrupt partial must propagate through the final diagnostic.
    unsigned int *part=nullptr;U *out=nullptr;CK(cudaMalloc(&part,8));CK(cudaMalloc(&out,16));
    unsigned int poisoned[2]={1,26};CK(cudaMemcpy(part,poisoned,8,cudaMemcpyHostToDevice));CK(cudaMemset(out,0,16));
    carry_partial_finish_kernel<<<1,256>>>(part,1,out);U got[2];CK(cudaMemcpy(got,out,16,cudaMemcpyDeviceToHost));
    const bool fault=got[0]==1 && got[1]==26;bad+=!fault;CK(cudaFree(part));CK(cudaFree(out));
    std::printf("carry_partial_gate: cases=%llu words=%llu bad=%llu fault_propagated=%d\n",cases,words,bad,(int)fault);
    U cone_bad=cone_gate<10>(7)+cone_gate<5>(13)+cone_gate<3>(26)+cone_gate<5>(26)+cone_gate<6>(26)+cone_gate<2>(32)+cone_gate<2>(62)
        +cone_gate<1>(62)+cone_gate<4>(17)+cone_gate<7>(10)+cone_gate<8>(9)+cone_gate<9>(8);
    std::printf("carry_cone_gate: configurations=12 shapes=18 modes=2 bad=%llu (CPU exact ripple, long mask runs, tail guards, accumulated diagnostics)\n",cone_bad);bad+=cone_bad;
    return bad ? 3 : 0;
}
__global__ void fill(U *c,U n,int poison){const auto i=blockIdx.x*256ull+threadIdx.x;if(i<n)c[i]=poison && i%257==0 ? (1ull<<26) : ((i*1234567ull)&((1ull<<26)-1));}
__global__ void fill_cone(U *c,U n){const auto i=blockIdx.x*256ull+threadIdx.x;if(i<n)c[i]=(i*6364136223846793005ull+1442695040888963407ull)&((1ull<<60)-1);}
template<int R> static int bench_cone(int k,U batch){
    if(k<10 || k>27 || !batch || batch>65535 || batch*(1ull<<k)>1ull<<28)return 2;
    const U n=1ull<<k,total=n*batch,blocks=(n+255)/256;U *c=nullptr,*digits=nullptr,*out=nullptr;unsigned int *part=nullptr;
    CK(cudaMalloc(&c,total*8));CK(cudaMalloc(&digits,total*8));CK(cudaMalloc(&out,batch*16));CK(cudaMalloc(&part,blocks*batch*8));
    fill_cone<<<(unsigned int)((total+255)/256),256>>>(c,total);CK(cudaGetLastError());
    std::vector<U> reference(total),got(total);bool have_reference=false;cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    for(int mode:{0,1,1,0,1,0,0,1}){
        CK(cudaMemset(out,0,batch*16));for(int i=0;i<3;++i)cone<R>(mode,c,digits,n,26,part,out,n,batch);CK(cudaDeviceSynchronize());
        CK(cudaMemset(out,0,batch*16));const int repeats=20;CK(cudaEventRecord(a));for(int i=0;i<repeats;++i)cone<R>(mode,c,digits,n,26,part,out,n,batch);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
        float ms=0;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaMemcpy(got.data(),digits,total*8,cudaMemcpyDeviceToHost));U bad=0;
        if(!have_reference){reference=got;have_reference=true;}else for(size_t i=0;i<got.size();++i)bad+=got[i]!=reference[i];
        std::vector<U> verdict(batch*2);CK(cudaMemcpy(verdict.data(),out,batch*16,cudaMemcpyDeviceToHost));for(U s=0;s<batch;++s){U mx=0;for(U i=0;i<n;++i)mx=std::max(mx,got[s*n+i]);bad+=verdict[2*s]!=0;bad+=verdict[2*s+1]!=(U)height(mx);}
        std::printf("carry_cone_bench: mode=%d k=%d batch=%llu rounds=%d seconds=%.9f scratch_bytes=%llu bad=%llu\n",mode,k,batch,R,ms*.001/repeats,blocks*batch*8,bad);if(bad)return 3;
    }
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaFree(c));CK(cudaFree(digits));CK(cudaFree(out));CK(cudaFree(part));return 0;
}
static int bench(int k,U batch,int poison){
    if(k<12 || k>27 || !batch || batch>128 || poison<0 || poison>1)return 2;
    const U n=1ull<<k,total=n*batch,blocks=(n+255)/256;
    if(total>1ull<<28)return 2;
    U *c=nullptr,*out=nullptr;unsigned int *part=nullptr;
    CK(cudaMalloc(&c,total*8));CK(cudaMalloc(&out,batch*16));CK(cudaMalloc(&part,blocks*batch*8));
    fill<<<(unsigned int)((total+255)/256),256>>>(c,total,poison);CK(cudaGetLastError());
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    for(int mode:{0,1,1,0,1,0,0,1}){
        CK(cudaMemset(out,0,batch*16));for(int i=0;i<3;++i)launch(mode,c,n,26,part,out,n,batch);
        CK(cudaDeviceSynchronize());CK(cudaMemset(out,0,batch*16));
        const int repeats=20;CK(cudaEventRecord(a));for(int i=0;i<repeats;++i)launch(mode,c,n,26,part,out,n,batch);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
        float ms=0;CK(cudaEventElapsedTime(&ms,a,b));std::vector<U> got(batch*2);CK(cudaMemcpy(got.data(),out,got.size()*8,cudaMemcpyDeviceToHost));U bad=0;
        for(U s=0;s<batch;++s){const U lo=s*n,hi=lo+n;const U count=poison ? (hi+256)/257-(lo+256)/257 : 0;
            bad+=got[2*s]!=count*repeats;bad+=got[2*s+1]!=(U)(poison && count ? 27 : 26);}
        std::printf("carry_partial_bench: mode=%d k=%d batch=%llu poison=%d seconds=%.9f scratch_bytes=%llu bad=%llu\n",mode,k,batch,poison,ms*.001/repeats,blocks*batch*8,bad);
        if(bad)return 3;
    }
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaFree(c));CK(cudaFree(out));CK(cudaFree(part));return 0;
}
template<class T> static void resource(const char *name,T kernel){cudaFuncAttributes a{};int blocks=0;CK(cudaFuncGetAttributes(&a,kernel));CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,256,0));std::printf("carry_partial_resource: name=%s regs=%d shared=%zu local=%zu capacity=%d\n",name,a.numRegs,a.sharedSizeBytes,a.localSizeBytes,blocks);}
int main(int argc,char **argv){if(argc<2)return 2;const int device=std::atoi(argv[1]);CK(cudaSetDevice(device));std::printf("carry_partial_device: %d\n",device);
    resource("reference",carry_residual_kernel);resource("partial",carry_partial_kernel);resource("finish",carry_partial_finish_kernel);
    resource("cone5",carry_cone_kernel<5>);resource("cone_check5",carry_cone_check_kernel<5>);
    resource("cone6",carry_cone_kernel<6>);resource("cone_check6",carry_cone_check_kernel<6>);
    if(argc==2)return gate();if(argc==6 && std::string(argv[2])=="--bench")return bench(std::atoi(argv[3]),std::strtoull(argv[4],nullptr,10),std::atoi(argv[5]));
    if((argc==5 || argc==6) && std::string(argv[2])=="--bench-cone"){
        const int rounds=argc==6 ? std::atoi(argv[5]) : 5;
        if(rounds==5)return bench_cone<5>(std::atoi(argv[3]),std::strtoull(argv[4],nullptr,10));
        if(rounds==6)return bench_cone<6>(std::atoi(argv[3]),std::strtoull(argv[4],nullptr,10));}
    return 2;}
