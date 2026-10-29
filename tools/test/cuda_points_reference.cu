// Direct use of production kernels. No OpenCL source conversion or new CUDA math.
#include "cuda_points_reference.h"
#include "ecm/stage2/cuda/device/point.cuh"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffers {
    std::vector<void*> memory;
    ~Buffers(){cudaDeviceSynchronize();for(auto p:memory)cudaFree(p);}
    unsigned long long*make(const std::uint64_t*src,std::size_t words){unsigned long long*p=nullptr;check(cudaMalloc(&p,8*words));memory.push_back(p);if(src)check(cudaMemcpy(p,src,8*words,cudaMemcpyHostToDevice));return p;}
};
struct Events {cudaEvent_t start=nullptr,stop=nullptr;Events(){check(cudaEventCreate(&start));check(cudaEventCreate(&stop));}~Events(){if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}};
template<int NW>void launch(const CudaPointInput&i,int mode,const unsigned long long*n,const unsigned long long*x,const unsigned long long*z,const unsigned long long*a,const unsigned long long*one,const unsigned long long*ks,const unsigned long long*v,unsigned long long*ox,unsigned long long*oz){
    const int count=mode?(i.count+i.block-1)/i.block:i.count;EcmCudaLaunch cfg(dim3((count+63)/64),dim3(64));
    if(!mode)ecm_launch_s2g_ladder_kernel<NW,true>(cfg,n,i.ninv,i.nw,x,z,a,one,ks,i.count,ox,oz,true);
    else ecm_launch_s2g_block_prod_kernel<NW>(cfg,v,i.count,i.nw,i.block,n,i.ninv,ox);
}
}
extern "C" __declspec(dllexport) int ecm_cuda_point_run(const CudaPointInput*input,int mode,std::uint64_t*ox,std::uint64_t*oz,double*kernel,double*wall,char*error,int bytes){
    try{const auto&i=*input;if(i.nw<1||i.nw>128||i.count<1||i.block<1)throw std::runtime_error("CUDA reference shape unsupported");
        check(cudaSetDevice(i.device));const auto begin=std::chrono::steady_clock::now();Buffers b;Events ev;
        auto n=b.make(i.n,i.nw),x=b.make(i.x,i.nw),z=b.make(i.z,i.nw),a=b.make(i.a24,i.nw),one=b.make(i.mone,i.nw),ks=b.make(i.scalars,i.count),v=b.make(mode?i.values:nullptr,std::size_t(i.count)*i.nw);
        const int count=mode?(i.count+i.block-1)/i.block:i.count;auto dx=b.make(nullptr,std::size_t(count)*i.nw),dz=b.make(nullptr,std::size_t(count)*i.nw);
        check(cudaEventRecord(ev.start));
        if(i.nw<=4)launch<4>(i,mode,n,x,z,a,one,ks,v,dx,dz);
        else if(i.nw<=8)launch<8>(i,mode,n,x,z,a,one,ks,v,dx,dz);
        else if(i.nw<=64)launch<64>(i,mode,n,x,z,a,one,ks,v,dx,dz);
        else launch<128>(i,mode,n,x,z,a,one,ks,v,dx,dz);
        check(cudaGetLastError());check(cudaEventRecord(ev.stop));check(cudaEventSynchronize(ev.stop));float ms=0;check(cudaEventElapsedTime(&ms,ev.start,ev.stop));*kernel=ms*.001;
        check(cudaMemcpy(ox,dx,std::size_t(count)*i.nw*8,cudaMemcpyDeviceToHost));if(!mode)check(cudaMemcpy(oz,dz,std::size_t(count)*i.nw*8,cudaMemcpyDeviceToHost));
        *wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();return 0;
    }catch(const std::exception&e){if(bytes>0)std::snprintf(error,bytes,"%s",e.what());return 1;}
}
