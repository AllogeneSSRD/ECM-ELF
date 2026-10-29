#pragma once
#include <cstdint>
// Optional timing adapter to the existing production CUDA point.cuh kernels.
// Prepared Montgomery constants/values; output X/Z ordinary, products Montgomery.
struct CudaPointInput {
    int device,nw,count,block;
    std::uint64_t ninv;
    const std::uint64_t *n,*x,*z,*a24,*mone,*scalars,*values;
};
using CudaPointRun=int(*)(const CudaPointInput*,int,std::uint64_t*,std::uint64_t*,double*,double*,char*,int);
