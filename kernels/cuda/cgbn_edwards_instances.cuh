// Shared implementation; instantiated only by the selected CUDA file group.
#pragma once
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "../../src/cuda/ecm_cuda_edwards_internal.h"
#include "cgbn_edwards_point.cuh"
#include "cgbn_edwards_naf.cuh"

namespace {
void check(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
template<class T> struct Device {
    T *p=nullptr;
    explicit Device(size_t n) {check(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){if(p)cudaFree(p);}
};
struct Report {
    cgbn_error_report_t *p=nullptr;
    Report(){check(cgbn_error_report_alloc(&p));}
    ~Report(){if(p)cgbn_error_report_free(p);}
};
template<uint32_t W,uint32_t T> struct Params {
    static const uint32_t BITS=W,TPI=T,TPB=128,MAX_ROTATION=1,SHM_LIMIT=0;
    static const bool CONSTANT_TIME=false;
};
template<class P> __global__ void slice(cgbn_error_report_t *report,
        cgbn_mem_t<P::BITS> *data,const uint32_t *scalar,uint32_t total_bits,
        uint32_t start,uint32_t stop,uint32_t count) {
    using Context=cgbn_context_t<P::TPI,P>;
    using Env=cgbn_env_t<Context,P::BITS>;
    const uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/P::TPI;
    if(i>=count)return;
    Context ctx(cgbn_report_monitor,report,i);Env env(ctx);
    auto *slot=data+10*i;
    typename Env::cgbn_t n,d;
    cgbn_edwards_point<Env> p,r;
    cgbn_load(env,n,slot);cgbn_load(env,d,slot+1);
    const uint32_t np0=cgbn_bn2mont(env,d,d,n);
    cgbn_load(env,p.x,slot+2);cgbn_bn2mont(env,p.x,p.x,n);
    cgbn_load(env,p.y,slot+3);cgbn_bn2mont(env,p.y,p.y,n);
    cgbn_load(env,p.z,slot+4);cgbn_bn2mont(env,p.z,p.z,n);
    cgbn_load(env,p.t,slot+5);cgbn_bn2mont(env,p.t,p.t,n);
    cgbn_load(env,r.x,slot+6);cgbn_bn2mont(env,r.x,r.x,n);
    cgbn_load(env,r.y,slot+7);cgbn_bn2mont(env,r.y,r.y,n);
    cgbn_load(env,r.z,slot+8);cgbn_bn2mont(env,r.z,r.z,n);
    cgbn_load(env,r.t,slot+9);cgbn_bn2mont(env,r.t,r.t,n);
    cgbn_edwards_field<Env> f{env,n,np0};
    for(uint32_t pos=start;pos<stop;++pos) {
        f.dbl(r,r);const uint32_t bit=total_bits-1-pos;
        if((scalar[bit/32]>>(bit%32))&1u)f.add_point(r,r,p,d);
    }
    cgbn_mont2bn(env,r.x,r.x,n,np0);cgbn_store(env,slot+6,r.x);
    cgbn_mont2bn(env,r.y,r.y,n,np0);cgbn_store(env,slot+7,r.y);
    cgbn_mont2bn(env,r.z,r.z,n,np0);cgbn_store(env,slot+8,r.z);
    cgbn_mont2bn(env,r.t,r.t,n,np0);cgbn_store(env,slot+9,r.t);
}
template<class Mem> void put(Mem &r,const mpz_t x) {
    std::memset(&r,0,sizeof(r));size_t used=0;mpz_export(r._limbs,&used,-1,4,0,0,x);
}
template<class P> int run(const mpz_t n,const mpz_t s,mpz_t *states,uint32_t count,
                         uint32_t &done,uint32_t chunk,EcmEdwardsProgress cb,void *context) {
    using Mem=cgbn_mem_t<P::BITS>;
    const uint32_t bits=static_cast<uint32_t>(mpz_sizeinbase(s,2));
    std::vector<Mem> host(size_t(count)*10);
    for(uint32_t i=0;i<count;++i) {
        put(host[10*i],n);
        for(unsigned j=0;j<9;++j)put(host[10*i+j+1],states[9*i+j]);
    }
    std::vector<uint32_t> words((bits+31ull)/32,0);size_t used=0;
    mpz_export(words.data(),&used,-1,4,0,0,s);
    Device<Mem> data(host.size());Device<uint32_t> scalar(words.size());Report report;
    check(cudaMemcpy(data.p,host.data(),host.size()*sizeof(Mem),cudaMemcpyHostToDevice));
    check(cudaMemcpy(scalar.p,words.data(),words.size()*4,cudaMemcpyHostToDevice));
    while(done<bits) {
        const uint32_t stop=done+std::min(chunk,bits-done);
        const uint32_t ipb=P::TPB/P::TPI;
        slice<P><<<(count+ipb-1)/ipb,P::TPB>>>(report.p,data.p,scalar.p,bits,done,stop,count);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        if(cgbn_error_report_check(report.p))throw std::runtime_error("CGBN Edwards arithmetic error");
        check(cudaMemcpy(host.data(),data.p,host.size()*sizeof(Mem),cudaMemcpyDeviceToHost));
        for(uint32_t i=0;i<count;++i)for(unsigned j=5;j<9;++j)
            mpz_import(states[9*i+j],P::BITS/32,-1,4,0,0,host[10*i+j+1]._limbs);
        done=stop;
        if(cb && !cb(done,context))return 2;
    }
    return 0;
}
#include "../../src/cuda/ecm_cuda_edwards_naf.inl"
}

namespace ecm_edwards_cuda_detail {
#define ECM_EDWARDS_TIER(W,T) \
int run_##W(const Request &r) { \
    if(r.plan)return run_naf<Params<W,T>>(r.n,*r.plan,r.states,r.count,r.done,r.chunk,r.progress,r.context,r.doubling,r.metrics); \
    return run<Params<W,T>>(r.n,r.scalar,r.states,r.count,r.done,r.chunk,r.progress,r.context); \
}
#include "../../src/cuda/ecm_cuda_edwards_tiers.inc"
#undef ECM_EDWARDS_TIER
}
