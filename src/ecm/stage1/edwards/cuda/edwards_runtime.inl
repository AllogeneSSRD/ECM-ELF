// Host allocations, GMP packing, slicing, timers and publication.
#pragma once
#include <gmp.h>
#include "../../../cuda/arithmetic/cgbn/types.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "edwards_internal.h"
#include "edwards_slice.h"
#include "launch.h"
namespace {
template<uint32_t W> struct HostMem {uint32_t _limbs[W/32];};
template<uint32_t W,uint32_t T> struct HostParams {static const uint32_t BITS=W,TPI=T,TPB=128;};
void check(cudaError_t rc) {if(rc!=cudaSuccess)throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Device,cudaGetErrorString(rc));}
template<class T> struct Device {
    T *p=nullptr;
    explicit Device(size_t n) {check(cudaMalloc(&p,ecm_cuda_stage1::checked_size({n,sizeof(T)})));}
    Device(const Device &)=delete;
    Device &operator=(const Device &)=delete;
    ~Device(){if(p)cudaFree(p);}
};
struct Report {
    cgbn_error_report_t *p=nullptr;
    Report(){check(cgbn_error_report_alloc(&p));}
    Report(const Report &)=delete;
    Report &operator=(const Report &)=delete;
    ~Report(){if(p)cgbn_error_report_free(p);}
};
template<class Mem> void put(Mem &r,const mpz_t x) {
    std::memset(&r,0,sizeof(r));size_t used=0;mpz_export(r._limbs,&used,-1,4,0,0,x);
}
template<class P> int run(const mpz_t n,const mpz_t s,mpz_t *states,uint32_t count,
                         uint32_t &done,uint32_t chunk,EcmEdwardsProgress cb,void *context,
                         EcmEdwardsPrepared prepared,const ecm_edwards_cuda_detail::Kernels &k) {
    using Mem=HostMem<P::BITS>;
    const uint32_t bits=static_cast<uint32_t>(mpz_sizeinbase(s,2));
    std::vector<Mem> host(ecm_cuda_stage1::checked_size({size_t(count),10}));
    for(uint32_t i=0;i<count;++i) {
        put(host[size_t(i)*10],n);
        for(unsigned j=0;j<9;++j)put(host[size_t(i)*10+j+1],states[size_t(i)*9+j]);
    }
    std::vector<uint32_t> words((bits+31ull)/32,0);size_t used=0;
    mpz_export(words.data(),&used,-1,4,0,0,s);
    Device<Mem> data(host.size());Device<uint32_t> scalar(words.size());Report report;
    check(cudaMemcpy(data.p,host.data(),host.size()*sizeof(Mem),cudaMemcpyHostToDevice));
    check(cudaMemcpy(scalar.p,words.data(),words.size()*4,cudaMemcpyHostToDevice));
    if(prepared)prepared(context);
    EcmEdwardsSlice slices(chunk);
    while(done<bits) {
        const uint32_t stop=done+slices.columns(bits-done);
        const uint32_t ipb=P::TPB/P::TPI;
        const auto started=std::chrono::steady_clock::now();
        k.binary(report.p,data.p,scalar.p,bits,done,stop,count);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        slices.observe(stop-done,std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count());
        if(cgbn_error_report_check(report.p))throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Device,"CGBN Edwards arithmetic error");
        check(cudaMemcpy(host.data(),data.p,host.size()*sizeof(Mem),cudaMemcpyDeviceToHost));
        for(uint32_t i=0;i<count;++i)for(unsigned j=5;j<9;++j)
            mpz_import(states[size_t(i)*9+j],P::BITS/32,-1,4,0,0,host[size_t(i)*10+j+1]._limbs);
        done=stop;
        if(cb && !cb(done,context))return 2;
    }
    return 0;
}
struct EdwardsEvent {
    cudaEvent_t value{};
    EdwardsEvent(){check(cudaEventCreate(&value));}
    EdwardsEvent(const EdwardsEvent &)=delete;
    EdwardsEvent &operator=(const EdwardsEvent &)=delete;
    ~EdwardsEvent(){cudaEventDestroy(value);}
    void wait(){check(cudaDeviceSynchronize());check(cudaEventSynchronize(value));}
};
template<class P> int run_naf(const mpz_t n,const EcmEdwardsNaf &plan,mpz_t *states,
        uint32_t count,uint32_t &done,uint32_t chunk,EcmEdwardsProgress cb,
        void *context,int doubling,EcmEdwardsNafMetrics *metrics,EcmEdwardsPrepared prepared,const ecm_edwards_cuda_detail::Kernels &k) {
    using Mem=HostMem<P::BITS>;
    const uint32_t entries=1u<<(plan.window-2),ipb=P::TPB/P::TPI;
    if(metrics){*metrics={};metrics->columns=plan.bits;metrics->additions=plan.additions;
        metrics->container_bits=P::BITS;metrics->tpi=P::TPI;
        metrics->dictionary_bytes=uint64_t(count)*entries*4*sizeof(Mem);}
    if(done==plan.bits){if(prepared)prepared(context);return 0;}
    std::vector<Mem> host(ecm_cuda_stage1::checked_size({size_t(count),10})),ordinary(ecm_cuda_stage1::checked_size({size_t(count),4}));
    for(uint32_t i=0;i<count;++i) {
        put(host[size_t(i)*10],n);
        for(unsigned j=0;j<9;++j)put(host[size_t(i)*10+j+1],states[size_t(i)*9+j]);
    }
    Device<Mem> data(host.size()),result(ordinary.size()),dictionary(ecm_cuda_stage1::checked_size({size_t(count),size_t(entries),4}));
    Device<uint32_t> nonzero(plan.nonzero.size()),codes(plan.codes.size());Report report;
    check(cudaMemcpy(data.p,host.data(),host.size()*sizeof(Mem),cudaMemcpyHostToDevice));
    check(cudaMemcpy(nonzero.p,plan.nonzero.data(),plan.nonzero.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(codes.p,plan.codes.data(),plan.codes.size()*4,cudaMemcpyHostToDevice));
    EdwardsEvent start,stop;
    const auto dictionary_start=std::chrono::steady_clock::now();
    check(cudaEventRecord(start.value));
    k.prepare(report.p,data.p,dictionary.p,entries,count);
    check(cudaGetLastError());check(cudaEventRecord(stop.value));stop.wait();
    if(cgbn_error_report_check(report.p))throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Device,"CGBN Edwards dictionary arithmetic error");
    if(metrics)check(cudaEventElapsedTime(&metrics->dictionary_gpu_ms,start.value,stop.value));
    if(metrics)metrics->dictionary_wall_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-dictionary_start).count();
    if(prepared)prepared(context);
    EcmEdwardsSlice slices(chunk);
    while(done<plan.bits) {
        const uint32_t end=done+slices.columns(plan.bits-done);
        check(cudaEventRecord(start.value));
        k.naf(report.p,data.p,dictionary.p,result.p,
            nonzero.p,codes.p,plan.bits,done,end,entries,plan.window,count,doubling);
        check(cudaGetLastError());check(cudaEventRecord(stop.value));stop.wait();
        if(cgbn_error_report_check(report.p))throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Device,"CGBN Edwards NAF arithmetic error");
        if(metrics || !chunk){float ms=0;check(cudaEventElapsedTime(&ms,start.value,stop.value));
            slices.observe(end-done,ms);
            if(metrics){metrics->loop_gpu_ms+=ms;++metrics->launches;}}
        check(cudaMemcpy(ordinary.data(),result.p,ordinary.size()*sizeof(Mem),cudaMemcpyDeviceToHost));
        for(uint32_t i=0;i<count;++i)for(unsigned j=0;j<4;++j)
            mpz_import(states[size_t(i)*9+j+5],P::BITS/32,-1,4,0,0,ordinary[size_t(i)*4+j]._limbs);
        done=end;if(cb && !cb(done,context))return 2;
    }
    return 0;
}
}
namespace ecm_edwards_cuda_detail {
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) \
int run_##W(const Request &r) { \
    static_assert(HostParams<W,T>::TPB==ET,"Edwards host block geometry"); \
    const auto &k=kernels_##W(); \
    if(r.plan)return run_naf<HostParams<W,T>>(r.n,*r.plan,r.states,r.count,r.done,r.chunk,r.progress,r.context,r.doubling,r.metrics,r.prepared,k); \
    return run<HostParams<W,T>>(r.n,r.scalar,r.states,r.count,r.done,r.chunk,r.progress,r.context,r.prepared,k); \
}
#include "../../../cuda/dispatch/tiers.def"
#undef ECM_CUDA_TIER
}
