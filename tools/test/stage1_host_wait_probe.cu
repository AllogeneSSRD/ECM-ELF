// Diagnostic only: instantiate a production kernel and substitute
// event construction, keeping its arithmetic, chunking and readback unchanged.
#include <gmp.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#define NOMINMAX
#include <windows.h>
#include "ecm/io/save.h"
#include "ecm/stage1/scalar.h"

static bool diagnostic_blocking_event=false;
static unsigned diagnostic_pre_sync=0,diagnostic_event_flags=0;
static cudaEvent_t diagnostic_stop_event=nullptr;
static unsigned diagnostic_events=0;
static double process_cpu();
struct ApiTime {double wall=0,cpu=0;unsigned calls=0;};
static ApiTime diagnostic_api[6];
static std::vector<float> diagnostic_intervals;
template<class F> static cudaError_t timed(unsigned index,F call) {
    const auto start=std::chrono::steady_clock::now();const double cpu=process_cpu();
    const auto result=call();auto &m=diagnostic_api[index];
    m.wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    m.cpu+=process_cpu()-cpu;++m.calls;return result;
}
static cudaError_t diagnostic_event_record(cudaEvent_t event,cudaStream_t stream=0) {
    if(diagnostic_pre_sync==1 && event==diagnostic_stop_event)
        {const auto rc=timed(0,[]{return cudaDeviceSynchronize();});if(rc!=cudaSuccess)return rc;}
    return timed(1,[&]{return cudaEventRecord(event,stream);});
}
static cudaError_t diagnostic_event_sync(cudaEvent_t event) {
    return timed(2,[&]{return cudaEventSynchronize(event);});
}
// The reference control suppresses only the production host device wait.
// Its event wait still completes all GPU work before report/readback.
static cudaError_t diagnostic_device_sync() {
    return diagnostic_pre_sync==2?timed(0,[]{return cudaDeviceSynchronize();}):cudaSuccess;
}
static cudaError_t diagnostic_memcpy(void *dest,const void *src,size_t size,cudaMemcpyKind kind) {return timed(3,[&]{return cudaMemcpy(dest,src,size,kind);});}
static cudaError_t diagnostic_last_error() {return timed(4,[]{return cudaGetLastError();});}
static cudaError_t diagnostic_elapsed(float *ms,cudaEvent_t start,cudaEvent_t stop) {
    const auto rc=timed(5,[&]{return cudaEventElapsedTime(ms,start,stop);});
    if(rc==cudaSuccess)diagnostic_intervals.push_back(*ms);return rc;
}
static cudaError_t diagnostic_event_create(cudaEvent_t *event) {
    const auto rc=cudaEventCreateWithFlags(event,diagnostic_blocking_event?cudaEventBlockingSync:cudaEventDefault);
    if(rc!=cudaSuccess)return rc;
    // Record flags accepted by cudaEventCreateWithFlags, not driver introspection.
    diagnostic_event_flags|=diagnostic_blocking_event?cudaEventBlockingSync:cudaEventDefault;
    if(++diagnostic_events==2)diagnostic_stop_event=*event;return rc;
}
#include "../../src/ecm/edwards_stage1/cuda/host_wait.h"
#include "ecm/stage1/scalar.h"

static void require(bool value,const char *why){if(!value)throw std::runtime_error(why);}
static double process_cpu() {
    FILETIME created{},exited{},kernel{},user{};
    require(GetProcessTimes(GetCurrentProcess(),&created,&exited,&kernel,&user)!=0,"GetProcessTimes failed");
    const auto ticks=[](FILETIME t){return (uint64_t(t.dwHighDateTime)<<32)|t.dwLowDateTime;};
    return double(ticks(kernel)+ticks(user))*1e-7;
}
int main(int argc,char **argv) {
    try {
        if(argc<7 || argc>11)throw std::runtime_error("usage: probe DEVICE CONTEXT_FLAGS EVENT_BLOCKING SCALAR_BITS CURVES POINTS_FILE [WAIT_ORDER:0=event,1=before-record,2=before-event-wait] [EXPONENT=1279] [CHUNK=256] [B1=0]");
        const int device=std::stoi(argv[1]);const unsigned flags=std::stoul(argv[2]);
        diagnostic_blocking_event=std::stoi(argv[3])!=0;
        diagnostic_pre_sync=argc>=8?std::stoul(argv[7]):0;
        require(diagnostic_pre_sync<=2,"invalid wait order");
        const unsigned bits=std::stoul(argv[4]),count=std::stoul(argv[5]);
        const unsigned exponent=argc>=9?std::stoul(argv[8]):1279;
        const unsigned chunk=argc>=10?std::stoul(argv[9]):256;
        const uint64_t b1=argc>=11?std::stoull(argv[10]):0;
        require(device==1 && (flags==0 || flags==4) && bits>=32 && bits<=1000000 && count && count<=1024,"invalid diagnostic controls");
        require(exponent>=1279 && exponent<ECM_HOST_WAIT_TIER && chunk<=4096 && b1<=10000000,"invalid arithmetic controls");
        require(cudaSetDevice(device)==cudaSuccess,"cudaSetDevice failed");
        require(cudaSetDeviceFlags(flags)==cudaSuccess,"cudaSetDeviceFlags failed");
        require(cudaFree(nullptr)==cudaSuccess,"context warmup failed");
        unsigned actual_flags=0;require(cudaGetDeviceFlags(&actual_flags)==cudaSuccess,"cannot inspect context flags");
        using ecm_edwards::Integer;
        Integer n,scalar,d,x,y,a,f,t;
        mpz_ui_pow_ui(n.v,2,exponent);mpz_sub_ui(n.v,n.v,1);
        // A deterministic long scalar avoids measuring the CPU exponent builder.
        if(b1)require(ecm_build_lcm_exponent(scalar.v,b1,48),"exponent builder failed");
        else {mpz_ui_pow_ui(scalar.v,2,bits);mpz_sub_ui(scalar.v,scalar.v,1);}
        ecm_edwards::Array states(size_t(count)*9);
        for(unsigned i=0;i<count;++i) {
            require(ecm_edwards::construct(d.v,x.v,y.v,a.v,f.v,17+i,n.v)==ecm_edwards::Setup::Ready,"curve setup failed");
            auto *s=states.data()+size_t(i)*9;
            mpz_set(s[0],d.v);mpz_set(s[1],x.v);mpz_set(s[2],y.v);mpz_set_ui(s[3],1);
            mpz_mul(s[4],x.v,y.v);mpz_mod(s[4],s[4],n.v);mpz_set_ui(s[6],1);mpz_set_ui(s[7],1);
        }
        EcmEdwardsNaf plan(scalar.v,6);EcmEdwardsNafMetrics metrics;uint32_t done=0;
        const ecm_cuda_diagnostics::HostWaitHooks hooks{diagnostic_event_create,diagnostic_event_record,
            diagnostic_event_sync,diagnostic_device_sync,diagnostic_memcpy,diagnostic_last_error,diagnostic_elapsed};
        const auto start=std::chrono::steady_clock::now();const auto cpu=process_cpu();
        const int rc=ecm_cuda_diagnostics::host_wait_run(hooks,n.v,scalar.v,states.data(),count,done,chunk,plan,4,&metrics);
        const auto cpu_seconds=process_cpu()-cpu;
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        require(rc==0 && done==plan.bits,"incomplete arithmetic");
        std::ofstream points(argv[6],std::ios::binary);
        for(unsigned i=0;i<count;++i) {
            auto *s=states.data()+size_t(i)*9;
            require(ecm_edwards::point_valid(n.v,s[0],s+5),"point invariant failed");
            for(unsigned j=5;j<9;++j) {
                std::vector<char> value(mpz_sizeinbase(s[j],16)+3);mpz_get_str(value.data(),16,s[j]);points<<value.data()<<'\n';
            }
        }
        points.close();require(bool(points),"cannot save final points");
        std::printf("{\"exponent\":%u,\"container_bits\":%u,\"tpi\":%u,\"chunk\":%u,\"b1\":%llu,\"additions\":%u,\"context_flags\":%u,\"event_blocking\":%s,\"wait_order\":%u,\"created_event_flags\":%u,\"event_creations\":%u,\"wall_seconds\":%.9f,\"cpu_seconds\":%.9f,\"cpu_cores\":%.6f,\"gpu_seconds\":%.9f,\"dictionary_ms\":%.6f,\"launches\":%u,\"columns\":%u,\"invariant_ok\":true,\"event_intervals_ms\":[",
            exponent,metrics.container_bits,metrics.tpi,chunk,(unsigned long long)b1,plan.additions,
            actual_flags,diagnostic_blocking_event?"true":"false",diagnostic_pre_sync,diagnostic_event_flags,diagnostic_events,seconds,cpu_seconds,cpu_seconds/seconds,
            (metrics.dictionary_gpu_ms+metrics.loop_gpu_ms)/1000.,metrics.dictionary_gpu_ms,metrics.launches,done);
        for(size_t i=0;i<diagnostic_intervals.size();++i)std::printf("%s%.6f",i?",":"",diagnostic_intervals[i]);
        std::printf("],\"api\":{");
        const char *names[]={"device_sync","event_record","event_sync","memcpy","last_error","elapsed_time"};
        for(unsigned i=0;i<6;++i)std::printf("%s\"%s\":{\"wall\":%.9f,\"cpu\":%.9f,\"calls\":%u}",i?",":"",names[i],diagnostic_api[i].wall,diagnostic_api[i].cpu,diagnostic_api[i].calls);
        std::printf("}}\n");
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"probe: %s\n",e.what());return 1;}
}
