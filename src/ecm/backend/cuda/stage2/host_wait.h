#pragma once
#include "host_wait_policy.h"
#include "../../../platform/ecm_stage2_logging.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace stage2_wait {
enum class Kind {timing, oracle, upload, output, canonical, readback, count};
inline bool option(const char *name,bool fallback) {
    const char *value=std::getenv(name);
    if(!value)return fallback;
    if(!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error(std::string(name)+" must be 0 or 1");
}
struct Runtime {
    bool adaptive=option("NTT_HOST_EVENT_WAIT",false);
    bool fifo=option("NTT_S4_TIMING_FIFO",true);
    bool readback=option("NTT_HOST_READBACK_WAIT",false);
    Counters stats[static_cast<unsigned>(Kind::count)]{};
    // An unnamed, non-inheritable per-thread timer; no global timer-resolution
    // changes. Unsupported/failed timers fall back to the native CUDA wait.
    void *timer=nullptr;
    cudaEvent_t readback_event=nullptr;
    Runtime();
    ~Runtime();
    bool pause(unsigned us);
    Runtime(const Runtime&)=delete;
    Runtime& operator=(const Runtime&)=delete;
};
inline Runtime &runtime(){static thread_local Runtime value;return value;}
inline bool timing_fifo(){return runtime().fifo;}
inline void reset(){for(auto &stats:runtime().stats)stats=Counters{};}
inline cudaError_t event_sync(cudaEvent_t event,Kind kind=Kind::timing) {
    auto &state=runtime();auto &stats=state.stats[static_cast<unsigned>(kind)];
    struct Backend {
        Runtime &state;cudaEvent_t event;
        const cudaError_t success=cudaSuccess,not_ready=cudaErrorNotReady;
        cudaError_t query(){return cudaEventQuery(event);}
        cudaError_t synchronize(){return cudaEventSynchronize(event);}
        bool pause(unsigned us){return state.pause(us);}
    } backend{state,event};
    const auto start=std::chrono::steady_clock::now();
    const auto status=complete_event(backend,state.adaptive,stats);
    stats.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return status;
}
inline cudaError_t before_readback() {
    auto &state=runtime();
    if(!state.readback)return cudaSuccess;
    const auto ready=cudaStreamQuery(nullptr);
    if(ready!=cudaErrorNotReady)return ready;
    if(!state.readback_event) {
        const auto status=cudaEventCreateWithFlags(&state.readback_event,cudaEventDisableTiming);
        if(status!=cudaSuccess)return status;
    }
    const auto status=cudaEventRecord(state.readback_event);
    return status==cudaSuccess ? event_sync(state.readback_event,Kind::readback) : status;
}
inline cudaError_t copy(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind) {
    if(bytes && kind==cudaMemcpyDeviceToHost) {
        const auto status=before_readback();
        if(status!=cudaSuccess)return status;
    }
    return cudaMemcpy(dst,src,bytes,kind);
}
inline void report() {
    const auto &state=runtime();Counters total;
    for(const auto &s:state.stats) {
        total.calls+=s.calls;total.ready+=s.ready;total.pending+=s.pending;
        total.queries+=s.queries;total.sleeps+=s.sleeps;total.sleep_us+=s.sleep_us;
        total.fallbacks+=s.fallbacks;total.seconds+=s.seconds;
    }
    stage2_log::print(stage2_log::debug,
        "stage2_host_wait_stats: adaptive=%d timing_fifo=%d readback=%d calls=%llu ready=%llu pending=%llu queries=%llu "
        "sleeps=%llu requested_sleep_us=%llu fallbacks=%llu wall_seconds=%.6f "
        "timing_seconds=%.6f oracle_seconds=%.6f upload_seconds=%.6f output_seconds=%.6f canonical_seconds=%.6f readback_seconds=%.6f scope=engine_events\n",
        (int)state.adaptive,(int)state.fifo,(int)state.readback,(unsigned long long)total.calls,(unsigned long long)total.ready,
        (unsigned long long)total.pending,(unsigned long long)total.queries,(unsigned long long)total.sleeps,
        (unsigned long long)total.sleep_us,(unsigned long long)total.fallbacks,total.seconds,
        state.stats[0].seconds,state.stats[1].seconds,state.stats[2].seconds,state.stats[3].seconds,state.stats[4].seconds,state.stats[5].seconds);
}
}
