#pragma once
#define ECM_STAGE2_MEMORY_LEDGER 1
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>
#include <limits>
#include <cstdlib>
#include <cstdio>

// Successful cudaMalloc payloads owned by this translation unit, including
// short-lived allocations. This is not driver usage: context/module overhead,
// other processes and pinned HOST memory are deliberately outside the contract.
namespace stage2_memory {
inline thread_local bool persistent_scope=false;
struct PersistentScope {
    bool previous=persistent_scope;
    PersistentScope(){persistent_scope=true;}
    ~PersistentScope(){persistent_scope=previous;}
};
struct Ledger {
    struct Entry {size_t bytes;std::string site;bool persistent;};
    std::map<void*,Entry> live;
    std::map<std::string,size_t> peak_sites;
    std::map<std::string,size_t> interval_sites;
    size_t bytes=0,peak=0,allocations=0,frees=0,failures=0,unknown_frees=0;
    size_t interval_peak=0,baseline_bytes=0,baseline_allocations=0;
    bool allocate(void *pointer,size_t size,const std::string &site,bool persistent=false) {
        if(!pointer || !size)return true;
        if(live.count(pointer) || size>std::numeric_limits<size_t>::max()-bytes)return false;
        live.emplace(pointer,Entry{size,site,persistent});bytes+=size;++allocations;
        if(bytes>peak) {
            peak=bytes;peak_sites.clear();
            for(const auto &entry:live)peak_sites[entry.second.site]+=entry.second.bytes;
        }
        if(bytes>interval_peak){interval_peak=bytes;interval_sites=sites();}
        return true;
    }
    void release(void *pointer) {
        if(!pointer)return;
        const auto entry=live.find(pointer);
        if(entry==live.end()){++unknown_frees;return;}
        bytes-=entry->second.bytes;live.erase(entry);++frees;
    }
    std::map<std::string,size_t> sites() const {
        std::map<std::string,size_t> result;
        for(const auto &entry:live)result[entry.second.site]+=entry.second.bytes;
        return result;
    }
};
inline bool enabled() {
    static const bool value=[] {const char *e=std::getenv("NTT_MEMORY_LEDGER");return e && std::atoi(e)!=0;}();
    return value;
}
inline Ledger ledger;
inline std::mutex lock;
inline std::string source(const char *file,int line) {
    std::string name=file;const auto pos=name.find_last_of("/\\");
    if(pos!=std::string::npos)name.erase(0,pos+1);
    return name+":"+std::to_string(line);
}
template<class T> cudaError_t allocate(T **pointer,size_t bytes,const char *file,int line) {
    const auto result=::cudaMalloc(reinterpret_cast<void**>(pointer),bytes);
    if(!enabled())return result;
    std::lock_guard<std::mutex> guard(lock);
    if(result!=cudaSuccess){++ledger.failures;return result;}
    if(!ledger.allocate(*pointer,bytes,source(file,line),persistent_scope)) {
        std::fprintf(stderr,"FATAL: Stage2 memory ledger duplicate/overflow at %s:%d\n",file,line);std::exit(3);
    }
    return result;
}
inline cudaError_t release(void *pointer) {
    const auto result=::cudaFree(pointer);
    if(enabled() && result==cudaSuccess) {
        std::lock_guard<std::mutex> guard(lock);ledger.release(pointer);
    }
    return result;
}
inline void snapshot(const char *name) {
    if(!enabled())return;
    std::lock_guard<std::mutex> guard(lock);
    size_t persistent_bytes=0,persistent_allocations=0;
    for(const auto &entry:ledger.live)if(entry.second.persistent){persistent_bytes+=entry.second.bytes;++persistent_allocations;}
    stage2_log::print(stage2_log::debug,
        "stage2_memory_ledger: snapshot=%s live_bytes=%llu peak_bytes=%llu allocations=%llu frees=%llu "
        "failed_allocations=%llu unknown_frees=%llu live_allocations=%llu persistent_bytes=%llu persistent_allocations=%llu "
        "baseline_bytes=%llu baseline_allocations=%llu interval_peak_bytes=%llu payload_only=1 version=1\n",
        name,(unsigned long long)ledger.bytes,(unsigned long long)ledger.peak,
        (unsigned long long)ledger.allocations,(unsigned long long)ledger.frees,
        (unsigned long long)ledger.failures,(unsigned long long)ledger.unknown_frees,
        (unsigned long long)ledger.live.size(),(unsigned long long)persistent_bytes,
        (unsigned long long)persistent_allocations,(unsigned long long)ledger.baseline_bytes,
        (unsigned long long)ledger.baseline_allocations,(unsigned long long)ledger.interval_peak);
    for(const auto &site:ledger.sites())stage2_log::print(stage2_log::debug,
        "stage2_memory_site: snapshot=%s scope=live site=%s bytes=%llu\n",
        name,site.first.c_str(),(unsigned long long)site.second);
    for(const auto &site:ledger.interval_sites)stage2_log::print(stage2_log::debug,
        "stage2_memory_site: snapshot=%s scope=interval_peak site=%s bytes=%llu\n",
        name,site.first.c_str(),(unsigned long long)site.second);
    if(std::string(name)=="final")for(const auto &site:ledger.peak_sites)stage2_log::print(stage2_log::debug,
        "stage2_memory_site: snapshot=%s scope=global_peak site=%s bytes=%llu\n",
        name,site.first.c_str(),(unsigned long long)site.second);
    ledger.interval_peak=ledger.bytes;ledger.interval_sites=ledger.sites();
}
struct Session {
    Session() {
        if(!enabled())return;
        std::lock_guard<std::mutex> guard(lock);
        for(const auto &entry:ledger.live)if(!entry.second.persistent) {
            std::fprintf(stderr,"FATAL: Stage2 curve-local allocation live at session start\n");std::exit(3);
        }
        auto retained=std::move(ledger.live);ledger=Ledger{};ledger.live=std::move(retained);
        for(const auto &entry:ledger.live)ledger.bytes+=entry.second.bytes;
        ledger.baseline_bytes=ledger.bytes;ledger.baseline_allocations=ledger.live.size();
        ledger.peak=ledger.interval_peak=ledger.bytes;ledger.peak_sites=ledger.interval_sites=ledger.sites();
    }
    ~Session(){snapshot("final");}
};
inline void fixture() {
    Ledger test;size_t checks=0,bad=0;
    auto check=[&](bool ok){++checks;if(!ok)++bad;};
    int tokens[4]{};
    check(test.allocate(tokens,100,"a"));check(test.allocate(tokens+1,200,"b"));
    check(test.bytes==300 && test.peak==300 && test.peak_sites.at("a")==100);
    check(!test.allocate(tokens,1,"duplicate"));check(test.bytes==300);
    test.release(tokens);check(test.bytes==200 && test.peak==300);
    check(test.allocate(tokens,50,"a"));check(test.bytes==250 && test.peak_sites.at("a")==100);
    check(test.allocate(tokens+2,75,"a"));check(test.bytes==325 && test.peak_sites.at("a")==125);
    size_t sum=0;for(const auto &site:test.sites())sum+=site.second;check(sum==test.bytes);
    sum=0;for(const auto &site:test.peak_sites)sum+=site.second;check(sum==test.peak);
    test.release(nullptr);check(test.unknown_frees==0);
    test.release(tokens+3);check(test.unknown_frees==1 && test.bytes==325);
    test.release(tokens);test.release(tokens+1);test.release(tokens+2);
    check(test.bytes==0 && test.live.empty() && test.allocations==4 && test.frees==4);
    check(test.allocate(tokens,std::numeric_limits<size_t>::max(),"maximum"));
    check(!test.allocate(tokens+1,1,"overflow"));test.release(tokens);
    check(test.bytes==0);
    check(test.allocate(tokens,10,"persistent",true));
    check(test.live.at(tokens).persistent);test.release(tokens);
    stage2_log::print(stage2_log::debug,"stage2_memory_ledger_check: checks=%llu bad=%llu\n",
        (unsigned long long)checks,(unsigned long long)bad);
    if(bad)std::exit(3);
}
}

// Defined after CUDA declarations and wrapper bodies. Every engine/NTT include
// below this header uses these two entry points; CUDA headers are already guarded.
// No async/managed/pitched allocator is used in the production source closure.
#define cudaMalloc(pointer,bytes) stage2_memory::allocate((pointer),(bytes),__FILE__,__LINE__)
#define cudaFree(pointer) stage2_memory::release((pointer))
