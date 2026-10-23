#pragma once
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <set>
#include <thread>

namespace ecmgui_v2 {
struct ClockRange {
    int low=0,high=0;
    bool valid() const{return low>0&&low<=high&&high<=100000;}
    bool operator==(const ClockRange& b)const{return low==b.low&&high==b.high;}
    bool operator!=(const ClockRange& b)const{return !(*this==b);}
};
using ClockProfiles=std::map<std::string,ClockRange>;
bool clock_uuid(const std::string& uuid);
// A schema-backed TOML string: GPU-<UUID>:min-max[:auto];GPU-<UUID>:min-max.
// Ordinary entries load as drafts; the explicit :auto opt-in seeds automation.
bool parse_clock_profiles(const std::string& text,ClockProfiles& out,std::set<std::string>* automatic=nullptr);
std::string encode_clock_profiles(const ClockProfiles& profiles,const std::set<std::string>& automatic={});
inline const char* clock_error_key(int code){
    return code==4?"gpu.clock_permission":code==3||code==13?"gpu.clock_unsupported":code==2?"gpu.clock_invalid":code==6||code==15?"gpu.clock_disconnected":"gpu.clock_failed";
}
std::string system_clock_query_tool();

enum class ClockAction {Query,Apply,Reset};
struct ClockSnapshot {
    bool queried=false,busy=false,can_apply=false,can_reset=false;
    bool pstate_range=false,smi_range=false;
    std::string uuid,detail;
    ClockRange supported,applied;
    bool applied_known=false,reset_accepted=false;
    int error=0;
    ClockAction action=ClockAction::Query;
    unsigned completed=0;
};
// Serial driver work on a dedicated thread. Rendering only reads snapshots or
// enqueues explicit commands. The service never restores or resets on shutdown.
class GpuClocks {
public:
    ~GpuClocks();
    void start(const std::string& nvml_source,const std::string& query_tool={});
    bool query(int index,const std::string& expected_uuid);
    bool apply(int index,const std::string& uuid,ClockRange range);
    bool reset(int index,const std::string& uuid);
    ClockSnapshot get(int index)const;
private:
    struct Request {int index;std::string uuid;ClockAction action;ClockRange range;};
    bool enqueue(Request request);
    void run(const std::string& source,const std::string& query_tool);
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Request> queue_;
    std::map<int,ClockSnapshot> snapshots_;
    bool stop_=false;
    std::thread thread_;
};
}
