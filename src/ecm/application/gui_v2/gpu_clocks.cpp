#include "gpu_clocks.h"
#include "../gui/platform.h"
#include <windows.h>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <sstream>

namespace ecmgui_v2 {
std::string system_clock_query_tool(){
    wchar_t dir[MAX_PATH]{};const auto size=GetSystemDirectoryW(dir,MAX_PATH);
    if(!size||size>=MAX_PATH)return {};
    const auto file=std::filesystem::path(dir)/L"nvidia-smi.exe";
    std::error_code error;return std::filesystem::is_regular_file(file,error)?file.u8string():std::string{};
}
bool clock_uuid(const std::string& id){
    if(id.size()!=40||id.substr(0,4)!="GPU-")return false;
    for(size_t i=4;i<id.size();++i){
        if(i==12||i==17||i==22||i==27){if(id[i]!='-')return false;}
        else if(!((id[i]>='0'&&id[i]<='9')||(id[i]>='a'&&id[i]<='f')))return false;
    }return true;
}
bool parse_clock_profiles(const std::string& text,ClockProfiles& out,std::set<std::string>* automatic){
    ClockProfiles parsed;
    std::set<std::string> autos;
    if(text.size()>16384)return false;
    size_t begin=0;
    while(begin<text.size()){
        const auto end=text.find(';',begin),limit=end==std::string::npos?text.size():end;
        const auto colon=text.find(':',begin),dash=text.find('-',begin+41);
        if(colon!=begin+40||dash==std::string::npos||dash>=limit||!clock_uuid(text.substr(begin,40)))return false;
        const auto integer=[&](size_t a,size_t b){int n=0;if(a==b||b-a>6)return 0;for(;a<b;++a){if(text[a]<'0'||text[a]>'9')return 0;n=n*10+text[a]-'0';}return n;};
        const auto flag=text.find(':',dash+1);const bool has_flag=flag<limit;
        if(has_flag&&text.substr(flag,limit-flag)!=":auto")return false;
        ClockRange range{integer(colon+1,dash),integer(dash+1,has_flag?flag:limit)};
        if(!range.valid()||!parsed.emplace(text.substr(begin,40),range).second)return false;
        if(has_flag)autos.insert(text.substr(begin,40));
        if(end==std::string::npos)break;
        begin=end+1;if(begin==text.size())return false;
    }
    out=std::move(parsed);if(automatic)*automatic=std::move(autos);return true;
}
std::string encode_clock_profiles(const ClockProfiles& profiles,const std::set<std::string>& automatic){
    std::string result;
    for(const auto& entry:profiles){
        if(!clock_uuid(entry.first)||!entry.second.valid())return {};
        if(!result.empty())result+=';';result+=entry.first+":"+std::to_string(entry.second.low)+"-"+std::to_string(entry.second.high);
        if(automatic.count(entry.first))result+=":auto";
    }return result;
}
namespace {
bool command_range(const std::string& tool,const std::string& uuid,ClockRange& out){
    if(tool.empty()||tool.find('"')!=std::string::npos||!clock_uuid(uuid))return false;
    std::string output;int exit=-1;
    if(!ecmgui::run_capture("\""+tool+"\" -i "+uuid+" --query-supported-clocks=graphics --format=csv,noheader,nounits",output,exit,5000)||exit||output.size()>262144)return false;
    ClockRange parsed;std::istringstream lines(output);std::string line;
    while(std::getline(lines,line)){
        const auto a=line.find_first_not_of(" \t\r"),b=line.find_last_not_of(" \t\r");if(a==std::string::npos)continue;
        if(b-a>=6)return false;int clock=0;
        for(size_t i=a;i<=b;++i){if(line[i]<'0'||line[i]>'9')return false;clock=clock*10+line[i]-'0';}
        if(clock<=0||clock>100000)return false;
        parsed.low=parsed.low?std::min(parsed.low,clock):clock;parsed.high=std::max(parsed.high,clock);
    }
    if(!parsed.valid())return false;out=parsed;return true;
}
struct Nvml {
    HMODULE module=nullptr;
    int init_error=9;
    int (*shutdown)()=nullptr;
    int (*index)(unsigned,void**)=nullptr;
    int (*by_uuid)(const char*,void**)=nullptr;
    int (*uuid)(void*,char*,unsigned)=nullptr;
    int (*memory)(void*,unsigned*,unsigned*)=nullptr;
    int (*graphics)(void*,unsigned,unsigned*,unsigned*)=nullptr;
    int (*pstate)(void*,int,int,unsigned*,unsigned*)=nullptr;
    int (*set)(void*,unsigned,unsigned)=nullptr;
    int (*reset)(void*)=nullptr;
    const char* (*error_text)(int)=nullptr;
    template<class F> void symbol(F& value,const char* name){value=reinterpret_cast<F>(GetProcAddress(module,name));}
    explicit Nvml(const std::string& source){
        if(source.empty())return;
        module=LoadLibraryW(std::filesystem::u8path(source).c_str());if(!module)return;
        int (*init)()=nullptr;symbol(init,"nvmlInit_v2");symbol(shutdown,"nvmlShutdown");
        if(!init||!shutdown){shutdown=nullptr;init_error=13;return;}
        init_error=init();if(init_error){shutdown=nullptr;return;}
        symbol(index,"nvmlDeviceGetHandleByIndex_v2");symbol(by_uuid,"nvmlDeviceGetHandleByUUID");symbol(uuid,"nvmlDeviceGetUUID");
        symbol(memory,"nvmlDeviceGetSupportedMemoryClocks");symbol(graphics,"nvmlDeviceGetSupportedGraphicsClocks");
        symbol(pstate,"nvmlDeviceGetMinMaxClockOfPState");symbol(set,"nvmlDeviceSetGpuLockedClocks");symbol(reset,"nvmlDeviceResetGpuLockedClocks");
        symbol(error_text,"nvmlErrorString");
    }
    ~Nvml(){if(shutdown)shutdown();if(module)FreeLibrary(module);}
    int identify(int ordinal,const std::string& expected,void*& device,std::string& actual){
        if(init_error)return init_error;
        if(!uuid||(!expected.empty()&&!by_uuid)||(expected.empty()&&!index))return 13;
        int code=expected.empty()?index(unsigned(ordinal),&device):by_uuid(expected.c_str(),&device);if(code)return code;
        char id[96]{};if((code=uuid(device,id,sizeof(id))))return code;
        actual=id;
        if(!clock_uuid(actual)||(!expected.empty()&&expected!=actual))return 2;
        return 0;
    }
    int range(void* device,ClockRange& out,bool& fallback,std::string& context){
        unsigned count=0;int code=3;
        if(memory&&graphics){
            context="nvmlDeviceGetSupportedMemoryClocks(count)";
            code=memory(device,&count,nullptr);
            if((code==0||code==7)&&count&&count<=256){
                std::vector<unsigned> memories(count);const unsigned capacity=count;
                context="nvmlDeviceGetSupportedMemoryClocks(values)";
                code=memory(device,&count,memories.data());
                if(code==0&&count<=capacity)for(unsigned i=0;i<count;++i){
                    context="nvmlDeviceGetSupportedGraphicsClocks(count)";
                    unsigned n=0;code=graphics(device,memories[i],&n,nullptr);
                    if((code!=0&&code!=7)||!n||n>4096)continue;
                    std::vector<unsigned> values(n);const unsigned cap=n;
                    context="nvmlDeviceGetSupportedGraphicsClocks(values)";
                    code=graphics(device,memories[i],&n,values.data());if(code||n>cap)continue;
                    for(unsigned j=0;j<n;++j)if(values[j]>0&&values[j]<=100000){
                        out.low=out.low?std::min(out.low,int(values[j])):int(values[j]);out.high=std::max(out.high,int(values[j]));
                    }
                }
                if(out.valid())return 0;
            }
        }
        if(pstate)for(int state=0;state<16;++state){
            unsigned low=0,high=0;const int r=pstate(device,0,state,&low,&high); // NVML_CLOCK_GRAPHICS, P0..P15.
            if(!r&&ClockRange{int(low),int(high)}.valid()){
                fallback=true;out.low=out.low?std::min(out.low,int(low)):int(low);out.high=std::max(out.high,int(high));
            }
        }
        return out.valid()?0:(code?code:3);
    }
};
}
GpuClocks::~GpuClocks(){
    {std::lock_guard<std::mutex> lock(mutex_);stop_=true;queue_.clear();}wake_.notify_one();
    if(thread_.joinable())thread_.join();
}
void GpuClocks::start(const std::string& source,const std::string& tool){if(!thread_.joinable())thread_=std::thread([this,source,tool]{run(source,tool);});}
ClockSnapshot GpuClocks::get(int index)const{
    std::lock_guard<std::mutex> lock(mutex_);const auto found=snapshots_.find(index);return found==snapshots_.end()?ClockSnapshot{}:found->second;
}
bool GpuClocks::enqueue(Request request){
    std::lock_guard<std::mutex> lock(mutex_);
    if(stop_||!thread_.joinable()||snapshots_[request.index].busy)return false;
    auto& snapshot=snapshots_[request.index];
    if(request.action!=ClockAction::Query){
        if(!clock_uuid(request.uuid)||request.uuid!=snapshot.uuid||!snapshot.queried)return false;
        if(request.action==ClockAction::Apply&&(!snapshot.can_apply||!request.range.valid()||request.range.low<snapshot.supported.low||request.range.high>snapshot.supported.high))return false;
        if(request.action==ClockAction::Reset&&!snapshot.can_reset)return false;
    }
    snapshot.busy=true;queue_.push_back(std::move(request));wake_.notify_one();return true;
}
bool GpuClocks::query(int index,const std::string& uuid){return enqueue({index,uuid,ClockAction::Query,{}});}
bool GpuClocks::apply(int index,const std::string& uuid,ClockRange range){return enqueue({index,uuid,ClockAction::Apply,range});}
bool GpuClocks::reset(int index,const std::string& uuid){return enqueue({index,uuid,ClockAction::Reset,{}});}
void GpuClocks::run(const std::string& source,const std::string& tool){
    try{
        Nvml api(source);
        for(;;){
            Request request;ClockSnapshot result;
            {std::unique_lock<std::mutex> lock(mutex_);wake_.wait(lock,[&]{return stop_||!queue_.empty();});if(stop_)return;
             request=std::move(queue_.front());queue_.pop_front();result=snapshots_[request.index];}
            result.error=0;result.detail.clear();result.action=request.action;std::string context;
            void* device=nullptr;std::string identity;
            result.error=api.identify(request.index,request.uuid,device,identity);
            if(!result.error){
                if(identity!=result.uuid){result.applied_known=false;result.applied={};result.reset_accepted=false;}
                result.uuid=identity;
                const auto limits=[&](ClockRange& range,bool& fallback,bool& smi){
                    int code=api.range(device,range,fallback,context);
                    if(code&&command_range(tool,identity,range)){smi=true;fallback=false;code=0;}
                    return code;
                };
                if(request.action==ClockAction::Query){
                    result.supported={};result.pstate_range=false;result.smi_range=false;
                    result.error=limits(result.supported,result.pstate_range,result.smi_range);
                    result.can_apply=api.set&&api.by_uuid&&result.supported.valid();result.can_reset=api.reset&&api.by_uuid;
                }else if(request.action==ClockAction::Apply){
                    // Refresh limits before a hardware write; cached bounds alone are not authority.
                    ClockRange current;bool fallback=false,smi=false;result.error=limits(current,fallback,smi);
                    if(!result.error&&(request.range.low<current.low||request.range.high>current.high)){context="requested GPU clock range";result.error=2;}
                    if(!result.error){context="nvmlDeviceSetGpuLockedClocks";result.error=api.set?api.set(device,unsigned(request.range.low),unsigned(request.range.high)):13;}
                    if(!result.error){result.applied=request.range;result.applied_known=true;result.reset_accepted=false;}
                }else{
                    context="nvmlDeviceResetGpuLockedClocks";result.error=api.reset?api.reset(device):13;
                    if(!result.error){result.applied={};result.applied_known=true;result.reset_accepted=true;}
                }
            }else{result.can_apply=false;result.can_reset=false;}
            if(result.error&&api.error_text){const auto* detail=api.error_text(result.error);if(detail)result.detail=(context.empty()?"":context+": ")+detail;}
            result.queried=true;result.busy=false;++result.completed;
            {std::lock_guard<std::mutex> lock(mutex_);snapshots_[request.index]=std::move(result);}
        }
    }catch(...){
        std::lock_guard<std::mutex> lock(mutex_);for(auto& entry:snapshots_){entry.second.busy=false;entry.second.queried=true;entry.second.error=999;++entry.second.completed;}stop_=true;queue_.clear();
    }
}
}
