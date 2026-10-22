#include "gpu_details.h"
#include <windows.h>
#include <array>
#include <cstdio>
#include <filesystem>

namespace ecmgui_v2 {
namespace {
struct Module {
    HMODULE handle;
    ~Module(){if(handle)FreeLibrary(handle);}
    template<class F> F function(const char* name) const{return reinterpret_cast<F>(GetProcAddress(handle,name));}
};
struct NvmlSession {
    int (*shutdown)();
    ~NvmlSession(){if(shutdown)shutdown();}
};
// NVML GPU UUID and CUDA's 16-byte physical UUID identify the same adapter;
// never pair NVML and CUDA enumeration indices on a multi-GPU system.
std::string uuid_text(const std::array<unsigned char,16>& bytes){
    char value[80];std::snprintf(value,sizeof(value),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        bytes[0],bytes[1],bytes[2],bytes[3],bytes[4],bytes[5],bytes[6],bytes[7],
        bytes[8],bytes[9],bytes[10],bytes[11],bytes[12],bytes[13],bytes[14],bytes[15]);
    return value;
}
std::map<int,GpuDetails> query(const std::string& source,const std::vector<ecmgui::GpuInfo>& devices){
    std::map<int,GpuDetails> result;
    Module nvml{LoadLibraryW(std::filesystem::u8path(source).c_str())};if(!nvml.handle)return result;
    const auto init=nvml.function<int(*)()>("nvmlInit_v2");
    const auto shutdown=nvml.function<int(*)()>("nvmlShutdown");
    if(!init||!shutdown||init()!=0)return result;
    NvmlSession session{shutdown};
    const auto driver=nvml.function<int(*)(char*,unsigned)>("nvmlSystemGetDriverVersion");
    auto cuda_version=nvml.function<int(*)(int*)>("nvmlSystemGetCudaDriverVersion_v2");
    if(!cuda_version)cuda_version=nvml.function<int(*)(int*)>("nvmlSystemGetCudaDriverVersion");
    const auto handle=nvml.function<int(*)(unsigned,void**)>("nvmlDeviceGetHandleByIndex_v2");
    const auto capability=nvml.function<int(*)(void*,int*,int*)>("nvmlDeviceGetCudaComputeCapability");
    const auto uuid=nvml.function<int(*)(void*,char*,unsigned)>("nvmlDeviceGetUUID");
    char driver_text[96]{};int version=0;
    if(!driver||driver(driver_text,sizeof(driver_text))!=0)driver_text[0]=0;
    if(!cuda_version||cuda_version(&version)!=0)version=0;
    std::map<std::string,int> ids;
    for(const auto& device:devices){
        auto& info=result[device.index];info.driver=driver_text;info.cuda_version=version;
        void* dev=nullptr;if(!handle||handle(device.index,&dev)!=0)continue;
        int major=-1,minor=-1;
        if(capability&&capability(dev,&major,&minor)==0){info.major=major;info.minor=minor;}
        char id[96]{};if(uuid&&uuid(dev,id,sizeof(id))==0)ids.emplace(id,device.index);
    }
    if(ids.empty())return result;
    Module cuda{LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32)};if(!cuda.handle)return result;
    const auto cu_init=cuda.function<int(WINAPI*)(unsigned)>("cuInit");
    const auto count=cuda.function<int(WINAPI*)(int*)>("cuDeviceGetCount");
    const auto get=cuda.function<int(WINAPI*)(int*,int)>("cuDeviceGet");
    const auto cu_uuid=cuda.function<int(WINAPI*)(void*,int)>("cuDeviceGetUuid");
    const auto attribute=cuda.function<int(WINAPI*)(int*,int,int)>("cuDeviceGetAttribute");
    int n=0;if(!cu_init||!count||!get||!cu_uuid||!attribute||cu_init(0)!=0||count(&n)!=0)return result;
    for(int ordinal=0;ordinal<n;++ordinal){
        int dev=0;std::array<unsigned char,16> bytes{};
        if(get(&dev,ordinal)!=0||cu_uuid(bytes.data(),dev)!=0)continue;
        const auto found=ids.find(uuid_text(bytes));if(found==ids.end())continue;
        int sms=0; // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16 (CUDA Driver API).
        if(attribute(&sms,16,dev)==0&&sms>0)result[found->second].sm_count=sms;
    }
    return result;
}
}
void GpuDetailsCache::start(const std::string& source,const std::vector<ecmgui::GpuInfo>& devices){
    if(thread_.joinable())return;
    thread_=std::thread([this,source,devices]{
        try{details_=query(source,devices);}catch(...){/* Optional metadata stays unknown. */}
        ready_.store(true,std::memory_order_release);
    });
}
}
