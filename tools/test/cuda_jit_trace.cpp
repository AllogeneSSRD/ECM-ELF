#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
namespace {
CUpti_SubscriberHandle subscriber{};
std::mutex guard;
std::string directory;
unsigned long long module_sequence=0;
LARGE_INTEGER frequency{};
void CUPTIAPI request_buffer(uint8_t **buffer,size_t *size,size_t *records) {
    *size=1024*1024;*records=0;*buffer=static_cast<uint8_t*>(std::malloc(*size));
}
void CUPTIAPI complete_buffer(CUcontext context,uint32_t stream,uint8_t *buffer,size_t,size_t valid) {
    CUpti_Activity *record=nullptr;
    while(cuptiActivityGetNextRecord(buffer,valid,&record)==CUPTI_SUCCESS) {
        if(record->kind!=CUPTI_ACTIVITY_KIND_JIT)continue;
        const auto *jit=reinterpret_cast<const CUpti_ActivityJit2*>(record);
        std::lock_guard<std::mutex> lock(guard);
        if(FILE *file=std::fopen((directory+"/jit.csv").c_str(),"a")) {
            std::fprintf(file,"%d,%llu,%llu,%u,%llu,%llu,\"%s\",%lu\n",int(jit->jitOperationType),
                static_cast<unsigned long long>(jit->start),static_cast<unsigned long long>(jit->end),jit->deviceId,
                static_cast<unsigned long long>(jit->jitOperationCorrelationId),static_cast<unsigned long long>(jit->cacheSize),jit->cachePath?jit->cachePath:"",GetCurrentProcessId());
            std::fclose(file);
        }
    }
    size_t dropped=0;cuptiActivityGetNumDroppedRecords(context,stream,&dropped);
    if(dropped) {
        std::lock_guard<std::mutex> lock(guard);
        if(FILE *file=std::fopen((directory+"/dropped.txt").c_str(),"a")){std::fprintf(file,"%zu\n",dropped);std::fclose(file);}
    }
    std::free(buffer);
}
void CUPTIAPI callback(void *,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void *data) {
    LARGE_INTEGER now;QueryPerformanceCounter(&now);
    if(domain==CUPTI_CB_DOMAIN_RESOURCE && id==CUPTI_CBID_RESOURCE_MODULE_LOADED) {
        const auto *resource=static_cast<const CUpti_ResourceData*>(data);
        const auto *module=static_cast<const CUpti_ModuleResourceData*>(resource->resourceDescriptor);
        if(!module || !module->pCubin || !module->cubinSize)return;
        std::lock_guard<std::mutex> lock(guard);
        const auto filename="module_"+std::to_string(GetCurrentProcessId())+"_"+std::to_string(module->moduleId)+"_"+std::to_string(++module_sequence)+".cubin";
        const auto path=directory+"/"+filename;
        if(FILE *file=std::fopen(path.c_str(),"wb")){std::fwrite(module->pCubin,1,module->cubinSize,file);std::fclose(file);}
        if(FILE *file=std::fopen((directory+"/modules.csv").c_str(),"a")){
            std::fprintf(file,"%u,%zu,%.9f,%lu,%s\n",module->moduleId,module->cubinSize,double(now.QuadPart)/frequency.QuadPart,GetCurrentProcessId(),filename.c_str());std::fclose(file);
        }
    } else if(domain==CUPTI_CB_DOMAIN_DRIVER_API) {
        const auto *info=static_cast<const CUpti_CallbackData*>(data);
        if(info->callbackSite==CUPTI_API_ENTER){*info->correlationData=now.QuadPart;return;}
        std::lock_guard<std::mutex> lock(guard);
        if(FILE *file=std::fopen((directory+"/driver.csv").c_str(),"a")){
            std::fprintf(file,"%s,%.6f,%u,%lu\n",info->functionName,
                1000.*double(now.QuadPart-*info->correlationData)/frequency.QuadPart,info->correlationId,GetCurrentProcessId());std::fclose(file);
        }
    }
    if(domain==CUPTI_CB_DOMAIN_DRIVER_API && id==CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel &&
       static_cast<const CUpti_CallbackData*>(data)->callbackSite==CUPTI_API_EXIT)
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}
}
extern "C" __declspec(dllexport) int InitializeInjection() {
    const char *path=std::getenv("ECM_JIT_TRACE_DIR");if(!path || !*path)return 0;
    directory=path;QueryPerformanceFrequency(&frequency);
    const auto result=cuptiSubscribe(&subscriber,callback,nullptr);
    const auto status_path=directory+"/injection_"+std::to_string(GetCurrentProcessId())+".txt";
    if(FILE *file=std::fopen(status_path.c_str(),"w")){
        std::fprintf(file,"cuptiSubscribe=%d\n",int(result));std::fclose(file);
    }
    if(FILE *file=std::fopen((directory+"/injection.txt").c_str(),"w")){
        std::fprintf(file,"cuptiSubscribe=%d\n",int(result));std::fclose(file);
    }
    if(result!=CUPTI_SUCCESS)return 0;
    const auto activity=cuptiActivityRegisterCallbacks(request_buffer,complete_buffer);
    const auto jit=activity==CUPTI_SUCCESS?cuptiActivityEnable(CUPTI_ACTIVITY_KIND_JIT):activity;
    if(FILE *file=std::fopen((directory+"/injection.txt").c_str(),"a")){
        std::fprintf(file,"jit_activity=%d\n",int(jit));std::fclose(file);
    }
    if(FILE *file=std::fopen(status_path.c_str(),"a")){
        std::fprintf(file,"jit_activity=%d\n",int(jit));std::fclose(file);
    }
    if(jit!=CUPTI_SUCCESS)return 0;
    // Flush at launch API exit while CUDA is live. Calling CUPTI from DLL
    // process-detach/atexit can deadlock under the Windows loader lock.
    if(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED)!=CUPTI_SUCCESS)return 0;
    // Keep tracing overhead bounded to initialization/loading/first launch APIs.
    for(const auto id:{CUPTI_DRIVER_TRACE_CBID_cuInit,CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,
        CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,
        CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
        CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData,CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile,
        CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel,CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction})
        if(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API,id)!=CUPTI_SUCCESS)return 0;
    return 1;
}
