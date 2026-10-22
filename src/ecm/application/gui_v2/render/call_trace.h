#pragma once
#include "vulkan_context.h"
#include <cstdio>

namespace ecmgui_v2 {
// Opt-in call-boundary evidence. No polling thread and no altered Vulkan semantics.
class CallTrace {
    struct Span {
        const char* name; ULONGLONG start=GetTickCount64();
        explicit Span(const char* value):name(value){std::fprintf(owner_->log_,"render_call: begin ticks=%llu op=%s\n",start,name);std::fflush(owner_->log_);}
        ~Span(){std::fprintf(owner_->log_,"render_call: end ticks=%llu op=%s ms=%llu\n",GetTickCount64(),name,GetTickCount64()-start);std::fflush(owner_->log_);}
    };
    FILE* log_;bool enabled_;
    inline static CallTrace* owner_=nullptr;
#define ECM_TRACE_CALL(name,params,args) \
    PFN_##name saved_##name=nullptr; \
    static VKAPI_ATTR VkResult VKAPI_CALL hook_##name params {Span span(#name);return owner_->saved_##name args;}
    ECM_TRACE_CALL(vkAcquireNextImageKHR,(VkDevice d,VkSwapchainKHR s,uint64_t t,VkSemaphore a,VkFence f,uint32_t* i),(d,s,t,a,f,i))
    ECM_TRACE_CALL(vkWaitForFences,(VkDevice d,uint32_t n,const VkFence* f,VkBool32 all,uint64_t t),(d,n,f,all,t))
    ECM_TRACE_CALL(vkQueueSubmit,(VkQueue q,uint32_t n,const VkSubmitInfo* i,VkFence f),(q,n,i,f))
    ECM_TRACE_CALL(vkQueuePresentKHR,(VkQueue q,const VkPresentInfoKHR* i),(q,i))
    ECM_TRACE_CALL(vkQueueWaitIdle,(VkQueue q),(q))
    ECM_TRACE_CALL(vkDeviceWaitIdle,(VkDevice d),(d))
    ECM_TRACE_CALL(vkCreateSwapchainKHR,(VkDevice d,const VkSwapchainCreateInfoKHR* i,const VkAllocationCallbacks* a,VkSwapchainKHR* s),(d,i,a,s))
#undef ECM_TRACE_CALL
public:
    CallTrace(FILE* log,bool enabled):log_(log),enabled_(enabled){
        if(!enabled_)return;owner_=this;
#define ECM_TRACE_HOOK(name) saved_##name=name;name=hook_##name;
        ECM_TRACE_HOOK(vkAcquireNextImageKHR) ECM_TRACE_HOOK(vkWaitForFences) ECM_TRACE_HOOK(vkQueueSubmit)
        ECM_TRACE_HOOK(vkQueuePresentKHR) ECM_TRACE_HOOK(vkQueueWaitIdle) ECM_TRACE_HOOK(vkDeviceWaitIdle) ECM_TRACE_HOOK(vkCreateSwapchainKHR)
#undef ECM_TRACE_HOOK
    }
    ~CallTrace(){
        if(!enabled_)return;
#define ECM_TRACE_RESTORE(name) name=saved_##name;
        ECM_TRACE_RESTORE(vkAcquireNextImageKHR) ECM_TRACE_RESTORE(vkWaitForFences) ECM_TRACE_RESTORE(vkQueueSubmit)
        ECM_TRACE_RESTORE(vkQueuePresentKHR) ECM_TRACE_RESTORE(vkQueueWaitIdle) ECM_TRACE_RESTORE(vkDeviceWaitIdle) ECM_TRACE_RESTORE(vkCreateSwapchainKHR)
#undef ECM_TRACE_RESTORE
        owner_=nullptr;
    }
    CallTrace(const CallTrace&)=delete;CallTrace& operator=(const CallTrace&)=delete;
};
}
