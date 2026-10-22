#pragma once
#include "dx11_window.h"
#include "vulkan_window.h"
#include "memory_audit.h"
#include "call_trace.h"
#include "imgui_impl_vulkan.h"
#include <memory>

namespace ecmgui_v2 {
// Own only the selected renderer. DX11 never initializes Volk or a Vulkan device.
class RenderHost {
    VulkanContext context_;
    std::unique_ptr<CallTrace> trace_;
    std::unique_ptr<MemoryAudit> memory_;
    std::unique_ptr<VulkanWindow> vk_;
    std::unique_ptr<Dx11Window> dx_;
    std::string name_;
public:
    RenderHost(HWND hwnd,bool vulkan,bool validation,bool no_loader,bool trace,bool audit,FILE* log){
        if(vulkan){
            context_.initialize(hwnd,validation,no_loader,log);
            trace_=std::make_unique<CallTrace>(log,trace);
            memory_=std::make_unique<MemoryAudit>(audit);
            vk_=std::make_unique<VulkanWindow>(context_,log);name_=context_.properties.deviceName;
        }else{
            dx_=std::make_unique<Dx11Window>(hwnd,log);name_=dx_->device_name();
            std::fprintf(log,"memory_audit_scope: explicit_vk_device_memory=0 DX11_memory=unavailable\n");
        }
    }
    const char* device_name() const{return name_.c_str();}
    bool needs_rebuild() const{return vk_&&vk_->needs_rebuild;}
    unsigned rebuilds() const{return vk_?vk_->rebuilds:dx_->rebuilds;}
    void rebuild(unsigned w,unsigned h){if(vk_)vk_->rebuild(w,h);else dx_->rebuild(w,h);}
    void new_frame(){if(vk_)ImGui_ImplVulkan_NewFrame();else dx_->new_frame();}
    bool render(ImDrawData* draw,const float* clear){
        if(dx_)return dx_->render(draw,clear);
        std::copy(clear,clear+4,vk_->clear_color);return vk_->render(draw);
    }
    void shutdown(){if(vk_)vk_->shutdown();if(dx_)dx_->shutdown();}
    unsigned validation_errors() const{return context_.validation_errors.load();}
    VkDeviceSize bytes() const{return memory_?memory_->bytes():0;}
    size_t allocations() const{return memory_?memory_->allocations():0;}
    void describe(FILE* log) const{if(memory_)memory_->describe(log);else std::fprintf(log,"gpu_blocks: unavailable backend=DX11\n");}
};
}
