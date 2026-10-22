#include "dx11_window.h"
#include "imgui_impl_dx11.h"
#include <stdexcept>

namespace ecmgui_v2 {
namespace {
void check_dx(HRESULT result,const char* operation){
    if(FAILED(result)){char message[160];std::snprintf(message,sizeof(message),"%s failed: HRESULT=0x%08lX",operation,static_cast<unsigned long>(result));throw std::runtime_error(message);}
}
}
Dx11Window::Dx11Window(HWND window,FILE* log):log_(log){
    // Match the unchanged original GUI's D3D11 swap-chain family.
    DXGI_SWAP_CHAIN_DESC desc{};
    desc.BufferCount=2;desc.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.OutputWindow=window;
    desc.SampleDesc.Count=1;desc.Windowed=TRUE;desc.SwapEffect=DXGI_SWAP_EFFECT_DISCARD;
    const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_10_0};
    D3D_FEATURE_LEVEL selected{};
    HRESULT result=D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,levels,2,D3D11_SDK_VERSION,&desc,swapchain_.GetAddressOf(),device_.GetAddressOf(),&selected,context_.GetAddressOf());
    bool software=false;
    if(result==DXGI_ERROR_UNSUPPORTED){
        software=true;swapchain_.Reset();context_.Reset();device_.Reset();
        result=D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,levels,2,D3D11_SDK_VERSION,&desc,swapchain_.GetAddressOf(),device_.GetAddressOf(),&selected,context_.GetAddressOf());
    }
    check_dx(result,"create DX11 device/swap chain");
    ComPtr<IDXGIDevice> dxgi;ComPtr<IDXGIAdapter> adapter;DXGI_ADAPTER_DESC info{};
    check_dx(device_.As(&dxgi),"query DXGI device");check_dx(dxgi->GetAdapter(adapter.GetAddressOf()),"query DX11 adapter");check_dx(adapter->GetDesc(&info),"describe DX11 adapter");
    char name[512]{};WideCharToMultiByte(CP_UTF8,0,info.Description,-1,name,sizeof(name),nullptr,nullptr);name_=name;
    create_target();
    if(!ImGui_ImplDX11_Init(device_.Get(),context_.Get()))throw std::runtime_error("ImGui DX11 initialization failed");
    initialized_=true;
    std::fprintf(log_,"d3d11: adapter=%s feature=0x%x software=%d\n",name_.c_str(),static_cast<unsigned>(selected),software);std::fflush(log_);
}
Dx11Window::~Dx11Window(){shutdown();}
void Dx11Window::create_target(){
    ComPtr<ID3D11Texture2D> back;
    check_dx(swapchain_->GetBuffer(0,IID_PPV_ARGS(back.GetAddressOf())),"get DX11 back buffer");
    check_dx(device_->CreateRenderTargetView(back.Get(),nullptr,target_.GetAddressOf()),"create DX11 render target");
}
void Dx11Window::rebuild(unsigned width,unsigned height){
    if(!width||!height)return;
    context_->OMSetRenderTargets(0,nullptr,nullptr);target_.Reset();
    check_dx(swapchain_->ResizeBuffers(0,width,height,DXGI_FORMAT_UNKNOWN,0),"resize DX11 buffers");
    create_target();++rebuilds;
    std::fprintf(log_,"dx11_swapchain: rebuild=%u extent=%ux%u\n",rebuilds,width,height);std::fflush(log_);
}
void Dx11Window::new_frame(){ImGui_ImplDX11_NewFrame();}
bool Dx11Window::render(ImDrawData* draw,const float* clear){
    if(occluded_){
        const HRESULT test=swapchain_->Present(0,DXGI_PRESENT_TEST);
        if(test==DXGI_STATUS_OCCLUDED)return false;
        check_dx(test,"test DX11 visibility");occluded_=false;
    }
    auto* target=target_.Get();context_->OMSetRenderTargets(1,&target,nullptr);
    context_->ClearRenderTargetView(target,clear);ImGui_ImplDX11_RenderDrawData(draw);
    const HRESULT result=swapchain_->Present(1,0);
    if(result==DXGI_STATUS_OCCLUDED){occluded_=true;return false;}
    check_dx(result,"present DX11 frame");++presents_;return true;
}
void Dx11Window::shutdown(){
    if(!initialized_)return;
    ImGui_ImplDX11_Shutdown();initialized_=false;
    context_->OMSetRenderTargets(0,nullptr,nullptr);context_->ClearState();context_->Flush();
    target_.Reset();swapchain_.Reset();context_.Reset();device_.Reset();
    std::fprintf(log_,"renderer_summary: backend=DX11 presents=%u shutdown=complete api_validation=unavailable\n",presents_);std::fflush(log_);
}
}
