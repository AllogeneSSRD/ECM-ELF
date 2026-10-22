#pragma once
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <cstdio>
#include <string>

struct ImDrawData;
namespace ecmgui_v2 {
class Dx11Window {
public:
    Dx11Window(HWND window, FILE* log);
    ~Dx11Window();
    Dx11Window(const Dx11Window&)=delete;
    Dx11Window& operator=(const Dx11Window&)=delete;
    void rebuild(unsigned width,unsigned height);
    void new_frame();
    bool render(ImDrawData* draw,const float* clear);
    void shutdown();
    const std::string& device_name() const {return name_;}
    unsigned rebuilds=0;
private:
    template<class T> using ComPtr=Microsoft::WRL::ComPtr<T>;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGISwapChain> swapchain_;
    ComPtr<ID3D11RenderTargetView> target_;
    FILE* log_;
    std::string name_;
    bool initialized_=false,occluded_=false;
    unsigned presents_=0;
    void create_target();
};
}
