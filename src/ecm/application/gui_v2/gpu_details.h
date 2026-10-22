#pragma once
#include "../gui/platform.h"
#include <atomic>
#include <map>
#include <thread>

namespace ecmgui_v2 {
struct GpuDetails {
    std::string driver;
    int cuda_version=0,major=-1,minor=-1,sm_count=0;
};
// Read once off the UI thread. Hovering only reads the published snapshot.
// No CUDA context, allocation, or compute work is created.
class GpuDetailsCache {
public:
    ~GpuDetailsCache(){if(thread_.joinable())thread_.join();}
    void start(const std::string& nvml_source,const std::vector<ecmgui::GpuInfo>& devices);
    bool ready() const{return ready_.load(std::memory_order_acquire);}
    GpuDetails get(int index) const{
        if(!ready())return {};
        const auto found=details_.find(index);return found==details_.end()?GpuDetails{}:found->second;
    }
private:
    std::thread thread_;
    std::atomic<bool> ready_{false};
    std::map<int,GpuDetails> details_;
};
}
