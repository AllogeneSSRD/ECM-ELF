#pragma once

// GPU monitoring through NVML (milestone M4, docs/usage/GUI.md).
//
// Design decisions:
//   * NVML is loaded DYNAMICALLY at runtime (LoadLibrary + GetProcAddress) and is
//     strictly optional: a machine without an NVIDIA driver still runs the GUI, the
//     panel just says why it is unavailable. No import library, no hard dependency.
//   * Sampling runs on its own thread (the UI never calls NVML), period from
//     [GUI] gpu_poll_ms, with a rolling history for the plots.
//   * The values are the documented NVML ones: util in percent, clocks in MHz, power
//     in W (NVML reports mW), temperature in C, memory in MB.
//
// The NVML ABI subset this file needs is declared here instead of shipping NVIDIA's
// nvml.h: the GUI links nothing from the driver, and the declarations below are the
// stable part of the ABI (plain structs of unsigned int/long long, functions by
// name). src/gui/gpu_selftest.cpp cross-checks every field against `nvidia-smi`
// so a wrong layout would fail loudly rather than silently show garbage.

#include "platform.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ecmgui {

class GpuMonitor {
public:
    GpuMonitor();
    ~GpuMonitor();

    GpuMonitor(const GpuMonitor &) = delete;
    GpuMonitor &operator=(const GpuMonitor &) = delete;

    // Loads NVML and starts the sampling thread. Returns false with `reason` when NVML
    // is unavailable (missing DLL, no devices, init error) -- the UI shows `reason()`.
    // `poll_ms` <= 0 samples once and does not start a thread.
    bool start(int poll_ms);
    void stop();

    bool available() const { return available_; }
    const std::string &reason() const { return reason_; }
    const std::string &nvml_source() const { return nvml_source_; }

    std::vector<GpuInfo> devices() const;
    // Newest sample of one device (false when there is none yet).
    bool latest(int device, GpuSample &out) const;
    // Rolling history, oldest first (for ImGui::PlotLines).
    std::vector<GpuSample> history(int device) const;

    // One-shot synchronous sample, also used by --gpu-selftest.
    bool sample_now(int device, GpuSample &out, std::string &err);

    // Human-readable throttle reason ("power cap, thermal" ...), empty when none.
    static std::string throttle_text(unsigned long long reasons);

private:
    void thread_main(int poll_ms);
    bool load_nvml(std::string &err);
    bool read_device(int index, GpuSample &out, std::string &err);

    void *lib_ = nullptr;                 // HMODULE
    std::string nvml_source_;
    bool available_ = false;
    std::string reason_;
    std::vector<GpuInfo> devices_;
    std::vector<std::vector<GpuSample>> history_;   // per device
    mutable std::mutex mu_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::size_t max_history_ = 240;       // 2 minutes at 500 ms

    // NVML entry points (null when NVML is absent).
    void *fn_init_ = nullptr;
    void *fn_shutdown_ = nullptr;
    void *fn_count_ = nullptr;
    void *fn_handle_ = nullptr;
    void *fn_name_ = nullptr;
    void *fn_util_ = nullptr;
    void *fn_clock_ = nullptr;
    void *fn_power_ = nullptr;
    void *fn_power_limit_ = nullptr;
    void *fn_temp_ = nullptr;
    void *fn_mem_ = nullptr;
    void *fn_throttle_ = nullptr;
    void *fn_cores_ = nullptr;            // optional (nvmlDeviceGetNumGpuCores)
};

} // namespace ecmgui
