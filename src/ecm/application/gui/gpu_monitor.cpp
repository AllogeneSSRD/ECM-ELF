#include "gpu_monitor.h"
#include "worker_proc.h"      // mono_ms()

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ecmgui {
namespace {

// ---------------------------------------------------------------------------
// Minimal NVML ABI subset (see the note in gpu_monitor.h). Only what the panel
// shows is declared; everything is called through function pointers.
// ---------------------------------------------------------------------------
struct nvmlDevice_st;
typedef nvmlDevice_st *nvmlDevice_t;

struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};

struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

typedef int nvmlReturn_t;
const nvmlReturn_t NVML_SUCCESS = 0;

// nvmlClockType_t
const unsigned int NVML_CLOCK_SM = 1;
const unsigned int NVML_CLOCK_MEM = 2;
// nvmlTemperatureSensors_t
const unsigned int NVML_TEMPERATURE_GPU = 0;

typedef nvmlReturn_t (*fn_init_t)(void);
typedef nvmlReturn_t (*fn_shutdown_t)(void);
typedef nvmlReturn_t (*fn_count_t)(unsigned int *);
typedef nvmlReturn_t (*fn_handle_t)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_name_t)(nvmlDevice_t, char *, unsigned int);
typedef nvmlReturn_t (*fn_util_t)(nvmlDevice_t, nvmlUtilization_t *);
typedef nvmlReturn_t (*fn_clock_t)(nvmlDevice_t, unsigned int, unsigned int *);
typedef nvmlReturn_t (*fn_power_t)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*fn_temp_t)(nvmlDevice_t, unsigned int, unsigned int *);
typedef nvmlReturn_t (*fn_mem_t)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*fn_throttle_t)(nvmlDevice_t, unsigned long long *);
typedef nvmlReturn_t (*fn_cores_t)(nvmlDevice_t, unsigned int *);

bool resolve_fn(void *lib, const char *name, void **slot, bool required, std::string &err) {
#ifdef _WIN32
    *slot = reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    *slot = nullptr;
#endif
    if (*slot == nullptr && required) {
        err = std::string("NVML entry point missing: ") + name;
        return false;
    }
    return true;
}

std::string trim_copy(const std::string &s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

} // namespace

GpuMonitor::GpuMonitor() {}

GpuMonitor::~GpuMonitor() {
    stop();
#ifdef _WIN32
    if (lib_ != nullptr && fn_shutdown_ != nullptr) {
        reinterpret_cast<fn_shutdown_t>(fn_shutdown_)();
        FreeLibrary(static_cast<HMODULE>(lib_));
    }
#endif
}

bool GpuMonitor::load_nvml(std::string &err) {
#ifdef _WIN32
    // ECM_GUI_NVML lets a test point at a wrong/absent DLL to exercise the
    // degradation path without touching the system one.
    std::string candidate;
    if (const char *env = std::getenv("ECM_GUI_NVML")) {
        candidate = env;
        lib_ = LoadLibraryA(candidate.c_str());
    } else {
        lib_ = LoadLibraryA("nvml.dll");          // System32 (driver 411+)
        if (lib_ != nullptr) {
            candidate = "nvml.dll";
        } else {
            candidate = "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll";
            lib_ = LoadLibraryA(candidate.c_str());
        }
    }
    if (lib_ == nullptr) {
        err = "nvml.dll not found (no NVIDIA driver?)";
        return false;
    }
    nvml_source_ = candidate;

    if (!resolve_fn(lib_, "nvmlInit_v2", &fn_init_, true, err)) return false;
    resolve_fn(lib_, "nvmlShutdown", &fn_shutdown_, false, err);
    resolve_fn(lib_, "nvmlDeviceGetCount_v2", &fn_count_, true, err);
    resolve_fn(lib_, "nvmlDeviceGetHandleByIndex_v2", &fn_handle_, true, err);
    // Either spelling is present depending on the driver version.
    if (!resolve_fn(lib_, "nvmlDeviceGetName", &fn_name_, false, err)) return false;
    if (fn_name_ == nullptr && !resolve_fn(lib_, "nvmlDeviceGetName_v2", &fn_name_, true, err)) {
        return false;
    }
    resolve_fn(lib_, "nvmlDeviceGetUtilizationRates", &fn_util_, true, err);
    resolve_fn(lib_, "nvmlDeviceGetClockInfo", &fn_clock_, true, err);
    resolve_fn(lib_, "nvmlDeviceGetPowerUsage", &fn_power_, true, err);
    resolve_fn(lib_, "nvmlDeviceGetEnforcedPowerLimit", &fn_power_limit_, false, err);
    resolve_fn(lib_, "nvmlDeviceGetTemperature", &fn_temp_, true, err);
    resolve_fn(lib_, "nvmlDeviceGetMemoryInfo", &fn_mem_, true, err);
    resolve_fn(lib_, "nvmlDeviceGetThrottleReasons", &fn_throttle_, false, err);
    resolve_fn(lib_, "nvmlDeviceGetNumGpuCores", &fn_cores_, false, err);

    const nvmlReturn_t rc = reinterpret_cast<fn_init_t>(fn_init_)();
    if (rc != NVML_SUCCESS) {
        err = "nvmlInit_v2 failed with code " + std::to_string(rc);
        return false;
    }
    unsigned int count = 0;
    if (reinterpret_cast<fn_count_t>(fn_count_)(&count) != NVML_SUCCESS || count == 0) {
        err = "NVML reports no devices";
        return false;
    }
    devices_.clear();
    history_.assign(count, {});
    for (unsigned int i = 0; i < count; ++i) {
        nvmlDevice_t dev = nullptr;
        if (reinterpret_cast<fn_handle_t>(fn_handle_)(i, &dev) != NVML_SUCCESS) continue;
        GpuInfo info;
        info.index = static_cast<int>(i);
        char name[96] = {0};
        if (reinterpret_cast<fn_name_t>(fn_name_)(dev, name, sizeof(name)) == NVML_SUCCESS) {
            info.name = name;
        }
        nvmlMemory_t mem{};
        if (reinterpret_cast<fn_mem_t>(fn_mem_)(dev, &mem) == NVML_SUCCESS) {
            info.mem_total_mb = static_cast<unsigned int>(mem.total / (1024ull * 1024ull));
        }
        if (fn_cores_ != nullptr) {
            unsigned int cores = 0;
            if (reinterpret_cast<fn_cores_t>(fn_cores_)(dev, &cores) == NVML_SUCCESS) {
                info.cores = static_cast<int>(cores);
            }
        }
        devices_.push_back(info);
    }
    if (devices_.empty()) {
        err = "NVML listed devices but none could be opened";
        return false;
    }
    return true;
#else
    err = "NVML monitoring is Windows-only for now";
    return false;
#endif
}

bool GpuMonitor::read_device(int index, GpuSample &out, std::string &err) {
#ifdef _WIN32
    out = GpuSample();
    if (index < 0 || index >= static_cast<int>(devices_.size())) {
        err = "device index out of range";
        return false;
    }
    nvmlDevice_t dev = nullptr;
    if (reinterpret_cast<fn_handle_t>(fn_handle_)(static_cast<unsigned int>(index), &dev) !=
        NVML_SUCCESS) {
        err = "nvmlDeviceGetHandleByIndex_v2 failed";
        return false;
    }
    nvmlUtilization_t util{};
    if (reinterpret_cast<fn_util_t>(fn_util_)(dev, &util) == NVML_SUCCESS) {
        out.util_gpu = static_cast<int>(util.gpu);
        out.util_mem = static_cast<int>(util.memory);
    }
    unsigned int clk = 0;
    if (reinterpret_cast<fn_clock_t>(fn_clock_)(dev, NVML_CLOCK_SM, &clk) == NVML_SUCCESS) {
        out.clock_sm_mhz = clk;
    }
    clk = 0;
    if (reinterpret_cast<fn_clock_t>(fn_clock_)(dev, NVML_CLOCK_MEM, &clk) == NVML_SUCCESS) {
        out.clock_mem_mhz = clk;
    }
    unsigned int mw = 0;
    if (reinterpret_cast<fn_power_t>(fn_power_)(dev, &mw) == NVML_SUCCESS) {
        out.power_w = mw / 1000.0;
    }
    if (fn_power_limit_ != nullptr) {
        mw = 0;
        if (reinterpret_cast<fn_power_t>(fn_power_limit_)(dev, &mw) == NVML_SUCCESS) {
            out.power_limit_w = mw / 1000.0;
        }
    }
    unsigned int temp = 0;
    if (reinterpret_cast<fn_temp_t>(fn_temp_)(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
        out.temp_c = static_cast<int>(temp);
    }
    nvmlMemory_t mem{};
    if (reinterpret_cast<fn_mem_t>(fn_mem_)(dev, &mem) == NVML_SUCCESS) {
        out.mem_used_mb = static_cast<unsigned int>(mem.used / (1024ull * 1024ull));
    }
    if (fn_throttle_ != nullptr) {
        unsigned long long reasons = 0;
        if (reinterpret_cast<fn_throttle_t>(fn_throttle_)(dev, &reasons) == NVML_SUCCESS) {
            out.throttle_reasons = reasons;
        }
    }
    out.ts_ms = mono_ms();
    out.valid = true;
    err.clear();
    return true;
#else
    (void)index;
    (void)out;
    err = "NVML monitoring is Windows-only for now";
    return false;
#endif
}

bool GpuMonitor::start(int poll_ms) {
    std::string err;
    if (!load_nvml(err)) {
        available_ = false;
        reason_ = err;
        return false;
    }
    available_ = true;
    reason_.clear();
    if (poll_ms <= 0) {
        GpuSample s;
        std::string e;
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (read_device(static_cast<int>(i), s, e)) {
                std::lock_guard<std::mutex> lk(mu_);
                history_[i].push_back(s);
            }
        }
        return true;
    }
    stop_.store(false);
    thread_ = std::thread(&GpuMonitor::thread_main, this, poll_ms);
    return true;
}

void GpuMonitor::thread_main(int poll_ms) {
    while (!stop_.load()) {
        std::vector<GpuSample> batch;
        std::vector<int> batch_index;
        // devices_ is immutable while this thread runs. Never hold mu_ across a
        // driver call: the UI also needs it to read devices/history, and a slow
        // NVML query would otherwise freeze the entire window.
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            GpuSample s;
            std::string err;
            if (read_device(static_cast<int>(i), s, err)) {
                batch.push_back(s);
                batch_index.push_back(static_cast<int>(i));
            }
        }
        if (!batch.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            for (std::size_t k = 0; k < batch.size(); ++k) {
                auto &hist = history_[static_cast<std::size_t>(batch_index[k])];
                hist.push_back(batch[k]);
                if (hist.size() > max_history_) {
                    hist.erase(hist.begin(), hist.begin() +
                                static_cast<std::ptrdiff_t>(hist.size() - max_history_));
                }
            }
        }
        for (int slept = 0; slept < poll_ms && !stop_.load(); slept += 50) {
#ifdef _WIN32
            Sleep(50);
#endif
        }
    }
}

void GpuMonitor::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

std::vector<GpuInfo> GpuMonitor::devices() const {
    std::lock_guard<std::mutex> lk(mu_);
    return devices_;
}

bool GpuMonitor::latest(int device, GpuSample &out) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (device < 0 || device >= static_cast<int>(history_.size())) return false;
    if (history_[device].empty()) return false;
    out = history_[device].back();
    return true;
}

std::vector<GpuSample> GpuMonitor::history(int device) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (device < 0 || device >= static_cast<int>(history_.size())) return {};
    return history_[device];
}

bool GpuMonitor::sample_now(int device, GpuSample &out, std::string &err) {
    if (!available_ && !load_nvml(err)) return false;
    available_ = true;
    return read_device(device, out, err);
}

std::string GpuMonitor::throttle_text(unsigned long long reasons) {
    // nvmlClocksThrottleReasons bits.
    struct Bit {
        unsigned long long mask;
        const char *text;
    };
    static const Bit bits[] = {
        {0x1ull, "idle"},
        {0x2ull, "app clocks"},
        {0x4ull, "power cap"},
        {0x8ull, "hw slowdown"},
        {0x10ull, "sync boost"},
        {0x20ull, "sw thermal"},
        {0x40ull, "hw thermal"},
        {0x80ull, "hw power brake"},
        {0x100ull, "display clock"},
    };
    std::string out;
    for (const Bit &b : bits) {
        if ((reasons & b.mask) == 0) continue;
        // "idle" is the normal state, not a problem: hide it when it is the only bit.
        if (b.mask == 0x1ull && reasons == 0x1ull) return std::string();
        if (!out.empty()) out += ", ";
        out += b.text;
    }
    return out;
}

} // namespace ecmgui
