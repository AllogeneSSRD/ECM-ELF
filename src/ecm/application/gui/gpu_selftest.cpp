// Headless GPU-monitor self-test (milestone M4). Run with
//
//   ecm_gui.exe --gpu-selftest [--expect-devices N] [--skip-nvidia-smi]
//
// It loads NVML exactly like the GUI does, enumerates the devices, takes samples
// through both the one-shot path and the sampling thread, sanity-checks every field,
// and -- the important part -- compares the numbers against `nvidia-smi` (a totally
// independent consumer of the same driver), which is what makes the hand-declared NVML
// ABI subset in gpu_monitor.cpp trustworthy: a wrong struct layout would show up as
// garbage rather than as a plausible-looking number.
//
// Exit code 0 = every check passed, 1 = a check failed, 2 = NVML unavailable
// (reported as a failure too, but with a distinct code so a caller can tell
// "no NVIDIA driver here" from "the monitor is broken").

#include "gpu_selftest.h"

#include "gpu_monitor.h"
#include "platform.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ecmgui {
namespace {

int g_pass = 0;
int g_fail = 0;
FILE *g_log = nullptr;

void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[1024];
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fputs(buf, stdout);
    std::fflush(stdout);
    if (g_log) {
        std::fputs(buf, g_log);
        std::fflush(g_log);
    }
}

void check(bool ok, const std::string &what, const std::string &detail = std::string()) {
    if (ok) {
        ++g_pass;
        say("  [ok]   %s\n", what.c_str());
    } else {
        ++g_fail;
        say("  [FAIL] %s%s%s\n", what.c_str(), detail.empty() ? "" : " -- ", detail.c_str());
    }
}

// Reads one column of `nvidia-smi -i <idx> --query-gpu=... --format=csv,noheader,nounits`.
// `index` < 0 means "all devices". Returns false when nvidia-smi is missing or the
// output cannot be parsed.
bool nvidia_smi_query(int index, const char *fields,
                      std::vector<std::vector<std::string>> &rows) {
    rows.clear();
    std::string cmd = "nvidia-smi ";
    if (index >= 0) cmd += "-i " + std::to_string(index) + " ";
    cmd += std::string("--query-gpu=") + fields + " --format=csv,noheader,nounits";
    FILE *p = _popen(cmd.c_str(), "r");
    if (p == nullptr) return false;
    char line[512];
    while (std::fgets(line, sizeof(line), p) != nullptr) {
        std::vector<std::string> cols;
        std::string cur;
        for (const char *c = line; *c; ++c) {
            if (*c == ',') {
                cols.push_back(cur);
                cur.clear();
            } else if (*c != '\n' && *c != '\r') {
                cur.push_back(*c);
            }
        }
        while (!cur.empty() && (cur.back() == ' ' || cur.back() == '\t')) cur.pop_back();
        while (!cur.empty() && (cur.front() == ' ' || cur.front() == '\t')) cur.erase(cur.begin());
        cols.push_back(cur);
        if (!cols.empty()) rows.push_back(cols);
    }
    const int rc = _pclose(p);
    return rc == 0 && !rows.empty();
}

double to_double(const std::string &s, double def = -1.0) {
    if (s.empty()) return def;
    char *end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    return (end != nullptr && *end == '\0') ? v : def;
}

// NB: do not call this `near` -- windef.h defines near/far as legacy macros.
bool within(double a, double b, double tol) {
    const double d = a - b;
    return (d < 0 ? -d : d) <= tol;
}

// One cross-check of one device: sample NVML, then read nvidia-smi three times for
// that device and compare. Returns false with a human-readable `detail` when the two
// disagree in a way that a correct ABI subset cannot produce.
bool cross_check_device(GpuMonitor &mon, int index, std::string *detail) {
    GpuSample s;
    std::string err;
    if (!mon.sample_now(index, s, err)) {
        if (detail) *detail = "sample failed: " + err;
        return false;
    }
    double u_min = 1e9, u_max = -1e9, p_min = 1e9, p_max = -1e9;
    double m_min = 1e9, m_max = -1e9, t_min = 1e9, t_max = -1e9;
    double sm_min = 1e9, sm_max = -1e9;
    int got = 0;
    for (int k = 0; k < 3; ++k) {
        std::vector<std::vector<std::string>> rows;
        if (nvidia_smi_query(index,
                             "utilization.gpu,clocks.current.sm,clocks.current.memory,"
                             "power.draw,temperature.gpu", rows) &&
            !rows.empty() && rows[0].size() >= 5) {
            const double u = to_double(rows[0][0]);
            const double sm = to_double(rows[0][1]);
            const double mm = to_double(rows[0][2]);
            const double pw = to_double(rows[0][3]);
            const double tp = to_double(rows[0][4]);
            if (u >= 0) { u_min = (u < u_min ? u : u_min); u_max = (u > u_max ? u : u_max); }
            if (sm > 0) { sm_min = (sm < sm_min ? sm : sm_min); sm_max = (sm > sm_max ? sm : sm_max); }
            if (mm > 0) { m_min = (mm < m_min ? mm : m_min); m_max = (mm > m_max ? mm : m_max); }
            if (pw >= 0) { p_min = (pw < p_min ? pw : p_min); p_max = (pw > p_max ? pw : p_max); }
            if (tp > 0) { t_min = (tp < t_min ? tp : t_min); t_max = (tp > t_max ? tp : t_max); }
            ++got;
        }
    }
    if (got == 0) {
        if (detail) *detail = "nvidia-smi gave no readable row";
        return false;
    }

    // Tolerances are chosen for what this cross-check is FOR: proving the hand-declared
    // NVML structs/enums read the right QUANTITY. A wrong field is off by orders of
    // magnitude (a core count showing up as a clock, a bogus 590 W on a 55 W card),
    // while on a busy machine -- and the GUI itself renders through D3D11 -- the two
    // tools legitimately disagree at the percentage level.
    const bool ok_util = (u_max < 0) ||
                         (s.util_gpu >= u_min - 40.0 && s.util_gpu <= u_max + 40.0);
    const bool ok_mem = (m_max < 0) ||
                        (s.clock_mem_mhz >= m_min - 100.0 && s.clock_mem_mhz <= m_max + 100.0);
    const bool ok_temp = (t_max < 0) || (s.temp_c >= t_min - 8.0 && s.temp_c <= t_max + 8.0);
    // SM clock: NOT compared point-by-point. A modern GPU ramps between ~200 MHz and its
    // boost clock (measured 210 <-> 2595 MHz on the 4070 Ti), a 12x swing, so two reads
    // taken even 100 ms apart can legitimately differ by an order of magnitude. The value
    // must instead land in the plausible SM-clock range -- which still catches a swapped
    // clock-type constant, since these cards report 7001..10501 MHz for memory.
    const bool ok_sm = (s.clock_sm_mhz >= 100 && s.clock_sm_mhz <= 4000);
    // Power: inside the range nvidia-smi reported around the same instant, with slack for
    // the fleeting bogus value. A reading above 3x the enforced power limit is physically
    // impossible and is what both tools show intermittently on the RTX 4060 Laptop GPU
    // (measured ~9.6 W and 590.01 W on a 55 W card), so such a sample is not compared.
    const bool ours_bogus = (s.power_limit_w > 0.0 && s.power_w > 3.0 * s.power_limit_w);
    const bool smi_bogus = (s.power_limit_w > 0.0 && p_max > 3.0 * s.power_limit_w);
    const bool ok_pow = ours_bogus || smi_bogus || (p_max < 0) ||
                        (s.power_w >= p_min * 0.4 - 20.0 && s.power_w <= p_max * 2.0 + 20.0);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "ours %d%%/%.1fW/sm %u/mem %u/%dC   smi util[%.0f,%.0f] power[%.1f,%.1f] "
                  "sm[%.0f,%.0f] mem[%.0f,%.0f] temp[%.0f,%.0f]",
                  s.util_gpu, s.power_w, s.clock_sm_mhz, s.clock_mem_mhz, s.temp_c, u_min, u_max,
                  p_min, p_max, sm_min, sm_max, m_min, m_max, t_min, t_max);
    if (detail) *detail = buf;
    say("      gpu %d  ours %d%% %.1fW sm %u mem %u %dC   smi util[%.0f,%.0f] power[%.1f,%.1f] "
        "sm[%.0f,%.0f] mem[%.0f,%.0f] temp[%.0f,%.0f]\n",
        index, s.util_gpu, s.power_w, s.clock_sm_mhz, s.clock_mem_mhz, s.temp_c, u_min, u_max,
        p_min, p_max, sm_min, sm_max, m_min, m_max, t_min, t_max);
    return ok_util && ok_mem && ok_temp && ok_sm && ok_pow;
}

} // namespace

int run_gpu_selftest(int expect_devices, bool skip_nvidia_smi, const std::string &log_path) {
    fopen_s(&g_log, log_path.c_str(), "w");
    say("ecm_gui GPU monitor self-test\n");

    GpuMonitor mon;
    const bool ok = mon.start(0);          // 0 = load + one synchronous sample, no thread
    if (!ok) {
        say("  [FAIL] NVML unavailable: %s\n", mon.reason().c_str());
        if (g_log) { std::fclose(g_log); g_log = nullptr; }
        return 2;
    }
    say("  nvml: %s\n", mon.nvml_source().c_str());
    check(true, "NVML loaded and initialised");

    const std::vector<GpuInfo> devices = mon.devices();
    check(!devices.empty(), "at least one GPU", std::to_string(devices.size()) + " device(s)");
    if (expect_devices > 0) {
        check(static_cast<int>(devices.size()) == expect_devices,
              "device count as expected (" + std::to_string(expect_devices) + ")",
              std::to_string(devices.size()));
    }

    std::vector<GpuSample> samples(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const GpuInfo &info = devices[i];
        say("  gpu %d: %s (%u MB, %d CUDA cores)\n", info.index, info.name.c_str(),
            info.mem_total_mb, info.cores);
        check(!info.name.empty(), "device has a name", info.name);
        check(info.mem_total_mb > 0, "device reports its memory size",
              std::to_string(info.mem_total_mb) + " MB");

        GpuSample s;
        std::string err;
        const bool got = mon.sample_now(static_cast<int>(i), s, err);
        check(got, "synchronous sample", err);
        if (!got) continue;
        samples[i] = s;
        check(s.util_gpu >= 0 && s.util_gpu <= 100, "utilisation in 0..100",
              std::to_string(s.util_gpu));
        check(s.clock_sm_mhz > 0, "SM clock > 0", std::to_string(s.clock_sm_mhz) + " MHz");
        check(s.power_w > 0.0 && s.power_w < 2000.0, "power draw plausible",
              std::to_string(s.power_w) + " W");
        check(s.temp_c > 0 && s.temp_c < 120, "temperature plausible",
              std::to_string(s.temp_c) + " C");
        check(s.mem_used_mb <= info.mem_total_mb, "used memory <= total",
              std::to_string(s.mem_used_mb) + " <= " + std::to_string(info.mem_total_mb));
        say("      util %d%% (mem %d%%), SM %u MHz, mem %u MHz, %.1f W (limit %.0f W), "
            "%d C, %u MB used, throttle: %s\n",
            s.util_gpu, s.util_mem, s.clock_sm_mhz, s.clock_mem_mhz, s.power_w,
            s.power_limit_w, s.temp_c, s.mem_used_mb,
            GpuMonitor::throttle_text(s.throttle_reasons).empty()
                ? "none"
                : GpuMonitor::throttle_text(s.throttle_reasons).c_str());
    }

    // Sampling thread: two samples a short interval apart, in increasing order.
    say("[2] the sampling thread collects a history\n");
    {
        GpuMonitor t;
        check(t.start(200), "start(200 ms) started the sampling thread", t.reason());
#ifdef _WIN32
        Sleep(900);
#endif
        for (std::size_t i = 0; i < devices.size(); ++i) {
            const std::vector<GpuSample> hist = t.history(static_cast<int>(i));
            check(hist.size() >= 2, "device " + std::to_string(i) + " has >= 2 samples",
                  std::to_string(hist.size()) + " samples");
            bool ordered = true;
            for (std::size_t k = 1; k < hist.size(); ++k) {
                if (hist[k].ts_ms < hist[k - 1].ts_ms) ordered = false;
            }
            check(ordered, "samples are time ordered");
        }
        t.stop();
    }

    // Cross-check against nvidia-smi (independent consumer of the same driver).
    //
    // Interleaved per device and using the RANGE of three quick reads, for two
    // measured reasons:
    //   * sampling all devices first and calling nvidia-smi afterwards puts 1-2 s
    //     between the pairs, and clocks/power move faster than that (measured: our
    //     210 MHz vs nvidia-smi's 2595 MHz for the same idle card);
    //   * a card can report a bogus value intermittently: the RTX 4060 Laptop GPU
    //     flips its power draw between ~9.6 W and 590.01 W in BOTH tools (a driver/
    //     power-gating quirk), so a single-point comparison is not meaningful.
    if (skip_nvidia_smi) {
        say("[3] nvidia-smi cross-check skipped\n");
    } else {
        say("[3] cross-check against nvidia-smi (interleaved, 3 samples per device)\n");
        for (std::size_t i = 0; i < devices.size(); ++i) {
            // The comparison is between two live tools on a machine that may be busy
            // (the GUI renders, other tests may run GPU work), so a single disagreeing
            // sample is not evidence: take a fresh pair and only report a failure when
            // BOTH attempts disagree. A wrong struct/enum disagrees every time.
            bool ok = false;
            std::string detail;
            for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
                ok = cross_check_device(mon, static_cast<int>(i), &detail);
            }
            check(ok, "gpu " + std::to_string(i) + " matches nvidia-smi", detail);
        }
    }

    // Degradation: a bogus DLL path must be reported, not crash.
    say("[4] degradation path (ECM_GUI_NVML points nowhere)\n");
    {
#ifdef _WIN32
        _putenv_s("ECM_GUI_NVML", "Z:\\definitely\\not\\nvml.dll");
#endif
        GpuMonitor bad;
        const bool started = bad.start(0);
        check(!started, "bogus NVML path is refused");
        check(!bad.reason().empty(), "a reason is reported", bad.reason());
#ifdef _WIN32
        _putenv_s("ECM_GUI_NVML", "");
#endif
    }

    say("\npassed: %d   failed: %d\n  log: %s\n", g_pass, g_fail, log_path.c_str());
    if (g_log) {
        std::fclose(g_log);
        g_log = nullptr;
    }
    return g_fail == 0 ? 0 : 1;
}

} // namespace ecmgui
