#pragma once

// Headless GPU-monitor self-test (milestone M4), called from main_win32.cpp for
// `ecm_gui.exe --gpu-selftest`. See gpu_selftest.cpp for what is checked.
//
// Returns 0 = all checks passed, 1 = a check failed, 2 = NVML unavailable.

#include <string>

namespace ecmgui {

int run_gpu_selftest(int expect_devices, bool skip_nvidia_smi, const std::string &log_path);

} // namespace ecmgui
