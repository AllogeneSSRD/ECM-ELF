#pragma once

// Headless worker-supervisor self-test (milestone M2). Called from main_win32.cpp
// for `ecm_gui.exe --worker-selftest --fake <exe>`.
//
// Returns 0 when every check passed, 1 on a failed check, 2 when the fake worker
// executable is missing. Writes the same report to `log_path`.

#include <string>

namespace ecmgui {

int run_worker_selftest(const std::string &fake_path, const std::string &log_path);

} // namespace ecmgui
