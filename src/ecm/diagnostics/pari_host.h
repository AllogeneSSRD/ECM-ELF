#pragma once
#include <filesystem>

namespace ecm_diagnostics {
int run_pari_request(const std::filesystem::path& request);
}
