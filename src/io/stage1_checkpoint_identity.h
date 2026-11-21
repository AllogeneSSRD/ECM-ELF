#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ecm_stage1_checkpoint {
// Read-only discovery of explicitly approved legacy identities. Envelope and
// record checksums/counts are checked here; Context still validates the payload.
std::vector<std::string> find_identity_aliases(
    const std::string& path, uint32_t curves, unsigned backups,
    const std::function<bool(std::string_view)>& compatible);
}
