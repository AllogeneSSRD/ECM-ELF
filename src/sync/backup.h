#pragma once
#include <string>

namespace ecm_sync {
void backup_saves(const std::string& dir, const std::string& first, const std::string& second,
                  bool full, long long since_epoch_seconds);
} // namespace ecm_sync
