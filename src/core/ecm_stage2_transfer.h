#pragma once

#include <cstddef>
#include <string>

struct Stage2TransferConfig {
    std::string worktodo;
    std::string finished;
    std::string pending;
    int worker = 1;
};

// Park the already-published Edwards save assignment before advancing Stage1.
bool ecm_stage2_transfer_park(const Stage2TransferConfig &config,
                              const std::string &line, std::string &error);

// Retry every pending handoff. Successful entries are removed from the outbox;
// active/finished Stage2 entries count as successful during crash recovery.
bool ecm_stage2_transfer_flush(const std::string &pending, size_t &remaining,
                               std::string &error);
