#pragma once

#include <cstddef>
#include <string>
#include <map>
#include <vector>

struct Stage2TransferConfig {
    std::string worktodo;
    std::string finished;
    std::string pending;
    int worker = 1;
    std::vector<int> workers;
    bool auto_workers = false;
    int configured_workers = 1;
    std::map<int,std::string> finished_by_worker;
};

// Park an already-published Stage1 save assignment before advancing its queue.
bool ecm_stage2_transfer_park(const Stage2TransferConfig &config,
                              const std::string &line, std::string &error,
                              int *receiver = nullptr);

// Retry every pending handoff. Successful entries are removed from the outbox;
// active/finished Stage2 entries count as successful during crash recovery.
bool ecm_stage2_transfer_flush(const std::string &pending, size_t &remaining,
                               std::string &error);
