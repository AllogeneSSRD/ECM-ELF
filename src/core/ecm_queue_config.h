#pragma once

// ECM queue-manager configuration, loaded from an INI file (ecm.ini).
//
// Format is deliberately minimal: `key = value`, lines starting with '#' are
// comments, blank lines are ignored. No third-party INI library is used.
//
// On first run, when the INI file is missing, the queue manager writes a default
// template (with bilingual comments, English first) and then proceeds with the
// built-in defaults.

#include <cstdint>
#include <string>

struct EcmQueueConfig {
    // --- queue / work management ---
    std::string worktodo = "worktodo.txt";
    std::string finished = "worktodo.finished.txt";
    std::string save_sync_dir_1;                 // empty = disabled
    std::string save_sync_dir_2;                 // empty = disabled
    std::string sync_mode = "incremental";       // incremental | full
    std::string log_file = "screen.log";         // empty = no file log

    // --- driver defaults (queue mode) ---
    int device = 0;
    double gpuckpt_seconds = 600.0;
    int verbose = 1;
    uint32_t tpi = 8u;
    int wg_size = 0;                             // 0 = auto
    std::string kernel_mul;
    std::string kernel_sqr;
    std::string kernel_add;
    std::string kernel_sub;
    std::string kernel_special_mult;
    uint32_t sigma = 0;                          // 0 = random
    std::string save_name_pattern = "m{n}_{b1}.save";
    std::string progress_color = "cyan";         // none|red|green|yellow|blue|magenta|cyan|white|grey
};

// Parse `key = value` lines from the INI file at `path` into `cfg`. Unknown keys
// and malformed values are ignored (the field keeps its default). Returns true
// if the file was read (even if empty); false if it could not be opened.
bool ecm_queue_config_load(const std::string &path, EcmQueueConfig &cfg);

// Write a commented default INI template to `path`. Returns true on success.
bool ecm_queue_config_write_default(const std::string &path);
