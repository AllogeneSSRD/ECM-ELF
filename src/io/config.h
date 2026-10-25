#pragma once

// Defaults and input bindings are generated from config/ecm_options.json.
// Worker override rules are shared with Stage2/GUI.
#include "ecm/application/configuration/generated/ecm_config_generated.h"

struct EcmQueueConfig : ecm_config::Stage1Values {
    // Presence only; raw values/locations remain in ParsedConfig. This keeps
    // explicit configuration distinct from mode-specific omitted-value policy.
    std::set<std::string> config_keys;
};

struct ParsedConfig {
    std::string path;
    std::vector<ecm_config::TomlLine> lines;
};

// No defaults, projection or file writes. Every explicit compute value is checked.
bool read_config(const std::string& path, ParsedConfig& parsed);

// Compute keys/tables are strict. GUI-only values are skipped; unknown GUI keys
// produce warnings. NumWorkers remains available/validated for Stage1 routing.
// TOML syntax and duplicate checks apply to every table.
// Matching [workers.N.*] tables override the global stage/shared tables.
// Key names are case sensitive. worker is 1-based (<=0: globals only).
bool ecm_queue_config_load(const std::string& path, EcmQueueConfig& cfg);
bool ecm_queue_config_load(const std::string& path, int worker, EcmQueueConfig& cfg);
bool ecm_queue_config_write_default(const std::string& path);
