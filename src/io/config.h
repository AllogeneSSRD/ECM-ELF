#pragma once

// Defaults and input bindings are generated from config/ecm_options.json.
// Worker override rules are shared with Stage2/GUI.
#include "ecm/application/configuration/generated/ecm_config_generated.h"
#include "ecm/stage1/selection.h"
#include <map>

struct EcmQueueConfig : ecm_config::Stage1Values {
    // Presence only; raw values/locations remain in ParsedConfig. This keeps
    // explicit configuration distinct from mode-specific omitted-value policy.
    std::set<std::string> config_keys;
    ecm_stage1::Selection selection;
    std::vector<unsigned> affinity_cpus;
};

struct ParsedConfig {
    struct Stage1Scope {
        // Only keys in bindings are present. Other fields are never projected.
        ecm_config::Stage1Values values;
        ecm_stage1::Selection selection;
        std::vector<unsigned> affinity_cpus;
        std::vector<const ecm_config::Binding<ecm_config::Stage1Values>*> bindings;
    };
    std::string path;
    std::vector<ecm_config::TomlLine> lines;
    std::map<int, Stage1Scope> stage1;
};

// Defaults followed by global values, then the selected worker's explicit
// values. Uses typed values retained by read_config; no source text is
// reparsed.
void project_stage1_config(const ParsedConfig&, int worker, EcmQueueConfig&);

// No defaults, projection or file writes. Every explicit compute value is
// checked.
bool read_config(const std::string& path, ParsedConfig& parsed);

// Compute keys/tables are strict. GUI-only values are skipped; unknown GUI keys
// produce warnings. NumWorkers remains available/validated for Stage1 routing.
// TOML syntax and duplicate checks apply to every table.
// Matching [workers.N.*] tables override the global stage/shared tables.
// Key names are case sensitive. worker is 1-based (<=0: globals only).
bool ecm_queue_config_load(const std::string& path, EcmQueueConfig& cfg);
bool ecm_queue_config_load(const std::string& path, int worker, EcmQueueConfig& cfg);
bool ecm_queue_config_write_default(const std::string& path);
