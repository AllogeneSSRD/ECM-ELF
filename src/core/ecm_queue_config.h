#pragma once

// Defaults and input bindings are generated from config/ecm_options.json.
// Worker override rules are shared with Stage2/GUI.
#include "generated/ecm_config_generated.h"

struct EcmQueueConfig : ecm_config::Stage1Values {};

// Unknown keys/tables, duplicate keys and invalid TOML values are errors.
// Matching [workers.N.*] tables override the global stage/shared tables.
// Key names are case sensitive. worker is 1-based (<=0: globals only).
bool ecm_queue_config_load(const std::string &path, EcmQueueConfig &cfg);
bool ecm_queue_config_load(const std::string &path, int worker, EcmQueueConfig &cfg);
bool ecm_queue_config_write_default(const std::string &path);
