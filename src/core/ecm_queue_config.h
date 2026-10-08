#pragma once

// Defaults and input bindings are generated from config/ecm_options.json.
// Worker overrides and compatibility policies are shared with Stage2/GUI.
#include "generated/ecm_config_generated.h"

struct EcmQueueConfig : ecm_config::Stage1Values {};

// Unknown keys and malformed legacy values keep the existing/default value.
// CLI sections: globals, then [Worker #worker]; other headings are labels.
// Key names are case sensitive. worker is 1-based (<=0: globals only).
bool ecm_queue_config_load(const std::string &path, EcmQueueConfig &cfg);
bool ecm_queue_config_load(const std::string &path, int worker, EcmQueueConfig &cfg);
bool ecm_queue_config_write_default(const std::string &path);
