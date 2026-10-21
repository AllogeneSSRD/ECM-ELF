#include "ecm/application/configuration/runtime.h"

// Process-wide singleton. Defaults (see header) match the old "env var unset" behavior.
EcmRuntimeConfig &ecm_runtime_config() {
    static EcmRuntimeConfig g_config;
    return g_config;
}
