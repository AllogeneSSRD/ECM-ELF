#pragma once

#include "ecm/platform/program_version.h"

namespace ecmgui_v2 {
// Each worker still reports the version of its actual running binary.
inline constexpr auto& version = ecm_program::version;
}
