#include "ecm/stage1/task.h"
#include "../platform/generated/configuration_defaults.h"
#include "libecm_compat.h"

namespace ecm_stage1 {

Stage1ExecutionPreferences::Stage1ExecutionPreferences()
    : stage1_threads(ecm_config::defaults::stage1_stage1_threads),
      naf_w(ecm_config::defaults::stage1_naf_w), device_index(ecm_config::defaults::stage1_device) {
}

Stage1Persistence::Stage1Persistence()
    : save_name_pattern(ecm_config::defaults::stage1_save_name_pattern),
      ckpt_ms(ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS),
      ckpt_backups(ecm_config::defaults::stage1_ckpt_backups) {
}

} // namespace ecm_stage1
