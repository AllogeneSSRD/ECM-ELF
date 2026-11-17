#pragma once
#include "task.h"
#include "input.h"
#include "io/config.h"

namespace s2prod {
inline TaskPaths task_paths(Options &o,const fs::path &cwd,const fs::path &ini,
                            const ecm_config::Stage2Values &s,const EcmQueueConfig &cfg,bool shared_output) {
    const auto base=ini.parent_path();
    const fs::path worktodo = o.worktodo.empty() ? absolute_from(base, s.worktodo) : absolute_from(cwd, o.worktodo);
    const auto& finished_name=shared_output?s.shared_finished_file:s.finished;
    if(finished_name.empty())throw std::runtime_error("finished_file must name a file");
    const fs::path finished = absolute_from(base, finished_name);
    const fs::path tmp = absolute_from(base, s.save_dir.empty()?cfg.tmp_dir:s.save_dir);
    const std::string suffix = o.worker == 1 ? "" : "_" + std::to_string(o.worker);
    const fs::path results = !o.results.empty() ? absolute_from(cwd, o.results) :
        absolute_from(base, s.results.empty() ? ecm_config::worker_file(ecm_config::defaults::stage2_stage2_results_file,o.worker) : s.results);
    const bool explicit_results = !o.results.empty() || !s.results.empty();
    fs::path log;
    o.log_max_kb=shared_output?s.log_file_max_kb:0;
    if (!o.log.empty()) log = absolute_from(cwd, o.log);
    else if (shared_output) { if (!s.shared_log_file.empty()) log = absolute_from(base, s.shared_log_file); }
    else if (!s.has_log) log = absolute_from(base, ecm_config::worker_file(ecm_config::defaults::stage2_stage2_log_file,o.worker));
    else if (!s.log.empty()) log = absolute_from(base, s.log);
    if(o.debug_log) {
        if(o.debug_file.empty())o.debug_file=absolute_from(base,s.debug_file.empty()?ecm_config::worker_file(ecm_config::defaults::stage2_stage2_debug_log_file,o.worker):s.debug_file).string();
        else o.debug_file=absolute_from(cwd,o.debug_file).string();
    }
    const fs::path legacy_progress=s.progress.empty()?fs::path(worktodo.string()+suffix+".progress"):absolute_from(base,s.progress);
    const fs::path progress_dir=!s.progress_dir.empty()?absolute_from(base,s.progress_dir):
        !s.progress.empty()?legacy_progress.parent_path():worktodo.parent_path();
    return {cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir,explicit_results};
}
}
