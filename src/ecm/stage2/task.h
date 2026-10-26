#pragma once
#include "types.h"
namespace s2prod {
struct TaskPaths {fs::path cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir;};
struct QueuePolicy {uint64_t b2_ratio;int worktodo_priority,worktodo_skip;std::string finished;};
extern fs::path fatal_log;
int run_tasks(Options o,const TaskPaths &paths,const QueuePolicy &cfg,const TaskDefaults &defaults);
int curve_worker(Options o);
int child_run(Options &o,const fs::path &save,const Record &r,
              uint64_t b2,uint64_t d,int device,const fs::path &results,const fs::path &log,
              uint64_t batch,const TaskDefaults &ini_settings);
}
