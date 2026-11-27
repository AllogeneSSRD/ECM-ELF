#pragma once
#include "geometry.h"
#include "types.h"
#include "ecm/io/stage2_checkpoint.h"
namespace s2prod {
struct TaskPaths {
    fs::path cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir;
    bool explicit_results;
};
struct QueuePolicy {
    uint64_t b2_ratio;
    int worktodo_priority,worktodo_skip;
    std::string finished;
    bool shared_finished = false;
};
using QueueReport=std::function<void(const std::string &,bool)>;
struct CurveQueue {
    const std::vector<Record> &plan;
    const fs::path &save,&results,&log,&progress;
    const TaskDefaults &defaults;
    stage2_queue::State &state;
    uint64_t &completed;
    const QueueReport &report;
    std::function<bool()> cancelled;
    std::function<void(const Record&, const std::string&)> commit_shared;
    void commit(uint64_t position,const std::string &outcome) const;
};

std::string checkpoint_outcome(const Record &,uint64_t b2,const std::string &report,uint64_t b2_min=0);
struct TaskEngine {
    const char *name;
    unsigned input_max_bits = ecm_stage2::max_input_bits; // Zero leaves size admission to the engine.
    EdwardsPreparation edwards_preparation = EdwardsPreparation::Construct;
    std::function<void()> begin;
    std::function<void(const Record &,uint64_t)> validate;
    std::function<void(Options &,const Record &)> prepare;
    std::function<void(const Options &,const Record &,uint64_t,uint64_t)> plan;
    std::function<bool(const CurveQueue &,Options &,uint64_t,uint64_t)> run;
};
int run_task_queue(const TaskEngine &,Options,const TaskPaths &,const QueuePolicy &,const TaskDefaults &);
void append_fatal_log(const std::string& message);
int run_tasks(Options o,const TaskPaths &paths,const QueuePolicy &cfg,const TaskDefaults &defaults);
int curve_worker(Options o);
int child_run(Options &o,const fs::path &save,const Record &r,
              uint64_t b2,uint64_t d,int device,const fs::path &results,const fs::path &log,
              uint64_t batch,const TaskDefaults &ini_settings);
}
