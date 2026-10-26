#pragma once
#include "../types.h"
namespace ecm_stage2 { namespace tune { namespace portable { struct RuntimeRouteHints; } } }
namespace s2prod {
bool open_runtime_routes(const std::string &text,ecm_stage2::tune::portable::RuntimeRouteHints &out);
std::string seal_runtime_routes(const ecm_stage2::tune::portable::RuntimeRouteHints &routes);
std::string select_auto(Options &o,const Record &r,bool apply=true);
std::string tune_metrics_json(const EcmStage2Metrics &m);
ecm_stage2::tune::Fields tune_environment();
std::string plan_scalar(const std::string &json,const char *key);
std::string curve_memory_record(const std::string &json);
std::string giant_memory_record(const std::string &json);
void complete_runtime_tune(RuntimeTuneObservation &observation,const EcmStage2Metrics &m);
void configure_short_ntt_batches(const Options &o);
void ensure_portable_calibration(Options &o,const Record &r,const TaskDefaults &settings,const std::vector<Record> &range={});
void merge_portable_tune(const Options &o,const fs::path &destination);
std::vector<Record> portable_queue_range(const fs::path &worktodo,int worker,const fs::path &save_dir);
RuntimeTuneObservation prepare_runtime_tune(const Options &o,const Record &r);
void publish_runtime_tune(const Options &o,const RuntimeTuneObservation &observation);
void run_portable_ntt(const Options &o,const fs::path &destination);
void run_portable_tune(Options o,const TaskDefaults &settings,const fs::path &destination,
    const fs::path &default_worktodo,const fs::path &save_dir);
std::string runtime_route_event(const Options &o);
std::string runtime_tune_json(const RuntimeTuneObservation &observation);
std::string select_auto_tuned(Options &o,const Record &r);
std::string select_runtime_tuned(Options &o,const Record &r);
std::string select_tuned(Options &o,const Record &r);
}
