#pragma once
#include "../stage2/runtime_types.h"
namespace ecm_stage2 { namespace tune { namespace portable { struct RuntimeRouteHints; } } }
namespace s2prod {
fs::path absolute_from(const fs::path &base, const std::string &p);
void append(const fs::path &path, const std::string &line);
int child_run(Options &o, const fs::path &save, const Record &r,
              uint64_t b2, uint64_t d, int device, const fs::path &results, const fs::path &log,
              uint64_t batch, const TaskDefaults &ini_settings);
std::string curve_memory_record(const std::string &json);
fs::path executable();
uint64_t fingerprint(const std::string &s);
std::string giant_memory_record(const std::string &json);
std::string json_string(const std::string &s);
std::string number(const mpz_t z, int base = 16);
bool open_runtime_routes(const std::string &text,ecm_stage2::tune::portable::RuntimeRouteHints &out);
std::string plan_scalar(const std::string &json,const char *key);
QueueSelection queue_fields(std::string line);
std::vector<Record> records(const fs::path &path, uint64_t skip, uint64_t count);
std::string seal_runtime_routes(const ecm_stage2::tune::portable::RuntimeRouteHints &routes);
std::string trim(std::string s);
ecm_stage2::tune::Fields tune_environment();
std::string tune_metrics_json(const EcmStage2Metrics &m);
uint64_t u64(std::string s,const char *label);
std::string upper(std::string s);
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
