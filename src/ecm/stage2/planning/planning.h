#pragma once
#include "../types.h"
#include <optional>
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
void round_giant_bounds(Options &o,const Record &r);
void ensure_portable_calibration(Options &o,const Record &r,const TaskDefaults &settings,const std::vector<Record> &range={});
void merge_portable_tune(const Options &o,const fs::path &destination);
std::vector<Record> portable_queue_range(const fs::path &worktodo,int worker,const fs::path &save_dir);
RuntimeTuneObservation prepare_runtime_tune(const Options &o,const Record &r);
std::shared_ptr<const PreparedRuntimeProfile> prepare_runtime_profile(const Options &o);
void publish_runtime_tune(const Options &o,const RuntimeTuneObservation &observation);
struct ParallelRoute {
    Options options;
    ecm_stage2::tune::portable::Shape shape;
    ecm_stage2::tune::Fields condition;
    std::string scope_key,device_uuid;
    uint64_t reservation=0;
    uint64_t pre_ftree_reservation=0;
};
struct ParallelEvidence {
    uint64_t curves=0;
    double serial_wall=0,serial_middle=0,paired_wall=0,paired_middle=0;
};
inline constexpr uint64_t parallel_memory_margin=256ull<<20;
inline uint64_t parallel_required_free_bytes(uint64_t reservation,unsigned slots) {
    if(slots && reservation>(UINT64_MAX-parallel_memory_margin)/slots)return UINT64_MAX;
    return parallel_memory_margin+reservation*slots;
}
std::optional<ParallelRoute> prepare_parallel_route(Options options,const Record &record,bool writes_log,
                                                    const char *&rejection);
std::vector<ParallelRoute> prepare_parallel_alternatives(const ParallelRoute &primary,const Record &record,
                                                        bool writes_log,bool explore_layout,uint64_t free,
                                                        uint64_t requested_b2,bool round_bounds);
std::optional<uint64_t> parallel_free_memory(int device);
ParallelEvidence read_parallel_evidence(const ParallelRoute &route,unsigned mask,uint64_t preferred_curves);
void record_parallel_window(const ParallelRoute &route,unsigned mask,unsigned workers,
                            uint64_t curves,double wall,double middle,uint64_t required_free);
void run_portable_ntt(const Options &o,const fs::path &destination);
void run_portable_tune(Options o,const TaskDefaults &settings,const fs::path &destination,
    const fs::path &default_worktodo,const fs::path &save_dir);
std::string runtime_route_event(const Options &o);
std::string runtime_tune_json(const RuntimeTuneObservation &observation);
std::string runtime_tune_observation_json(const RuntimeTuneObservation &observation);
std::string select_auto_tuned(Options &o,const Record &r);
std::string select_runtime_tuned(Options &o,const Record &r);
std::string select_tuned(Options &o,const Record &r);
}
