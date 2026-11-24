#include "../../io/files.h"
// Unique parent task/queue pipeline and complete isolated curve worker.
#include "task.h"
#include "backend.h"
#include "input.h"
#include "planning/planning.h"
#include "planning/runtime_cache.h"
#include "cuda/host_policy.h"
#include "process.h"
#include "../platform/compute_log.h"
#include "modulus.h"
#include "giant_tree.h"
#include "giant_policy.h"
#include "schedule_policy.h"
#include "active_clock.h"
#include "factorize.h"
#include "ecm/io/stage2_checkpoint.h"
#include "ecm/io/save.h"
#include "ecm/platform/fingerprint.h"
#include "ecm/platform/log_file.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "ecm/io/results.h"
#include "../../io/json.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
namespace s2prod {
// Shared by parent tasks and calibration; only process wire fields cross here.
static CurveProcessRequest curve_request(Options &o,const fs::path &save,const Record &r,
              uint64_t b2,uint64_t d,int device,const fs::path &results,const fs::path &log,
              uint64_t batch,const TaskDefaults &ini_settings) {
    CurveProcessRequest request;
    request.save=save;request.results=results;request.log=log;
    request.log_max_kb=o.log_max_kb;
    request.record={r.offset,r.hash,r.index};request.console_budget={ini_settings.arena,ini_settings.owner_mb};
    request.b2=b2;request.b2_min=o.b2_min;request.d=d;request.device=device;request.batch=batch;
    request.arena=o.arena;
    request.auto_b2=o.auto_b2;
    request.auto_carrier=o.auto_carrier;
    request.auto_max=o.auto_max;
    request.auto_min=o.auto_min;
    request.carrier_exponent=o.carrier_exponent;
    request.condition_tag=o.condition_tag;
    request.cost_profile=o.cost_profile;
    request.debug_file=o.debug_file;
    request.debug_log=o.debug_log;
    request.factor_bits=o.factor_bits;
    request.factor_only=o.factor_only;
    request.factor_timeout=o.factor_timeout;
    request.factorize_hits=o.factorize_hits;
    request.gp=o.gp;
    request.gui_events=o.gui_events;
    request.has_carrier=o.has_carrier;
    request.inferred_carrier=o.inferred_carrier;
    request.log_level=o.log_level;
    request.ntt_counts=o.ntt_counts;
    request.owner_mb=o.owner_mb;
    request.pack_zero=o.pack_zero;
    request.weighted_ntt=o.weighted_ntt;
    request.ntt_two_axis=o.ntt_two_axis;
    request.prediction_policy=o.prediction_policy;
    request.ratio_adjust=o.ratio_adjust;
    request.receipt=o.receipt;
    request.relocation=o.relocation;
    request.giant_tail_balance=o.giant_tail_balance;request.b2_round_chain=o.b2_round_chain;
    request.giant_chain=o.giant_chain;
    // Useful queue curves resolve auto in the child, after the save is read.
    // Explicit calibration already selected a concrete arithmetic identity.
    request.baby_backend=o.tune_child?ecm_stage2::point_backend_name(o.selected_baby_coop):o.baby_backend;
    request.giant_backend=o.tune_child?ecm_stage2::point_backend_name(o.selected_giant_coop):o.giant_backend;
    request.seed_backend=o.tune_child?ecm_stage2::point_backend_name(o.selected_seed_coop):o.seed_backend;
    request.reduce_backend=o.tune_child?ecm_stage2::point_backend_name(o.selected_reduce_coop):o.reduce_backend;
    request.reduce_threads=o.reduce_threads;request.reduce_kernel=o.tune_child && o.selected_reduce_coop?o.selected_reduce_kernel:o.reduce_kernel;
    request.runtime_prepare_cache=o.runtime_prepare_cache;
    request.runtime_prepare_hints=o.runtime_prepare_hints;
    request.runtime_remaining=o.runtime_remaining;
    request.runtime_route_hints=o.runtime_route_hints;
    request.runtime_samples=o.runtime_samples;
    request.tune_min_repeats=o.tune_min_repeats;
    request.short_batch_tune=o.short_batch_tune;
    request.groot_layout=o.groot_layout;
    request.groot_subtree=o.selected_groot_subtree>=0?std::to_string(o.selected_groot_subtree):o.groot_subtree;
    request.stage1_batch=o.stage1_batch;
    request.stage1_csv=o.stage1_csv;
    request.stage1_exponent=o.stage1_exponent;
    request.stage1_mhz=o.stage1_mhz;
    request.stage1_profile=o.stage1_profile;
    request.stage1_seconds=o.stage1_seconds;
    request.publication={o.submission,o.result_user,o.result_computer};
    request.submission_aid=o.submission_aid;
    request.submission_exponent=o.submission_exponent;
    request.submission_known=o.submission_known;
    request.tune_child=o.tune_child;
    request.tune_ignore=o.tune_ignore;
    request.tune_profile=o.tune_profile;
    request.tune_runtime=o.tune_runtime;
    request.fixed_queue_route=o.fixed_queue_route;
    request.queue_requested_b2=o.queue_requested_b2;
    request.queue_phase_gate=o.queue_phase_gate;
    request.queue_workspace_lease=o.queue_workspace_lease;
    request.worker=o.worker;
    request.worker_cpu_affinity=o.worker_cpu_affinity;
    request.engine_log_level=ecm_cuda_stage2_default_log_level()==stage2_log::debug?stage2_log::debug:stage2_log::batches;
    request.observe_selection=[&o](const std::string &text) {
            try {
                const auto event=ecm_results::json::parse(text);
                ecm_stage2::tune::portable::RuntimePauseHints hints;
                hints.assign(event.get("runtime_prepare_hints").text());o.runtime_prepare_hints=hints.text();
                const auto route_text=event.get("runtime_route_hints").text();
                ecm_stage2::tune::portable::RuntimeRouteHints routes;
                if(open_runtime_routes(route_text,routes))o.runtime_route_hints=route_text;else o.runtime_route_hints.clear();
            }catch(const std::runtime_error &){o.runtime_prepare_hints.clear();o.runtime_route_hints.clear();}
    };
    return request;
}
int child_run(Options &o,const fs::path &save,const Record &r,
              uint64_t b2,uint64_t d,int device,const fs::path &results,const fs::path &log,
              uint64_t batch,const TaskDefaults &ini_settings) {
    return run_curve_process(curve_request(o,save,r,b2,d,device,results,log,batch,ini_settings));
}
int curve_worker(Options o) {
    using Clock=std::chrono::steady_clock;
    const auto worker_start=Clock::now();
    Handle gpu_start,gpu_done,gpu_input;
    if(o.queue_gpu_start || o.queue_gpu_done || o.queue_gpu_input) {
        if(stage2_backend::opencl || !o.queue_gpu_start || !o.queue_gpu_done || o.tune_child)
            throw std::runtime_error("incomplete CUDA Stage2 pipeline worker");
        gpu_start.value=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(o.queue_gpu_start));
        gpu_done.value=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(o.queue_gpu_done));
        if(!SetHandleInformation(gpu_start.value,HANDLE_FLAG_INHERIT,0) ||
           !SetHandleInformation(gpu_done.value,HANDLE_FLAG_INHERIT,0))
            throw std::runtime_error("invalid CUDA Stage2 pipeline worker events");
        if(o.tune_runtime!=bool(o.queue_gpu_input))throw std::runtime_error("missing Stage2 pipeline tune feedback");
        if(o.queue_gpu_input) {
            gpu_input.value=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(o.queue_gpu_input));
            if(!SetHandleInformation(gpu_input.value,HANDLE_FLAG_INHERIT,0))
                throw std::runtime_error("invalid Stage2 pipeline tune feedback handle");
        }
    }
    // Internal children bypass the public TOML setup. Bind the forwarded S4
    // request before point setup publishes the per-curve environment.
    configure_s4_backend(o,false,{});
    configure_point_backends(o);
    configure_giant_policy(o);
    validate_groot_layout(o);
    configure_host_wait(o);
    configure_pack_zero(o);
    configure_weighted_ntt(o);
    configure_two_axis_ntt(o);
    // Children ignore the first console interrupt; the parent owns stop policy.
    SetConsoleCtrlHandler(nullptr,TRUE);
    std::setvbuf(stdout,nullptr,_IONBF,0);std::setvbuf(stderr,nullptr,_IONBF,0);
    if(!o.debug_file.empty())stage2_log::open_debug(fs::path(o.debug_file));
    if(o.log_level<0)o.log_level=ecm_cuda_stage2_default_log_level();
    if(ecm_cuda_stage2_set_log_level(o.log_level))throw std::runtime_error("engine rejected log level (development requires debug)");
    stage2_log::level=o.log_level;
    stage2_log::phase("Read and validate Stage1 point");
    if(ecm_cuda_stage2_set_ntt_counts(o.ntt_counts))throw std::runtime_error("NTT length counts require the production Stage2 engine");
    if(ecm_cuda_stage2_check_configuration())throw std::runtime_error("Stage2 engine configuration rejected");
    if (o.factor_only && _putenv_s("NTT_NAME_HITS", "0"))
        throw std::runtime_error("cannot disable optional prime-witness naming");
    if (o.save.empty() || ((!o.checkpoint_outcome || o.tune_child) && o.results.empty()) ||
        !o.index || o.device < 0 || (!o.b2 && !o.auto_b2))
        throw std::runtime_error("incomplete internal curve-worker arguments");
    const auto read_saved_line=[&] {
        std::ifstream input(o.save,std::ios::binary);
        if(!input)throw std::runtime_error("cannot reread save");
        input.seekg(static_cast<std::streamoff>(o.offset));
        std::string line;
        if(!std::getline(input,line) || fingerprint(line)!=o.hash)
            throw std::runtime_error("save changed after planning");
        return line;
    };
    const auto line=read_saved_line();
    const Record r = parse_record(line);
    if(r.edwards && !o.b2 && o.auto_b2)
        throw std::runtime_error("Edwards Stage2 currently requires an explicit B2");
    if(o.carrier_exponent) {
        ecm_stage2::ModulusContext modulus;std::string error;
        if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
    }
    Big n, x, gcd;
    mpz_set_str(n.z, r.n.c_str(), 16); mpz_set_str(x.z, r.x.c_str(), 16);
    mpz_gcd(gcd.z, x.z, n.z);
    if(o.queue_gpu_start) {
        if(o.tune_runtime)try {o.queue_tune_snapshot=prepare_runtime_profile(o);}
        catch(const std::exception &) {} // The normal selector will read a fresh profile on admission.
        stage2_log::print(stage2_log::debug,"stage2_queue_prepare: record=%llu seconds=%.6f\n",
            static_cast<unsigned long long>(o.index),std::chrono::duration<double>(Clock::now()-worker_start).count());
        const auto admission_started=Clock::now();
        const auto admission=WaitForSingleObject(gpu_start.value,INFINITE);
        ecm_stage2::timing::record_lock_wait(admission_started);
        if(admission!=WAIT_OBJECT_0)
            throw std::runtime_error("cannot wait for CUDA Stage2 queue admission");
        if(read_saved_line()!=line)throw std::runtime_error("save changed during Stage2 prefetch");
        if(o.queue_gpu_input) {
            const auto *view=static_cast<const char *>(MapViewOfFile(gpu_input.value,FILE_MAP_READ,0,0,queue_feedback_bytes));
            if(!view)throw std::runtime_error("cannot receive Stage2 pipeline tune feedback");
            const auto *end=static_cast<const char *>(std::memchr(view,0,queue_feedback_bytes));
            std::string payload;
            if(end)payload.assign(view,end);
            UnmapViewOfFile(view);
            if(payload.empty())throw std::runtime_error("empty Stage2 pipeline tune feedback");
            const auto feedback=ecm_results::json::parse(payload);
            o.runtime_prepare_hints=feedback.get("prepare_hints").text();
            o.runtime_route_hints=feedback.get("route_hints").text();
            o.queue_tune_observation=feedback.get("observation").dump();
        }
    }
    std::string result;
    EcmStage2Metrics tune_metrics;
    const uint64_t requested_d=o.d,requested_b2=o.queue_requested_b2?o.queue_requested_b2:o.b2;
    const unsigned requested_carrier=o.carrier_exponent;
#ifdef ECM_STAGE2_OPENCL
    stage2_log::phase("Plan Stage2 bounds and OpenCL payload");
#else
    stage2_log::phase("Plan Stage2 bounds and memory");
#endif
    const auto start = std::chrono::steady_clock::now();
    const auto wait_at_start=ecm_stage2::timing::lock_wait;
    std::string auto_json;
    configure_short_ntt_batches(o);
    if(!o.fixed_queue_route && !o.b2&&o.auto_b2){auto_json=select_auto(o,r);std::cout<<auto_json<<std::endl;}
    o.has_d=requested_d!=0;
    if(o.fixed_queue_route && (!o.has_d || !o.b2 || o.tune_runtime || o.tune_child))
        throw std::runtime_error("incomplete fixed Stage2 queue route");
    auto tuned_json=(o.tune_child || o.fixed_queue_route)?std::string{}:auto_json.empty()?select_tuned(o,r):select_runtime_tuned(o,r);
    if(!o.fixed_queue_route)round_giant_bounds(o,r);
    apply_point_backends(o);
    resolve_groot_subtree(o,r);
    ecm_stage2::GiantPolicyScope point_execution(o.giant_tail_balance?1:0,o.selected_chain);
    const double selection_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double auto_seconds=auto_json.empty()?0:std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (o.b2 <= r.b1 || (o.b2_min && (o.b2_min<r.b1 || o.b2_min>=o.b2)))
        throw std::runtime_error("Stage2 bounds require saved B1 <= B2min < B2");
    RuntimeTuneObservation observation;
    const auto observation_start=std::chrono::steady_clock::now();
    if(!o.relocation && !o.tune_child && o.tune_runtime && !o.tune_profile.empty() && mpz_cmp_ui(gcd.z,1)==0) {
        try {observation=prepare_runtime_tune(o,r);}
        catch(const std::exception &e){std::cerr<<"WARNING: runtime tune preparation skipped: "<<e.what()<<std::endl;}
    }
    if(!tuned_json.empty()) {
        tuned_json.pop_back();tuned_json+=",\"selection_seconds\":"+ecm_stage2::tune::portable::numeric(selection_seconds)+
            ",\"observation_prepare_seconds\":"+ecm_stage2::tune::portable::numeric(std::chrono::duration<double>(
                std::chrono::steady_clock::now()-observation_start).count())+
            ",\"observation_plan_cache_hits\":"+std::to_string(observation.plan_cache_hits)+'}';
        auto event=tuned_json;event.pop_back();event+=",\"runtime_route_hints\":"+json_string(runtime_route_event(o))+'}';
        std::cout<<event<<std::endl;
    }
    std::string status = "stage2_completed";
    int code = 0;
    if (mpz_cmp_ui(gcd.z, 1) > 0 && mpz_cmp(gcd.z, n.z) < 0) {
        status = "factor_in_saved_X";
        result = "\"hits\":1,\"bad_factors\":0,\"factors\":[" + json_string(number(gcd.z, 10)) + "]";
        if(stage2_log::enabled(stage2_log::curve))std::cout << "saved_X_factor: " << number(gcd.z, 10) << '\n';
    } else {
        if (!mpz_cmp(gcd.z, n.z)) throw std::runtime_error("saved X=0 gives no usable Stage1 point");
        ecm_stage2::GrootLayoutScope layout_scope(o.groot_layout);
        ecm_stage2::GrootSubtreeScope subtree_scope(o.selected_groot_subtree==1);
        configure_cuda_wait(o.device);
        if(ecm_cuda_stage2_set_relocation(o.relocation))throw std::runtime_error("relocation requires the production Stage2 engine");
        const EcmStage2GpuHandoff handoff{[](void *handle){
            if(!SetEvent(static_cast<HANDLE>(handle)))throw std::runtime_error("cannot hand off Stage2 GPU before GCD");
        },gpu_done.value};
        code = ecm_cuda_stage2_run(r.n.c_str(), r.x.c_str(), r.sigma, r.b1, o.b2, o.d, o.device,
            [](const char *text, void *ctx) { *static_cast<std::string *>(ctx) = text; }, &result,o.carrier_exponent,
            &tune_metrics,
            r.a24.empty()?nullptr:r.a24.c_str(),
            o.queue_gpu_done && !o.tune_runtime && o.factor_only && early_gpu_handoff_requested()?&handoff:nullptr,o.b2_min);
    }
    if (code || result.empty()) return code ? code : 1;
    if(o.queue_gpu_done && !o.tune_runtime && !SetEvent(gpu_done.value))
        throw std::runtime_error("cannot release CUDA Stage2 queue admission");
    if(o.tune_child) {
        const auto &m=tune_metrics;
        if(status!="stage2_completed" || !m.clean || !m.selftest_cases || !m.checked || m.bad || !m.phases.complete)
            throw std::runtime_error("ECM tune requires a clean no-factor curve with arithmetic checks");
        const auto calibration_factors=ecm_stage2::result_factors(result);
        if(!calibration_factors.empty()) {
            // A verified factor is not a timing sample. Startup calibration
            // may skip this curve; explicit tune still rejects the nonzero exit.
            Big factor;
            for(const auto &text:calibration_factors)
                if(mpz_set_str(factor.z,text.c_str(),10) || mpz_cmp_ui(factor.z,1)<=0 ||
                    mpz_cmp(factor.z,n.z)>=0 || !mpz_divisible_p(n.z,factor.z))
                    throw std::runtime_error("invalid calibration factor");
            append(o.results,"{\"status\":\"verified_factor_not_timing_sample\",\"clean\":1,\"bad\":0,\"selftest_cases\":"+
                std::to_string(m.selftest_cases)+",\"checked\":"+std::to_string(m.checked)+"}");
            return 3;
        }
        if(result.find("\"hits\":0,")!=0)
            throw std::runtime_error("ECM tune requires a clean no-factor curve with arithmetic checks");
        append(o.results,tune_metrics_json(m));return 0;
    }
    const auto verify_start=Clock::now();
    if(!observation.sample.empty()) {
        if(status!="stage2_completed" || result.find("\"hits\":0,")!=0 || !ecm_stage2::result_factors(result).empty())
            observation.sample.clear();
        else try {complete_runtime_tune(observation,tune_metrics);}
        catch(const std::exception &e){observation.sample.clear();std::cerr<<"WARNING: runtime tune sample skipped: "<<e.what()<<std::endl;}
    }
    if(o.queue_gpu_done && o.tune_runtime) {
        std::cout<<"stage2_queue_observation: "<<runtime_tune_observation_json(observation)<<std::endl;
        if(!SetEvent(gpu_done.value))throw std::runtime_error("cannot release CUDA Stage2 tune admission");
    }
    const auto publish_start=Clock::now();
    const auto lock_wait=ecm_stage2::timing::lock_wait-wait_at_start;
    const double lock_wait_seconds=std::chrono::duration<double>(lock_wait).count();
    const double seconds=std::chrono::duration<double>(publish_start-start-lock_wait).count();
    double factor_seconds = 0;
    ecm_results::Submission submission;submission.edwards=r.edwards;submission.exponent=o.submission_exponent;
    submission.when=ecm_results::utc_timestamp();
    submission.known=o.submission_known;submission.aid=o.submission_aid;
    submission.sigma=r.sigma;submission.b1=r.b1;submission.b2=o.b2;
    submission.b2_min=o.b2_min;
    submission.d=tune_metrics.d;submission.poly_size=tune_metrics.p;
    submission.user=o.result_user;submission.computer=o.result_computer;
    submission.factors=ecm_stage2::result_factors(result);
    submission.stage=submission.factors.empty()?0:(status=="factor_in_saved_X"?1:2);
    const auto external_path=fs::u8path(o.submission);
    if(submission.stage==1){submission.d=0;submission.poly_size=0;}
    if(o.checkpoint_outcome) {
        std::cout<<"stage2_checkpoint_outcome: "<<checkpoint_outcome(r,o.b2,ecm_results::submission_json(submission,n.z),o.b2_min)<<std::endl;
    }
    // The primary verified factor is durable before optional decomposition.
    // A failed diagnostic cannot turn a successful curve into a replay/failure.
    if (o.factorize_hits) {
        stage2_log::phase("Optional factor decomposition");
        const auto begin = std::chrono::steady_clock::now();
        try {
            result += ',' + ecm_stage2::factor_details(result, n.z, fs::u8path(o.gp), o.factor_timeout,
                                                       (o.results.empty()?external_path:fs::path(o.results)).parent_path() / "factor_details");
        } catch(const std::exception &error) {
            std::cerr<<"WARNING: optional factor decomposition unresolved: "<<error.what()<<std::endl;
            result+=",\"factorization_complete\":false,\"prime_factors\":[],\"factor_analysis\":[],\"factor_analysis_error\":"+json_string(error.what());
        }
        factor_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    stage2_log::phase("Publish result");
    std::string engine="{\"name\":"+json_string(stage2_backend::name)+",\"status\":"+json_string(status)+
        ",\"save\":"+json_string(o.save)+",\"record\":"+std::to_string(o.index)+
        ",\"record_hash\":"+json_string(std::to_string(o.hash))+",\"worker\":"+std::to_string(o.worker)+
        ",\"device\":"+std::to_string(o.device)+",\"seconds\":"+std::to_string(seconds)+
        ",\"lock_wait_seconds\":"+std::to_string(lock_wait_seconds)+
        ",\"carrier_exponent\":"+std::to_string(o.carrier_exponent)+
        ",\"requested_carrier_exponent\":"+std::to_string(requested_carrier)+
        ",\"groot_subtree\":"+std::to_string(int(tune_metrics.groot_subtree))+
        ",\"requested_D\":"+std::to_string(requested_d)+",\"requested_B2\":"+std::to_string(requested_b2)+
        ",\"giant_tail_balance\":"+std::string(tune_metrics.giant_tail_balance?"true":"false")+
        ",\"giant_tail_balance_requested\":"+std::string(o.giant_tail_balance?"true":"false")+
        ",\"point_policy_replayed\":"+std::string(tune_metrics.point_policy_replayed?"true":"false")+
        ",\"giant_chain_block\":"+std::to_string(o.selected_chain)+
        ",\"baby_coop\":"+std::to_string(tune_metrics.baby_coop)+",\"giant_coop\":"+std::to_string(tune_metrics.giant_coop)+",\"seed_coop\":"+std::to_string(tune_metrics.seed_coop)+
        ",\"reduce_coop\":"+std::to_string(tune_metrics.reduce_coop)+",\"reduce_threads\":"+std::to_string(tune_metrics.reduce_threads)+",\"reduce_kernel\":"+std::to_string(tune_metrics.reduce_kernel)+
        (tuned_json.empty()?"":",\"tune_plan\":"+tuned_json)+
        (o.factorize_hits?",\"factorization_seconds\":"+std::to_string(factor_seconds):"")+
        (o.factor_only?",\"requested_factor_only\":true":"")+
        (o.b2_min && !o.relocation?",\"giant_first\":"+std::to_string(tune_metrics.giant_first)+
            ",\"giant_last\":"+std::to_string(tune_metrics.giant_last)+
            ",\"giant_points\":"+std::to_string(tune_metrics.giant_points):"")+
        (auto_json.empty()?"":",\"auto_b2\":true,\"auto_planning_seconds\":"+std::to_string(auto_seconds)+",\"auto_plan\":"+auto_json)+
        (o.relocation?",\"relocation\":true,\"relocation_replayed\":"+std::string(tune_metrics.relocation_replayed?"true":"false")+
            ",\"giant_first\":"+std::to_string(tune_metrics.giant_first)+",\"giant_last\":"+std::to_string(tune_metrics.giant_last)+
            ",\"giant_points\":"+std::to_string(tune_metrics.giant_points)+",\"relocation_points\":"+std::to_string(tune_metrics.relocation_points)+
            ",\"relocation_prime\":"+std::to_string(tune_metrics.relocation_prime)+
            ",\"relocation_probe_seconds\":"+std::to_string(tune_metrics.relocation_probe_seconds)+
            ",\"relocation_replay_seconds\":"+std::to_string(tune_metrics.relocation_replay_seconds)+
            ",\"relocation_estimate_scope\":\"original_replay_range\"":"")+","+result+"}";
    auto details=ecm_results::json::parse(engine);
    details.members.erase(std::remove_if(details.members.begin(),details.members.end(),[](const auto &p){return p.first=="factors";}),details.members.end());
    const auto extra=",\"queue_receipt\":"+json_string(o.receipt)+",\"stage2-engine\":"+details.dump()+runtime_tune_json(observation);
    const auto receipt_start=Clock::now();
    if(!o.results.empty())try {
        static const auto source_build=ecm_stage2::sha256_file(executable());
        auto internal=ecm_results::internal_json(submission,n.z,0,source_build,extra);internal.pop_back();
        append(o.results,internal);
    }
    catch(const std::exception &e) {
        if(!o.checkpoint_outcome)throw;
        std::cerr<<"WARNING: optional Stage2 JSONL write failed: "<<e.what()<<std::endl;
    }
    const auto receipt_done=Clock::now();
    // Optional cache work follows the durable result. It cannot invalidate the
    // successful curve or cause the parent to replay a committed receipt.
    if(!observation.sample.empty())try {publish_runtime_tune(o,observation);}
    catch(const std::exception &e){std::cerr<<"WARNING: runtime tune write skipped: "<<e.what()<<std::endl;}
    if(o.queue_gpu_done)stage2_log::print(stage2_log::debug,
        "stage2_queue_publish: record=%llu tune_verify=%.6f prepare=%.6f receipt=%.6f tune_write=%.6f\n",
        static_cast<unsigned long long>(o.index),std::chrono::duration<double>(publish_start-verify_start).count(),
        std::chrono::duration<double>(receipt_start-publish_start).count(),
        std::chrono::duration<double>(receipt_done-receipt_start).count(),
        std::chrono::duration<double>(Clock::now()-receipt_done).count());
    if(o.fixed_queue_route) {
        const auto &m=tune_metrics;
        std::cout<<"stage2_parallel_curve: {\"checked_nf\":"
                 <<(m.clean && m.selftest_cases && m.checked && !m.bad && m.phases.complete &&
                    !m.point_policy_replayed && submission.factors.empty()?1:0)
                 <<",\"d\":"<<m.d<<",\"p\":"<<m.p<<",\"b2\":"<<o.b2<<",\"carrier\":"<<o.carrier_exponent
                 <<",\"layout\":"<<int(m.groot_balanced)<<",\"subtree\":"<<int(m.groot_subtree)<<",\"fold\":"<<int(m.fold_resident)
                 <<",\"frontier\":"<<int(m.frontier_resident)<<",\"points\":"<<m.giant_points
                 <<",\"baby\":"<<m.baby_coop<<",\"giant\":"<<m.giant_coop<<",\"seed\":"<<m.seed_coop
                 <<",\"reduce\":"<<m.reduce_coop<<",\"reduce_threads\":"<<m.reduce_threads
                 <<",\"reduce_kernel\":"<<m.reduce_kernel<<"}"<<std::endl;
    }
    std::cout << "stage2_result: record=" << o.index << " seconds=" << seconds
              << " lock_wait_seconds=" << lock_wait_seconds
              << " factorization_seconds=" << factor_seconds
              << " sigma=" << r.sigma << " B1=" << r.b1
              << " curve_model=" << (r.edwards?"ed":"mont")
              << (r.edwards?"":" param=0") << " save=" << json_string(o.save)
              << " " << result << std::endl;
    return 0;
}

namespace {
using QueueClock=std::chrono::steady_clock;
constexpr double parallel_min_gain=1.03;

struct QueueWindow {
    std::vector<double> finishes;
    bool checked=true;
};

bool checked_queue_curve(const std::string &payload,const ParallelRoute &route) {
    if(payload.empty())return false;
    const auto row=ecm_results::json::parse(payload);
    const auto &s=route.shape;
    const std::pair<const char*,uint64_t> expected[]={
        {"checked_nf",1},{"d",s.d},{"p",ecm_stage2::d_catalog::degree(s.d)},
        {"b2",s.b2},{"carrier",s.carrier},{"layout",s.groot_balanced},{"subtree",s.groot_subtree},{"fold",s.fold},
        {"frontier",s.frontier},{"points",s.giant_points()},{"baby",s.baby_coop},{"giant",s.giant_coop},
        {"seed",s.seed_coop},{"reduce",s.reduce_coop},{"reduce_threads",s.reduce_threads},{"reduce_kernel",s.reduce_kernel}
    };
    for(const auto &item:expected)
        if(row.get(item.first).text()!=std::to_string(item.second))return false;
    return true;
}

struct QueueMemory {
    std::optional<uint64_t> free;
    uint64_t required;
    const char *rejection()const {
        if(!free)return "device_memory_query_failed";
        return *free<required || *free<=parallel_memory_margin?"insufficient_device_memory":nullptr;
    }
    std::string fields()const {
        return " required_free_bytes="+std::to_string(required)+" free_bytes="+(free?std::to_string(*free):"unknown");
    }
};
uint64_t queue_memory_required(const ParallelRoute &route,unsigned workers,bool beside_active=false) {
    const auto lease=route.options.queue_workspace_lease;
    const auto slots=lease?(beside_active?0u:1u):workers;
    const auto base=parallel_required_free_bytes(route.reservation,slots);
    if(lease!=2 || (workers<2 && !beside_active))return base;
    return route.pre_ftree_reservation>UINT64_MAX-base?UINT64_MAX:base+route.pre_ftree_reservation;
}
QueueMemory pair_memory(const ParallelRoute &route,unsigned workers,bool beside_active=false) {
    return {parallel_free_memory(route.options.device),queue_memory_required(route,workers,beside_active)};
}

struct PairLease {
    Handle mutex;
    bool held=false;
    const char *rejection(const std::string &device) {
        if(held)return nullptr;
        if(mutex.value==INVALID_HANDLE_VALUE)
            mutex.value=CreateMutexA(nullptr,FALSE,("Local\\ECM-ELF.Stage2.Queue."+device).c_str());
        if(!mutex.value)return "device_queue_lock_failed";
        const auto status=WaitForSingleObject(mutex.value,0);
        held=status==WAIT_OBJECT_0 || status==WAIT_ABANDONED;
        return held?nullptr:status==WAIT_TIMEOUT?"device_queue_busy":"device_queue_lock_failed";
    }
    ~PairLease(){if(held)ReleaseMutex(mutex.value);}
};

QueueWindow run_curve_window(const CurveQueue &queue,Options &o,size_t end,unsigned workers,
                            uint64_t b2,uint64_t d,const ParallelRoute *route=nullptr) {
    struct Active {
        size_t position=0;
        std::unique_ptr<CurveProcess> process;
        QueueClock::time_point started;
        bool admitted=false,released=false,committed=false;
    };
    std::array<Active,3> active;
    auto &state=queue.state;
    const bool persist=!queue.progress.empty();
    const bool checkpoint_outcomes=persist;
    unsigned arithmetic_bits=route?route->shape.arithmetic_bits:std::max(o.carrier_exponent,o.inferred_carrier);
    if(!route)for(size_t i=static_cast<size_t>(state.done);i<end && arithmetic_bits<=4096;++i)
        arithmetic_bits=std::max(arithmetic_bits,unsigned(queue.plan[i].n.size()*4));
    const auto pipeline_level=queue_pipeline_level(arithmetic_bits,workers,o.factor_only);
    const bool pipeline=pipeline_level && (!o.tune_runtime || pipeline_level>=2) && !stage2_backend::opencl &&
        !o.tune_child && !o.relocation && !std::getenv("ECM_STAGE2_PHASE_GATE") &&
        state.reserved.empty() && end>state.done+1;
    const unsigned capacity=pipeline?workers+1:workers;
    const bool tagged=pipeline || workers==2;
    size_t next=static_cast<size_t>(state.done);
    unsigned running=0;
    Active *prepared=nullptr;
    std::string latest_observation="null";
    bool failed=false;
    QueueWindow result;
    const auto begin=QueueClock::now();
    auto observed_from=begin;
    unsigned samples=0;
    double curve_wall_sum=0;
    const bool events=stage2_log::enabled(stage2_log::curve)||o.gui_events;
    const auto completed_count=[&] {
        return state.done+std::count_if(state.reserved.begin(),state.reserved.end(),[](const auto &entry){return entry.completed;});
    };
    const auto report_mode=[&] {
        if(events)queue.report("stage2_queue: workers="+std::to_string(workers)+
            " completed="+std::to_string(completed_count()),false);
    };
    const auto report_start=[&](Active &slot,const char *event) {
        const auto &r=queue.plan[slot.position];
        if(events)
            queue.report(std::string(event)+": "+std::to_string(slot.position+1)+"/"+std::to_string(queue.plan.size())+
                " record="+std::to_string(r.index)+" sigma="+std::to_string(r.sigma)+" B1="+std::to_string(r.b1)+
                " checksum="+(r.checksum?"verified":"absent")+" curve_model="+(r.edwards?"ed":"mont param=0"),false);
    };
    const auto admit_worker=[&](Active &slot) {
        report_start(slot,"curve_start");
        slot.started=QueueClock::now();slot.admitted=true;
        ++running;
        if(pipeline) {
            const auto feedback=o.tune_runtime?"{\"prepare_hints\":"+json_string(o.runtime_prepare_hints)+
                ",\"route_hints\":"+json_string(o.runtime_route_hints)+",\"observation\":"+latest_observation+'}':std::string{};
            slot.process->allow_gpu(feedback);
        }
    };
    const auto can_admit=[&] {
        if(running>=workers)return false;
        if(workers<2)return true;
        const auto memory=pair_memory(*route,running?1:2,running!=0);
        if(!memory.rejection())return true;
        if(pipeline && std::any_of(active.begin(),active.end(),[](const auto &slot){return slot.process && slot.released;}))
            return false; // A finishing worker can still own a CUDA context.
        result.checked=false;
        if(running)return false;
        workers=1;
        queue.report(std::string("stage2_parallel_memory: workers=1 reason=")+memory.rejection()+memory.fields(),true);
        observed_from=QueueClock::now();samples=0;curve_wall_sum=0;
        report_mode();
        return true;
    };
    report_mode();
    while(true) {
        poll_inherited_stop();
        for(auto &slot:active) {
            if(!slot.process)continue;
            const auto code=slot.process->poll();
            if(pipeline && slot.admitted && !slot.released && slot.process->gpu_released()) {
                --running;slot.released=true;
                if(o.tune_runtime)latest_observation=slot.process->gpu_feedback().value_or("null");
                if(events) {
                    std::ostringstream message;
                    message<<"stage2_queue_gpu_free: record="<<queue.plan[slot.position].index<<" wall="
                           <<std::chrono::duration<double>(QueueClock::now()-slot.started).count()
                           <<" feedback="<<int(latest_observation!="null")<<" curve="<<slot.position+1;
                    queue.report(message.str(),false);
                }
            }
            if(!code)continue;
            if(slot.admitted && !slot.released)--running;
            if(prepared==&slot)prepared=nullptr;
            if((*code && !slot.committed) || !slot.admitted) {
                if(events)queue.report("stage2_curve_failed: curve="+std::to_string(slot.position+1),true);
                failed=true;slot.process.reset();continue;
            }
            const auto position=slot.position;
            if(route && result.checked) {
                try {result.checked=!*code && checked_queue_curve(slot.process->queue_sample(),*route);}
                catch(const std::runtime_error &){result.checked=false;}
            }
            const double active_seconds=slot.process->active_seconds();
            slot.process.reset();
            if(checkpoint_outcomes) {
                if(!slot.committed)
                    throw std::runtime_error("successful worker has no checkpoint outcome; queue retained");
                if(*code)queue.report("WARNING: Stage2 worker exited after committing its outcome.",true);
            } else {
                state.reserved.at(position-state.done).completed=true;
                state.advance_completed();
            }
            ++queue.completed;
            if(events) {
                const auto now=QueueClock::now();
                const double wall=std::chrono::duration<double>(now-slot.started).count();
                const auto done=completed_count(),remaining=queue.plan.size()-std::min<uint64_t>(done,queue.plan.size());
                curve_wall_sum+=wall;++samples;
                const double latency=curve_wall_sum/samples;
                const double cadence=workers==2 && samples>=2
                    ? std::chrono::duration<double>(now-observed_from).count()/samples : 0;
                double eta=-1;
                if(!remaining)eta=0;
                else if(!stop_requests&&!failed) {
                    // A partially drained queue is limited by its slowest active curve.
                    unsigned admitted=0;
                    double tail=0;
                    for(const auto &other:active)if(other.process&&other.admitted){
                        ++admitted;
                        tail=std::max(tail,std::max(.1,latency-std::chrono::duration<double>(now-other.started).count()));
                    }
                    if(admitted>=remaining)eta=tail;
                    else if(remaining<=workers)eta=latency;
                    else if(workers==1)eta=remaining*latency;
                    else if(cadence>0){
                        eta=latency+(remaining-workers)*cadence;
                    }
                }
                const auto &r=queue.plan[position];
                std::ostringstream message;
                message<<"curve_done: "<<position+1<<'/'<<queue.plan.size()<<" record="<<r.index<<" sigma="<<r.sigma
                       <<" completed="<<done;
                if(eta>=0 && std::isfinite(eta))message<<" eta="<<std::fixed<<std::setprecision(1)<<eta<<std::defaultfloat<<std::setprecision(6);
                message<<" wall="<<wall<<" s active="<<std::fixed<<std::setprecision(6)<<active_seconds
                       <<std::defaultfloat;
                if(cadence>0 && std::isfinite(cadence))
                    message<<" queue_s_per_curve="<<cadence;
                queue.report(message.str(),false);
            }
            if(route)result.finishes.push_back(std::chrono::duration<double>(QueueClock::now()-begin).count());
        }
        if(prepared && (failed || stop_requests)) {
            // This reservation was never admitted and cannot have a receipt.
            prepared->process.reset();prepared=nullptr;
            state.reserved.pop_back();
        }
        if(prepared && can_admit()) {admit_worker(*prepared);prepared=nullptr;}
        while(!failed && !stop_requests && next<end && !prepared) {
            const auto position=next-state.done;
            if(position<state.reserved.size() && state.reserved[position].completed) {
                ++next;result.checked=false;continue;
            }
            if((pipeline || checkpoint_outcomes) && position==state.reserved.size() &&
               state.reserved.size()>=capacity)break;
            if(!pipeline && !can_admit())break;
            const auto slot=std::find_if(active.begin(),active.begin()+capacity,[](const auto &a){return !a.process;});
            if(slot==active.begin()+capacity)break;
            const auto &r=queue.plan[next];
            slot->position=next;slot->admitted=slot->released=slot->committed=false;
            if(pipeline)report_start(*slot,"curve_prepare");
            if(position==state.reserved.size())
                state.reserved.push_back({next});
            o.runtime_remaining=queue.plan.size()-next;
            auto request=curve_request(o,queue.save,r,b2,d,o.device,queue.results,queue.log,o.batch,queue.defaults);
            if(checkpoint_outcomes)request.commit_outcome=[&,slot](const std::string &outcome) {
                if(!slot->admitted || slot->committed)
                    throw std::runtime_error("unexpected Stage2 worker outcome");
                queue.commit(slot->position,outcome);
                slot->committed=true;
            };
            request.queue_pipeline=pipeline;
            if(tagged)request.queue_curve=slot->position+1;
            if((workers==2 || pipeline) && request.debug_log)
                request.debug_file+=".slot"+std::to_string(slot-active.begin()+1);
            ++next;
            if(!pipeline)admit_worker(*slot);
            slot->process=std::make_unique<CurveProcess>(std::move(request));
            if(pipeline) {
                if(can_admit())admit_worker(*slot);
                else prepared=&*slot;
            }
        }
        HANDLE handles[6];
        DWORD count=0;
        for(const auto &slot:active)if(slot.process) {
            handles[count++]=slot.process->handle();
            if(pipeline && slot.admitted && !slot.released)handles[count++]=slot.process->gpu_done_handle();
        }
        if(!count)break;
        if(WaitForMultipleObjects(count,handles,FALSE,workers==1?50:10)==WAIT_FAILED)
            throw std::runtime_error("Stage2 queue worker wait failed");
    }
    if(failed)throw std::runtime_error("curve failed; queue and completed progress retained; inspect "+queue.log.string());
    return result;
}

std::pair<double,double> queue_window_timing(const CurveQueue &queue,unsigned workers,
                                            uint64_t count,uint64_t sample_curves,const QueueWindow &timing) {
    if(timing.finishes.size()!=count || count<2)return {};
    const double wall=timing.finishes.back();
    const uint64_t middle_count=count>=5?count-3:0;
    const double middle=middle_count?timing.finishes[count-2]-timing.finishes[1]:0;
    std::ostringstream message;
    message<<std::setprecision(9)<<"stage2_parallel_window: workers="<<workers<<" curves="<<count
           <<" wall_seconds="<<wall<<" middle_curves="<<middle_count<<" middle_seconds="<<middle
           <<" middle_curves_per_second="<<(middle>0?middle_count/middle:0)
           <<" checked="<<int(timing.checked);
    queue.report(message.str(),false);
    if(timing.checked && count==sample_curves)
        return {wall,middle_count?middle:wall};
    return {};
}

bool run_curve_queue(const CurveQueue &queue,Options &o,uint64_t b2,uint64_t d) {
    const bool serial=o.parallel==ParallelMode::serial;
    const bool automatic=o.parallel==ParallelMode::automatic;
    const bool paired=o.parallel==ParallelMode::paired;
    const auto mode=static_cast<unsigned>(o.parallel);
    const auto parallel_phase_mask=!serial?stage2_schedule::queue_phase_mask():0;
    const auto parallel_window_curves=o.parallel_tune_curves;
    const auto middle_intervals=parallel_window_curves>=5?parallel_window_curves-3:parallel_window_curves;
    auto &state=queue.state;
    const auto size=queue.plan.size();
    if(!state.reserved.empty()) {
        // First resolve the durable in-flight range, regardless of this run's
        // scheduling preference. Already acknowledged entries are skipped.
        run_curve_window(queue,o,static_cast<size_t>(state.done)+state.reserved.size(),1,b2,d);
    }
    std::optional<ParallelRoute> route;
    ParallelEvidence evidence{parallel_window_curves};
    bool evidence_read=false;
    const auto *geometry_setting=std::getenv("ECM_STAGE2_QUEUE_D_SEARCH");
    const bool automatic_workspace=o.parallel_workspace==0;
    const bool standard_workspace=o.parallel_workspace!=2;
    const bool explicit_geometry_search=automatic && !d && standard_workspace &&
        geometry_setting && std::string_view(geometry_setting)=="1";
    struct GeometryArm {const ParallelRoute *route;ParallelEvidence evidence;bool attempted=false;};
    std::vector<ParallelRoute> neighbours;
    std::optional<ParallelRoute> split_workspace;
    std::vector<GeometryArm> geometry_arms;
    const bool explicit_serial_sample=serial && !o.tune_runtime &&
        o.portable_cli.count("--parallel") && o.portable_cli.count("--tune-profile");
    const bool can_learn_parallel=automatic;
    bool direct_workspace_measured=false;
    PairLease lease;
    while(state.done<size && !stop_requests) {
        const auto remaining=size-static_cast<size_t>(state.done);
        const auto &r=queue.plan[state.done];
        const bool homogeneous=std::all_of(queue.plan.begin()+static_cast<size_t>(state.done),queue.plan.end(),
            [&r](const auto &next){return next.n==r.n && next.b1==r.b1 && next.edwards==r.edwards;});
        const bool requested=remaining>=2 && (!serial || explicit_serial_sample);
        const char *ineligible=!requested?"insufficient_remaining_curves":stage2_backend::opencl?"backend_not_cuda":
            !b2?"unresolved_b2":o.relocation?"relocation_active":!homogeneous?"mixed_curve_shapes":
            std::getenv("ECM_STAGE2_PHASE_GATE")?"external_phase_gate":nullptr;
        if(ineligible) {
            if(!serial)
                queue.report("stage2_parallel: mode="+std::to_string(mode)+" workers=1 reason="+ineligible,false);
            run_curve_window(queue,o,size,1,b2,d);
            break;
        }
        if(!route) {
            Options local=o;local.b2=b2;local.d=d;local.has_d=d!=0;local.runtime_remaining=remaining;
            const char *rejection="single_curve_plan_rejected";
            try {route=prepare_parallel_route(std::move(local),r,!queue.log.empty(),rejection);}
            catch(const std::exception &e){queue.report("stage2_parallel: route unavailable: "+std::string(e.what()),true);}
            if(!route) {
                if(!serial)
                    queue.report("stage2_parallel: mode="+std::to_string(mode)+" workers=1 reason="+rejection,false);
                const auto next=automatic && o.tune_runtime && remaining>=parallel_window_curves?
                    std::min(size,static_cast<size_t>(state.done)+2):size;
                run_curve_window(queue,o,next,1,b2,d);
                continue;
            }
        }
        unsigned workers=paired?2:1;
        size_t count=remaining;
        const char *reason=paired?"forced":serial?"serial":"insufficient_samples";
        const ParallelRoute *selected=&*route;
        const auto primary_memory=!serial?pair_memory(*route,2):QueueMemory{};
        const bool memory_limited=!serial && primary_memory.free && primary_memory.rejection();
        const bool geometry_search=automatic && (automatic_workspace || explicit_geometry_search || !geometry_arms.empty() ||
            (memory_limited && !d && standard_workspace));
        ParallelEvidence split_evidence{parallel_window_curves};
        if(!serial && automatic_workspace && !split_workspace) {
            split_workspace=*route;
            split_workspace->options.queue_workspace_lease=2;
        }
        GeometryArm *geometry_arm=nullptr;
        bool geometry_probe=false;
        if(automatic && !evidence_read) {
            evidence_read=true;
            try {evidence=read_parallel_evidence(*route,parallel_phase_mask,parallel_window_curves);}
            catch(const std::exception &e){queue.report("stage2_parallel: tune evidence unavailable: "+std::string(e.what()),true);}
        }
        if(split_workspace && (geometry_arms.empty() || memory_limited)) {
            try {
                split_evidence=read_parallel_evidence(*split_workspace,parallel_phase_mask,parallel_window_curves);
                // Explicit lease runs used this scope for their identical serial baseline.
                if(!evidence.serial_middle && split_evidence.serial_middle)
                    evidence={split_evidence.curves,split_evidence.serial_wall,split_evidence.serial_middle};
            }catch(const std::exception &e){queue.report("stage2_parallel: workspace evidence unavailable: "+std::string(e.what()),true);}
        }
        const bool direct_workspace=memory_limited && split_workspace && !pair_memory(*split_workspace,2).rejection();
        if(direct_workspace) {
            if(!direct_workspace_measured)
                queue.report("stage2_parallel_admission: candidate=workspace_mode_1 reason=insufficient_device_memory"+
                    primary_memory.fields(),false);
            selected=&*split_workspace;
            workers=2;
            reason="workspace_memory_fallback";
            if(can_learn_parallel && !direct_workspace_measured && !split_evidence.paired_middle &&
               remaining>=parallel_window_curves)
                count=parallel_window_curves;
            direct_workspace_measured=true;
        } else if(geometry_search && evidence.serial_middle) {
            if(geometry_arms.empty()) {
                geometry_arms.push_back({&*route,evidence});
                try {
                    if(memory_limited)queue.report(std::string("stage2_parallel_admission: candidate=primary reason=")+
                        primary_memory.rejection()+primary_memory.fields(),false);
                    if(split_workspace)
                        geometry_arms.push_back({&*split_workspace,split_evidence});
                    if(primary_memory.free && !d && (memory_limited || explicit_geometry_search))
                        neighbours=prepare_parallel_alternatives(*route,r,!queue.log.empty(),
                        o.groot_layout<0,*primary_memory.free,b2,o.b2_round_chain);
                    for(const auto &other:neighbours)geometry_arms.push_back({&other,
                        read_parallel_evidence(other,parallel_phase_mask,parallel_window_curves)});
                }catch(const std::exception &e){queue.report("stage2_parallel_geometry: "+std::string(e.what()),true);}
            }
            const double serial_wall=evidence.serial_wall/evidence.curves;
            const double serial_middle=evidence.serial_middle/middle_intervals;
            double fastest=serial_wall;
            bool fits=false,has_timing=false;
            for(auto &arm:geometry_arms) {
                const QueueMemory memory{primary_memory.free,queue_memory_required(*arm.route,2)};
                if(memory.rejection())continue;
                fits=true;
                if(!arm.evidence.paired_middle) {
                    if(!arm.attempted && can_learn_parallel && remaining>=parallel_window_curves) {
                        geometry_arm=&arm;geometry_probe=true;
                        break;
                    }
                    continue;
                }
                has_timing=true;
                const auto wall=arm.evidence.paired_wall/arm.evidence.curves;
                const auto middle=arm.evidence.paired_middle/middle_intervals;
                if(wall<=fastest && serial_middle>=middle*parallel_min_gain) {
                    geometry_arm=&arm;fastest=wall;
                }
            }
            if(geometry_arm) {
                selected=geometry_arm->route;workers=2;
                if(geometry_probe)count=parallel_window_curves;
                if(selected->options.queue_workspace_lease==2)
                    reason=geometry_probe?"measure_parallel_workspace":"measured_workspace_gain";
                else reason=geometry_probe?"measure_parallel_geometry":"measured_geometry_gain";
            } else if(!fits)reason=primary_memory.rejection();
            else reason=has_timing?"parallel_geometry_no_gain":"parallel_geometry_samples_insufficient";
        } else if(automatic) {
            if(evidence.serial_middle && evidence.paired_middle) {
                const auto gain=evidence.serial_middle/evidence.paired_middle;
                workers=(gain>=parallel_min_gain && evidence.serial_wall>=evidence.paired_wall)?2:1;
                reason=workers==2?"measured_gain":"measured_serial";
            } else if(can_learn_parallel && remaining>=evidence.curves) {
                count=static_cast<size_t>(evidence.curves);
                workers=evidence.serial_middle?2:1;
                reason=workers==2?"measure_parallel":"measure_serial";
            }
        }
        std::string admission;
        if(workers==2){
            const auto memory=pair_memory(*selected,2);
            admission=memory.fields();
            const char* reject=memory.rejection();
            if(!reject)reject=lease.rejection(selected->device_uuid);
            if(reject){
                workers=1;count=remaining;reason=reject;
                selected=&*route;geometry_arm=nullptr;geometry_probe=false;
            }
        }else if(geometry_search&&evidence.serial_middle){
            admission=primary_memory.fields();
        }
        if(geometry_probe)geometry_arm->attempted=true;
        std::ostringstream decision;
        decision<<"stage2_parallel: mode="<<mode<<" workers="<<workers<<" reason="<<reason
                <<" curves="<<count<<" D="<<selected->shape.d<<" P="<<ecm_stage2::d_catalog::degree(selected->shape.d)
                <<" carrier="<<selected->shape.carrier<<" phase_mask="<<(workers==2?parallel_phase_mask:0)
                <<" layout="<<unsigned(selected->shape.groot_balanced)<<" B2="<<selected->shape.b2<<" requested_B2="<<b2
                <<admission;
        if(workers==2)decision<<" giant_gate="<<(stage2_schedule::whole_giant_loop_requested()?"loop":"batch")
                             <<" workspace_mode="<<(selected->options.queue_workspace_lease?2:1);
        if(!geometry_search && automatic && evidence.serial_middle && evidence.paired_middle)
            decision<<" middle_gain="<<evidence.serial_middle/evidence.paired_middle-1
                    <<" wall_gain="<<evidence.serial_wall/evidence.paired_wall-1;
        queue.report(decision.str(),false);
        auto fixed=selected->options;
        fixed.queue_requested_b2=b2;
        fixed.queue_phase_gate=workers==2?std::to_string(parallel_phase_mask)+':'+stage2_queue::token():std::string{};
        if(workers==1)fixed.queue_workspace_lease=0;
        const auto timing=run_curve_window(queue,fixed,static_cast<size_t>(state.done)+count,workers,fixed.b2,fixed.d,selected);
        const auto [wall,middle]=queue_window_timing(queue,workers,count,parallel_window_curves,timing);
        if(middle>0) {
            if(geometry_probe) {
                auto &entry=geometry_arm->evidence;
                entry.curves=count;entry.paired_wall=wall;entry.paired_middle=middle;
            } else if(automatic && !geometry_arm && !direct_workspace) {
                if(workers==2){evidence.paired_wall=wall;evidence.paired_middle=middle;}
                else {evidence.serial_wall=wall;evidence.serial_middle=middle;}
            }
            try {record_parallel_window(*selected,workers==2?parallel_phase_mask:0,workers,count,wall,middle,
                queue_memory_required(*selected,workers));}
            catch(const std::exception &e){queue.report("stage2_parallel: tune write skipped: "+std::string(e.what()),true);}
        }
        if(!timing.checked && !geometry_probe && state.done<size && !stop_requests) {
            run_curve_window(queue,o,size,1,b2,d);
            break;
        }
    }
    return state.done>=size;
}
} // namespace


int run_tasks(Options o,const TaskPaths &paths,const QueuePolicy &cfg,const TaskDefaults &defaults) {
    std::vector<Record> calibration_range;
    TaskEngine engine;
    engine.name="Stage2";
    engine.begin=[&] {
        if(o.save.empty() && o.parallel!=ParallelMode::paired && !o.dry && o.short_calibration)
            calibration_range=portable_queue_range(paths.worktodo,o.worker,paths.tmp);
    };
    engine.prepare=[&](Options &task,const Record &record) {
        if(task.parallel!=ParallelMode::paired && !task.relocation && !task.dry && !task.tune_profile.empty()) {
            auto calibration=task;
            calibration.b2_min=0;
            ensure_portable_calibration(calibration,record,defaults,calibration_range);
        }
    };
    engine.plan=[](const Options &o,const Record &r,uint64_t b2,uint64_t d) {
        const auto device=o.device;
        Options local=o;local.b2=b2;local.has_d=d!=0;
        configure_short_ntt_batches(local);
        if(o.auto_b2&&!b2){
            std::cout<<select_auto(local,r,false)<<std::endl;
            if(!o.tune_profile.empty()){const auto tuned=select_runtime_tuned(local,r);if(!tuned.empty())std::cout<<tuned<<std::endl;}
        }
        else {const auto tuned=select_tuned(local,r);if(!tuned.empty())std::cout<<tuned<<std::endl;}
        round_giant_bounds(local,r);
        ecm_stage2::GiantPolicyScope point_scope(local.giant_tail_balance?1:0,local.selected_chain);
        std::string result;
        if(ecm_cuda_stage2_set_relocation(o.relocation))throw std::runtime_error("relocation requires the production Stage2 engine");
        ecm_stage2::GrootLayoutScope layout_scope(local.groot_layout);
        const int code=ecm_cuda_stage2_plan(r.n.c_str(),r.sigma,r.b1,local.b2,local.d,device,
            [](const char *json,void *ctx){*static_cast<std::string*>(ctx)=json;},&result,local.carrier_exponent,local.b2_min);
        if(code||result.empty())throw std::runtime_error("Stage2 planning failed");
        std::cout<<result<<std::endl;
    };
    engine.run=run_curve_queue;
    return run_task_queue(engine,o,paths,cfg,defaults);
}
}
