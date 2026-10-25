// Unique parent task/queue pipeline and complete isolated curve worker.
#include "task.h"
#include "../../storage/queue/owner_lease.h"
#include "input.h"
#include "../planning/stage2.h"
#include "../planning/ecm_stage2_runtime_cache.h"
#include "../../backend/cuda/stage2/host_policy.h"
#include "../../application/configuration/stage2_identity.h"
#include "../../platform/process/stage2_curve_process.h"
#include "../../platform/ecm_stage2_logging.h"
#include "../../core/stage2/ecm_stage2_modulus.h"
#include "ecm_stage2_factorize.h"
#include "../../storage/queue/ecm_stage2_queue_state.h"
#include "../../storage/save/ecm_edwards_io.h"
#include "ecm/platform/fingerprint.h"
#include "ecm/platform/program_version.h"
#include "ecm/storage/results/publication.h"
#include "../../storage/results/ecm_result_json.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
namespace s2prod {
fs::path fatal_log;
// Shared by parent tasks and calibration; only process wire fields cross here.
int child_run(Options &o,const fs::path &save,const Record &r,
              uint64_t b2,uint64_t d,int device,const fs::path &results,const fs::path &log,
              uint64_t batch,const TaskDefaults &ini_settings) {
    CurveProcessRequest request;
    request.save=save;request.results=results;request.log=log;
    request.record={r.offset,r.hash,r.index};request.console_budget={ini_settings.arena,ini_settings.owner_mb};
    request.b2=b2;request.d=d;request.device=device;request.batch=batch;
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
    request.prediction_policy=o.prediction_policy;
    request.publication_task=o.publication_task;
    request.ratio_adjust=o.ratio_adjust;
    request.receipt=o.receipt;
    request.relocation=o.relocation;
    request.runtime_prepare_cache=o.runtime_prepare_cache;
    request.runtime_prepare_hints=o.runtime_prepare_hints;
    request.runtime_remaining=o.runtime_remaining;
    request.runtime_route_hints=o.runtime_route_hints;
    request.runtime_samples=o.runtime_samples;
    request.short_batch_tune=o.short_batch_tune;
    request.stage1_batch=o.stage1_batch;
    request.stage1_csv=o.stage1_csv;
    request.stage1_exponent=o.stage1_exponent;
    request.stage1_mhz=o.stage1_mhz;
    request.stage1_profile=o.stage1_profile;
    request.stage1_seconds=o.stage1_seconds;
    request.submission=o.submission;
    request.submission_aid=o.submission_aid;
    request.submission_exponent=o.submission_exponent;
    request.submission_ini=o.submission_ini;
    request.submission_known=o.submission_known;
    request.tune_child=o.tune_child;
    request.tune_ignore=o.tune_ignore;
    request.tune_profile=o.tune_profile;
    request.tune_runtime=o.tune_runtime;
    request.worker=o.worker;
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
    return run_curve_process(request);
}
int curve_worker(Options o) {
    configure_host_wait(o);
    configure_pack_zero(o);
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
    if (o.save.empty() || o.results.empty() || !o.index || o.device < 0 || (!o.b2 && !o.auto_b2))
        throw std::runtime_error("incomplete internal curve-worker arguments");
    if(!o.tune_child && !o.submission.empty())ecm_results::check_stage2_outputs(o.results,o.submission);
    std::ifstream in(o.save, std::ios::binary);
    if (!in) throw std::runtime_error("cannot reread save");
    in.seekg(static_cast<std::streamoff>(o.offset));
    std::string line;
    if (!std::getline(in, line) || fingerprint(line) != o.hash)
        throw std::runtime_error("save changed after planning");
    const Record r = parse_record(line);
    if(r.edwards && !o.b2 && o.auto_b2)
        throw std::runtime_error("Edwards Stage2 currently requires an explicit B2");
    if(o.carrier_exponent) {
        ecm_stage2::ModulusContext modulus;std::string error;
        if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
    }
    std::string result;
    EcmStage2Metrics tune_metrics;
    const uint64_t requested_d=o.d,requested_b2=o.b2;
    const unsigned requested_carrier=o.carrier_exponent;
    const auto start = std::chrono::steady_clock::now();
    std::string auto_json;
    configure_short_ntt_batches(o);
    if(!o.b2&&o.auto_b2){auto_json=select_auto(o,r);std::cout<<auto_json<<std::endl;}
    o.has_d=requested_d!=0;
    auto tuned_json=o.tune_child?std::string{}:auto_json.empty()?select_tuned(o,r):select_runtime_tuned(o,r);
    const double selection_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double auto_seconds=auto_json.empty()?0:std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (o.b2 <= r.b1) throw std::runtime_error("B2 must be greater than saved B1");
    Big n, x, gcd;
    mpz_set_str(n.z, r.n.c_str(), 16); mpz_set_str(x.z, r.x.c_str(), 16);
    mpz_gcd(gcd.z, x.z, n.z);
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
        configure_cuda_wait(o.device);
        if(ecm_cuda_stage2_set_relocation(o.relocation))throw std::runtime_error("relocation requires the production Stage2 engine");
        code = ecm_cuda_stage2_run(r.n.c_str(), r.x.c_str(), r.sigma, r.b1, o.b2, o.d, o.device,
            [](const char *text, void *ctx) { *static_cast<std::string *>(ctx) = text; }, &result,o.carrier_exponent,
            &tune_metrics,
            r.a24.empty()?nullptr:r.a24.c_str());
    }
    if (code || result.empty()) return code ? code : 1;
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
    if(!observation.sample.empty()) {
        if(status!="stage2_completed" || result.find("\"hits\":0,")!=0 || !ecm_stage2::result_factors(result).empty())
            observation.sample.clear();
        else try {complete_runtime_tune(observation,tune_metrics);}
        catch(const std::exception &e){observation.sample.clear();std::cerr<<"WARNING: runtime tune sample skipped: "<<e.what()<<std::endl;}
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double factor_seconds = 0;
    const auto identity_ini=o.ini.empty()?executable().parent_path()/"ecm.toml":fs::path(o.ini);
    const auto identity=read_result_identity(identity_ini,o.worker,!o.ini.empty());
    ecm_results::Submission submission;submission.edwards=r.edwards;submission.exponent=o.submission_exponent;
    submission.publication_task=o.publication_task;submission.when=ecm_results::utc_timestamp();
    submission.publication_order=o.index;
    submission.save_name=fs::path(o.save).stem().u8string();
    submission.known=o.submission_known;submission.aid=o.submission_aid;
    submission.sigma=r.sigma;submission.b1=r.b1;submission.b2=o.b2;
    submission.d=tune_metrics.d;submission.poly_size=tune_metrics.p;
    submission.user=identity.result_user;submission.computer=identity.result_computer;
    submission.factors=ecm_stage2::result_factors(result);
    submission.stage=submission.factors.empty()?0:(status=="factor_in_saved_X"?1:2);
    const auto external_path=o.submission.empty()?identity_ini.parent_path()/identity.submission:fs::path(o.submission);
    if(submission.stage==1){submission.d=0;submission.poly_size=0;}
    ecm_results::check_stage2_outputs(o.results,external_path);
    ecm_results::publish_submission(external_path,submission,n.z);
    // The primary verified factor is durable before optional decomposition.
    // A failed diagnostic cannot turn a successful curve into a replay/failure.
    if (o.factorize_hits) {
        stage2_log::phase("Optional factor decomposition");
        const auto begin = std::chrono::steady_clock::now();
        try {
            result += ',' + ecm_stage2::factor_details(result, n.z, fs::u8path(o.gp), o.factor_timeout,
                                                       fs::path(o.results).parent_path() / "factor_details");
        } catch(const std::exception &error) {
            std::cerr<<"WARNING: optional factor decomposition unresolved: "<<error.what()<<std::endl;
            result+=",\"factorization_complete\":false,\"prime_factors\":[],\"factor_analysis\":[],\"factor_analysis_error\":"+json_string(error.what());
        }
        factor_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    stage2_log::phase("Publish result");
    std::string engine="{\"name\":\"cuda-ntt\",\"status\":"+json_string(status)+
        ",\"save\":"+json_string(o.save)+",\"record\":"+std::to_string(o.index)+
        ",\"record_hash\":"+json_string(std::to_string(o.hash))+",\"worker\":"+std::to_string(o.worker)+
        ",\"device\":"+std::to_string(o.device)+",\"seconds\":"+std::to_string(seconds)+
        ",\"carrier_exponent\":"+std::to_string(o.carrier_exponent)+
        ",\"requested_carrier_exponent\":"+std::to_string(requested_carrier)+
        ",\"requested_D\":"+std::to_string(requested_d)+",\"requested_B2\":"+std::to_string(requested_b2)+
        (tuned_json.empty()?"":",\"tune_plan\":"+tuned_json)+
        (o.factorize_hits?",\"factorization_seconds\":"+std::to_string(factor_seconds):"")+
        (o.factor_only?",\"requested_factor_only\":true":"")+
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
    static const auto source_build=ecm_stage2::sha256_file(executable());
    auto internal=ecm_results::internal_json(submission,n.z,0,source_build,extra);internal.pop_back();append(o.results,internal);
    // Optional cache work follows the durable result. It cannot invalidate the
    // successful curve or cause the parent to replay a committed receipt.
    if(!observation.sample.empty())try {publish_runtime_tune(o,observation);}
    catch(const std::exception &e){std::cerr<<"WARNING: runtime tune write skipped; observation retained in results: "<<e.what()<<std::endl;}
    std::cout << "stage2_result: record=" << o.index << " seconds=" << seconds
              << " factorization_seconds=" << factor_seconds
              << " sigma=" << r.sigma << " B1=" << r.b1
              << " curve_model=" << (r.edwards?"ed":"mont")
              << (r.edwards?"":" param=0") << " save=" << json_string(o.save)
              << " " << result << std::endl;
    return 0;
}
int run_tasks(Options o,const TaskPaths &paths,const QueuePolicy &cfg,const TaskDefaults &defaults) {
    const auto &[cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir]=paths;
    const int device=o.device;const uint64_t d=o.d,batch=o.batch;
    const bool queue = o.save.empty();
    ecm_queue::OwnerLease owner;
    if (queue) owner.acquire(worktodo.u8string(), o.worker);
    if (queue && o.selection) throw std::runtime_error("queue curve selection comes from ECMSTAGE2; use --save for overrides");
    Handle lock;
    if (queue) {
        if (!fs::exists(worktodo)) throw std::runtime_error("worktodo not found: " + worktodo.string());
        if (!o.dry && !o.plan_only) {
            // Each worker section has one Stage2 consumer. The shared queue edit
            // lock serializes row rewrites across different workers.
            fs::path lockpath(worktodo.string() + ".worker" + std::to_string(o.worker) + ".stage2.lock");
            lock.value = CreateFileW(lockpath.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
            if (lock.value == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot lock worktodo worker (another process may own it)");
        }
    }
    auto same_path=[](const fs::path &a,const fs::path &b) {
        if(a.empty()||b.empty())return false;
        std::error_code error;
        if(fs::equivalent(a,b,error)&&!error)return true;
        return upper(a.lexically_normal().string())==upper(b.lexically_normal().string());
    };
    // A common TOML may share device/save defaults, never writable queue state.
    const std::vector<fs::path> writable={worktodo,finished,results,log,legacy_progress,fs::path(o.debug_file),fs::path(o.submission)};
    for(size_t i=0;i<writable.size();++i) {
        if(!o.tune_profile.empty() && same_path(writable[i],fs::path(o.tune_profile)))throw std::runtime_error("output path conflicts with measured tune profile");
        if(!o.stage1_profile.empty() && same_path(writable[i],fs::path(o.stage1_profile)))throw std::runtime_error("output path conflicts with Stage1 tune profile");
        if(!o.stage1_csv.empty() && same_path(writable[i],fs::path(o.stage1_csv)))throw std::runtime_error("output path conflicts with Stage1 cost CSV");
        if(same_path(writable[i],ini)||same_path(writable[i],executable()))throw std::runtime_error("output path conflicts with configuration or executable");
        for(size_t j=0;j<i;++j)if(same_path(writable[i],writable[j]))throw std::runtime_error("queue, finished, progress, result and log paths must be distinct");
    }

    if(queue && same_path(finished,absolute_from(executable().parent_path(),cfg.finished)))
        throw std::runtime_error("Stage1 and Stage2 finished records must use distinct paths");
    auto report=[&](const std::string &message,bool error=false) {
        (error?std::cerr:std::cout)<<message<<std::endl;
        if(!o.dry&&!o.plan_only&&!log.empty())append(log,message);
    };
    report(std::string(ecm_program::name)+" "+ecm_program::version+" (Stage2)");
    if(!SetConsoleCtrlHandler(console_control,TRUE))throw std::runtime_error("cannot install console stop handler");
    struct ControlGuard { ~ControlGuard(){SetConsoleCtrlHandler(console_control,FALSE);} } control_guard;
    const auto calibration_range=queue && !o.dry && o.short_calibration?portable_queue_range(worktodo,o.worker,tmp):std::vector<Record>{};
    uint64_t completed=0,failed_tasks=0;
    for (;;) {
        poll_inherited_stop();
        fatal_log.clear();
        if(stop_requests)break;
        fs::path save;
        std::string task_line,expected_n;
        std::vector<Record> plan;
        uint64_t skip=o.skip,count=o.curves,b2=o.b2?o.b2:defaults.b2;
        if(queue) {
            std::string read_error;
            const auto read = ecm_worktodo_read_next_line(worktodo.string(), o.worker,
                {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage2,
                task_line, read_error);
            if(read == WorktodoReadResult::Error)throw std::runtime_error(read_error);
            if(read == WorktodoReadResult::Empty)break;
        }
        try {
            if(queue) {
                QueueSelection fields;
                try { fields=queue_fields(task_line); }
                catch(const std::runtime_error &e){throw TaskInputError(e.what());}
                skip=fields.skip;count=fields.count;
                if(!o.b2&&fields.b2)b2=fields.b2;
                Big n;std::string error;
                if(!ecm_compute_stage2_n(fields.task,n.z,error))throw TaskInputError("worktodo N: "+error);
                expected_n=number(n.z);save=absolute_from(tmp,fields.task.save_name);
                o.submission_exponent=fields.task.k=="1" && fields.task.b=="2" && fields.task.c=="-1"?fields.task.n:0;
                o.submission_known=fields.task.factors;o.submission_aid=fields.task.aid;
                o.inferred_carrier=o.auto_carrier && fields.task.k=="1" && fields.task.b=="2" && fields.task.c=="-1" &&
                    fields.task.n>=2 && fields.task.n<=ecm_stage2::max_input_bits?unsigned(fields.task.n):0;
            } else save=absolute_from(cwd,o.save);
            for(const auto &path:writable)if(same_path(save,path))throw std::runtime_error("save path conflicts with writable queue/log/result state");
            if(!o.dry&&!o.plan_only)fatal_log=log;
            if(o.auto_b2&&!b2&&o.cost_profile.empty()&&o.tune_profile.empty())throw std::runtime_error("Auto B2 requires --tune-profile or --cost-profile FILE");
            if(o.carrier_exponent && o.auto_b2 && !b2 && o.tune_profile.empty())throw std::runtime_error("Mersenne carrier Auto B2 requires full ECM tune data");
            plan=records(save,skip,count);
            if(cfg.b2_ratio && !plan.empty()) {
                if(plan.front().b1 > (static_cast<uint64_t>(INT64_MAX)-8192)/cfg.b2_ratio)
                    throw TaskInputError("b2_ratio multiplication exceeds Stage2 B2 range");
                b2=plan.front().b1*cfg.b2_ratio;
            }
            for(const auto &r:plan) {
                if(r.edwards && o.auto_b2 && !b2)throw std::runtime_error("Edwards Stage2 currently requires an explicit B2");
                if((!(o.auto_b2&&!b2)&&b2<=r.b1)||b2>static_cast<uint64_t>(INT64_MAX)-8192)
                    throw TaskInputError("B2 must exceed every saved B1 and fit the engine's signed index range");
                if(!expected_n.empty()&&r.n!=expected_n)throw TaskInputError("worktodo N differs from save N");
                if(queue&&r.b1!=plan.front().b1)throw TaskInputError("queue saves must have the same B1");
                if(r.x=="0")throw TaskInputError("saved X=0 gives no usable Stage1 point");
                if(o.carrier_exponent) {
                    ecm_stage2::ModulusContext modulus;std::string error;
                    // An invalid command option must not mark a valid queue row as erroneous.
                    if(!modulus.configure(r.n.c_str(),16,o.carrier_exponent,error))throw std::runtime_error(error);
                }
            }
        } catch(const TaskInputError &e) {
            if(!queue || o.dry || o.plan_only)throw;
            // Record the reason durably before commenting out the original row.
            report("ERROR: task input: "+std::string(e.what())+" | task="+task_line,true);
            append(finished,"# ERROR reason="+std::string(e.what())+"\n# "+task_line);
            if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::MarkError))
                throw std::runtime_error("cannot mark invalid task; queue changed or rewrite failed");
            ++failed_tasks;
            if(o.once)break;
            continue;
        }
        if(count && plan.size()<count)
            report("WARNING: requested="+std::to_string(count)+" available="+std::to_string(plan.size())+
                   " after skip="+std::to_string(skip)+"; running available curves only.",true);
        if(plan.empty())report("WARNING: no available curves after skip; task will finish with zero curves.",true);
        if(!o.relocation && !o.dry && !plan.empty() && !o.tune_profile.empty())ensure_portable_calibration(o,plan.front(),defaults,calibration_range);
        stage2_queue::State state;
        const bool persist=queue&&!o.dry&&!o.plan_only;
        fs::path progress;
        Handle progress_lock;
        if(persist) {
            const auto fields=queue_fields(task_line);
            stage2_queue::Task task;task.aid=fields.task.aid;task.n=expected_n;
            task.save_sha256=ecm_stage2::sha256_file(save);task.skip=skip;task.total=plan.size();
            task.b2=b2;task.auto_b2=o.auto_b2&&!b2;
            Big target;mpz_set_str(target.z,expected_n.c_str(),16);
            progress=task.path(progress_dir,mpz_sizeinbase(target.z,2));
            for(const auto &path:writable)if(same_path(progress,path))throw std::runtime_error("task checkpoint conflicts with output state");
            if(same_path(progress,save)||same_path(progress,ini)||same_path(progress,executable())||
               (!o.tune_profile.empty()&&same_path(progress,fs::path(o.tune_profile))))
                throw std::runtime_error("task checkpoint conflicts with input");
            fs::create_directories(progress_dir);
            const fs::path lockpath(progress.string()+".lock");
            progress_lock.value=CreateFileW(lockpath.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
            if(progress_lock.value==INVALID_HANDLE_VALUE)throw std::runtime_error("another process owns this Stage2 task checkpoint");
            const auto legacy_bounds=[&] {return ecm_results::legacy_stage2_bounds(o.submission);};
            auto source=stage2_queue::load_task(progress_dir,mpz_sizeinbase(target.z,2),task,state,legacy_bounds);
            const auto match_legacy=[&](const std::string &line) {
                const auto old=queue_fields(line);Big n;std::string error;
                return old.task.aid==task.aid && ecm_compute_stage2_n(old.task,n.z,error) && number(n.z)==task.n;
            };
            if(source.empty() && stage2_queue::import_legacy(legacy_progress,task,match_legacy,state,legacy_bounds))source=legacy_progress;
            else if(source==progress && fs::exists(legacy_progress)) {
                stage2_queue::State previous;
                if(stage2_queue::import_legacy(legacy_progress,task,match_legacy,previous,legacy_bounds) &&
                   previous.run==state.run && previous.publication_task==state.publication_task)source=legacy_progress;
            }
            const auto previous_total=state.total,previous_b2=state.b2;
            const auto previous_auto=state.auto_b2;
            const auto pending_results=state.pending_results.empty()?results:fs::path(state.pending_results);
            if(!state.pending.empty() && stage2_queue::result_written(pending_results,state.pending)) {
                ++state.done;state.pending.clear();state.pending_results.clear();state.save(progress);
                report("Recovered completed curve from durable result receipt.");
            }
            if(state.done>plan.size() && records(save,skip,state.done).size()!=state.done)
                throw std::runtime_error("Stage2 completed progress exceeds the available save records");
            if(!state.pending.empty() && (state.total!=task.total || state.b2!=task.b2 || state.auto_b2!=task.auto_b2)) {
                // A force-stopped attempt with no durable outcome can be abandoned.
                // A receipt written before a torn JSONL append must first be
                // resolved under the original policy, rather than counted twice.
                const auto next=records(save,skip+state.done,1);
                if(next.empty())throw std::runtime_error("Stage2 pending curve is absent from its original save");
                if(!ecm_results::has_stage2_attempt(o.submission,state.publication_task,next[0].index,
                        target.z,next[0].sigma,next[0].b1,next[0].edwards)) {
                    state.pending.clear();state.pending_results.clear();
                    report("Abandoned uncommitted Stage2 curve attempt before applying the new task policy.");
                }
            }
            state.reconfigure(task);state.save(progress);
            if(!source.empty() && source!=progress) {
                // Retain old bytes outside checkpoint discovery; never replace an archive.
                const fs::path archive(source.string()+".migrated."+ecm_stage2::sha256_file(source).substr(0,16));
                const bool archived=MoveFileExW(source.c_str(),archive.c_str(),0)!=0;
                if(!archived)throw std::runtime_error("cannot archive imported Stage2 checkpoint; preserve both files and resolve the source before continuing: "+source.string());
                report("stage2_resume_import: completed="+std::to_string(state.done)+" total="+std::to_string(state.total)+
                    " legacy_archived="+std::to_string(archived)+" checkpoint="+progress.string());
            }
            if(!source.empty() && (previous_total!=task.total || previous_b2!=task.b2 || previous_auto!=task.auto_b2))
                report("stage2_resume_override: previous_total="+std::to_string(previous_total)+" requested="+std::to_string(task.total)+
                    " completed="+std::to_string(state.done)+" effective_total="+std::to_string(state.total)+
                    " previous_B2="+std::to_string(previous_b2)+" B2="+std::to_string(task.b2)+
                    "; completed curves retain their actual bounds.",true);
            if(state.done>plan.size())report("WARNING: requested="+std::to_string(plan.size())+
                " is below already completed="+std::to_string(state.done)+
                "; no further curves will run; sealing all completed curves (factor hits reported separately).",true);
        }
        if(stage2_log::enabled(stage2_log::curve)||o.dry) {
            if(o.gui_events && !task_line.empty())std::cout<<"START: "<<task_line<<std::endl;
            std::ostringstream message;
            message << "stage2_plan: save=" << save.string() << " requested=" << (count?std::to_string(count):"all")
                    << " available=" << plan.size() << " completed=" << state.done << " remaining=" << (state.done<plan.size()?plan.size()-state.done:0)
                    << " B2=" << b2 << " D=" << d << " device=" << device << " worker=" << o.worker
                    << " log=" << log.string() << " results=" << results.string();
            if(persist)message<<" checkpoint="<<progress.string();
            report(message.str());
        }
        o.publication_task=persist?state.publication_task:std::to_string(ecm_edwards::hash(ecm_stage2::sha256_file(save)+"|"+std::to_string(skip)+"|"+std::to_string(plan.size())+"|"+std::to_string(b2)));
        bool task_completed=true;
        for(size_t i=static_cast<size_t>(state.done);i<plan.size();++i) {
            poll_inherited_stop();
            if(stop_requests){task_completed=false;break;}
            const auto &r=plan[i];
            o.runtime_remaining=plan.size()-i;
            if(stage2_log::enabled(stage2_log::curve)||o.dry)
                report("curve_start: "+std::to_string(i+1)+"/"+std::to_string(plan.size())+" record="+std::to_string(r.index)+
                       " sigma="+std::to_string(r.sigma)+" B1="+std::to_string(r.b1)+" checksum="+(r.checksum?"verified":"absent")+
                       " curve_model="+(r.edwards?std::string("ed"):std::string("mont"))+(r.edwards?"":" param=0"));
            if(o.dry)continue;
            if(o.plan_only) {
                Options local=o;local.b2=b2;local.has_d=d!=0;
                configure_short_ntt_batches(local);
                if(o.auto_b2&&!b2){
                    std::cout<<select_auto(local,r,false)<<std::endl;if(o.tune_profile.empty())continue;
                    const auto tuned=select_runtime_tuned(local,r);if(!tuned.empty())std::cout<<tuned<<std::endl;
                }
                else {const auto tuned=select_tuned(local,r);if(!tuned.empty())std::cout<<tuned<<std::endl;}
                std::string result;
                if(ecm_cuda_stage2_set_relocation(o.relocation))throw std::runtime_error("relocation requires the production Stage2 engine");
                const int code=ecm_cuda_stage2_plan(r.n.c_str(),r.sigma,r.b1,local.b2,local.d,device,
                    [](const char *json,void *ctx){*static_cast<std::string*>(ctx)=json;},&result,local.carrier_exponent);
                if(code||result.empty())throw std::runtime_error("Stage2 planning failed");
                std::cout<<result<<std::endl;continue;
            }
            if(!o.tune_child)ecm_results::check_stage2_outputs(results,o.submission,persist && state.done>0);
            if(persist) {
                // Publish intent first, then launch. Only a complete result with
                // this random receipt can resolve an ambiguous interrupted exit.
                state.pending=stage2_queue::token();state.pending_results=results.string();state.save(progress);o.receipt=state.pending;
            }
            const auto begin=std::chrono::steady_clock::now();
            if(child_run(o,save,r,b2,d,device,results,log,batch,defaults))
                throw std::runtime_error("curve failed; queue and completed progress retained; inspect "+log.string());
            if(persist) {
                if(!stage2_queue::result_written(results,state.pending))throw std::runtime_error("successful worker has no durable result receipt; queue retained");
                ++state.done;state.pending.clear();state.pending_results.clear();state.save(progress);
            }
            ++completed;
            if(stage2_log::enabled(stage2_log::curve)) {
                std::ostringstream message;
                message << "curve_done: " << i+1 << '/' << plan.size() << " record=" << r.index << " sigma=" << r.sigma
                        << " wall=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count() << " s";
                report(message.str());
            }
        }
        if(task_completed && !o.dry && !o.plan_only && !o.tune_child) {
            ecm_results::check_stage2_outputs(results,o.submission,persist && state.done>0);
            ecm_results::seal_submissions(o.submission,2,o.publication_task,true);
        }
        if(!queue||o.dry||o.plan_only)break;
        if(state.done<state.total)break;
        std::string current;
        std::string read_error;
        const auto read = ecm_worktodo_read_next_line(worktodo.string(), o.worker,
            {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage2,
            current, read_error);
        if(read == WorktodoReadResult::Error)throw std::runtime_error(read_error);
        if(read != WorktodoReadResult::Task||current!=task_line)
            throw std::runtime_error("task completed but worktodo changed; queue and progress retained");
        if(!stage2_queue::finished_written(finished,state.run,task_line))
            append(finished,"# stage2_task_id="+state.run+"\n"+task_line);
        if(!ecm_worktodo_advance(worktodo.string(),o.worker,task_line,WorktodoAction::Remove))
            throw std::runtime_error("task completed but queue could not be advanced; progress retained");
        // A complete stale state is safe to replace if a crash happens here.
        if(!DeleteFileW(progress.c_str()) && GetLastError()!=ERROR_FILE_NOT_FOUND)
            throw std::runtime_error("task advanced but cannot remove completed progress file");
        report("task_done: requested="+(count?std::to_string(count):"all")+" available="+std::to_string(plan.size())+
               " completed="+std::to_string(state.done));
        if(o.once||stop_requests)break;
    }
    if(stage2_log::enabled(stage2_log::curve)||o.dry)
        report(std::string(o.plan_only?"plan_complete":o.dry?"dry_run_complete":stop_requests?"stage2_stopped":"stage2_complete")+
               ": curves_this_run="+std::to_string(completed)+" invalid_tasks="+std::to_string(failed_tasks));
    return failed_tasks?1:0;
}
}
