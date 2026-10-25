#include "execution_contract.h"
#include "../../storage/checkpoint/ecm_checkpoint.h"
#include "../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "ecm/core/stage1/selection.h"
#include "../../storage/save/ecm_edwards_io.h"
#include "ecm/storage/results/publication.h"
#include "../../platform/generated/configuration_defaults.h"
#include "ecm/platform/logging.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <cstring>
#include <cmath>
#include "sync/files.h"

namespace ecm_stage1 {
struct TaskResult::State : ExecutionResult {ecm_save::Range save_range;std::string completed_checkpoint;};

Stage1ExecutionPreferences::Stage1ExecutionPreferences()
    : stage1_threads(ecm_config::defaults::stage1_stage1_threads),
      naf_w(ecm_config::defaults::stage1_naf_w),
      device_index(ecm_config::defaults::stage1_device) {}
Stage1Persistence::Stage1Persistence()
    : save_name_pattern(ecm_config::defaults::stage1_save_name_pattern),
      ckpt_ms(ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS),
      ckpt_backups(ecm_config::defaults::stage1_ckpt_backups) {}

TaskResult::TaskResult() : data_(std::make_unique<State>()) {}
TaskResult::~TaskResult() = default;
TaskResult::TaskResult(TaskResult &&) noexcept = default;
TaskResult &TaskResult::operator=(TaskResult &&) noexcept = default;
TaskStatus TaskResult::status() const noexcept {return data_?data_->status:TaskStatus::Failed;}
FailureSource TaskResult::failure_source() const noexcept {return data_?data_->failure_source:FailureSource::Execution;}
int TaskResult::task_return() const noexcept {return data_?data_->task_return:ECM_ERROR;}
int TaskResult::legacy_return() const noexcept {return data_?data_->ret:ECM_ERROR;}
uint32_t TaskResult::curves() const noexcept {return data_?data_->curves:0;}
uint32_t TaskResult::first_sigma() const noexcept {return data_?data_->firstsigma:0;}
bool TaskResult::has_factors() const noexcept {return data_ && data_->factors;}
bool TaskResult::has_found() const noexcept {return data_ && data_->array_found;}
bool TaskResult::has_sigmas() const noexcept {return data_ && data_->sigmas;}
mpz_srcptr TaskResult::factor(uint32_t i) const noexcept {return has_factors()?data_->factors[i]:nullptr;}
int TaskResult::found(uint32_t i) const noexcept {return has_found()?data_->array_found[i]:ECM_ERROR;}
uint64_t TaskResult::sigma(uint32_t i) const noexcept {return has_sigmas()?data_->sigmas[i]:0;}

const ecm_save::Range &TaskResult::save_range() const noexcept {return data_->save_range;}
const std::string &TaskResult::completed_checkpoint() const noexcept {return data_->completed_checkpoint;}

static void stage1_copy_save_history(const Stage1Options &opt,const std::string &destination) {
    if(opt.persistence.legacy_save_source.empty())return;
    const auto source=opencl_ecm_resolve_data_path_buf(opt.persistence.legacy_save_source.c_str());
    const auto target=opencl_ecm_resolve_data_path_buf(destination.c_str());
    if(!opencl_ecm_ensure_parent_dir(target.c_str()))throw std::runtime_error("cannot create prefixed save directory");
    ecm_sync::FileLock publication(target);
    if((std::filesystem::exists(target) && std::filesystem::file_size(target)) ||
       !std::filesystem::exists(source) || !std::filesystem::file_size(source))return;
    std::ifstream input(source,std::ios::binary);if(!input)throw std::runtime_error("cannot read legacy save history");
    std::string line;uint64_t records=0;
    const bool ed=opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards;
    const auto param=opt.execution.selection.target==ecm_stage1::ExecutionTarget::Gpu?opt.execution.selection.gpu_param:0;
    const auto trim=[](std::string s){const auto a=s.find_first_not_of(" \r\n\t"),b=s.find_last_not_of(" \r\n\t");return a==s.npos?std::string{}:s.substr(a,b-a+1);};
    while(std::getline(input,line)) {
        auto text=trim(line);
        if(text.compare(0,3,"\xef\xbb\xbf")==0)text.erase(0,3);
        if(text.empty() || text[0]=='#' || text[0]==';')continue;
        std::string method,kind="0";size_t at=0;
        while(at<text.size()) {
            const auto end=text.find(';',at);const auto field=trim(text.substr(at,end==text.npos?end:end-at));
            at=end==text.npos?text.size():end+1;const auto eq=field.find('=');if(eq==field.npos)continue;
            const auto key=trim(field.substr(0,eq)),value=trim(field.substr(eq+1));
            if(key=="METHOD")method=value;else if(key=="PARAM")kind=value;
        }
        if((ed && method!="ECM-EDWARDS-AM") || (!ed && (method!="ECM" || kind!=std::to_string(param))))
            throw std::runtime_error("legacy save belongs to another curve/parametrization; choose a separate curve-prefixed save name");
        ++records;
    }
    if(input.bad())throw std::runtime_error("cannot read complete legacy save history");
    if(!records)return;
    if(!opencl_ecm_ensure_parent_dir(target.c_str()))throw std::runtime_error("cannot create prefixed save directory");
    ecm_sync::write_atomic(std::filesystem::u8path(target),ecm_sync::read_file(std::filesystem::u8path(source)));
    ecm_ts_fprintf(stdout,"Stage1 save history copied: source=%s target=%s records=%llu original_retained=1\n",
                   source.c_str(),target.c_str(),static_cast<unsigned long long>(records));
}

// Retained legacy sources must not replay curves already durably published.
static ecm_results::Submission stage1_submission(const Stage1Options &opt,const mpz_t N,uint64_t b1) {
    ecm_results::Submission r;r.b1=b1;r.b2=b1;r.stage1_only=true;
    r.edwards=opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards;
    r.param=opt.execution.selection.target==ecm_stage1::ExecutionTarget::Gpu?opt.execution.selection.gpu_param:0;
    r.user=opt.identity.result_user;r.computer=opt.identity.result_computer;r.aid=opt.identity.result_aid;
    if(opt.identity.target_kbnc) {
        r.kbnc=true;r.k=opt.identity.handoff_k_str;r.b=opt.identity.handoff_b;r.exponent=opt.identity.handoff_n;r.c=opt.identity.handoff_c;
        std::istringstream known(opt.identity.handoff_factors);std::string f;
        while(std::getline(known,f,','))if(!f.empty())r.known.push_back(f);
    }
    // Validate complete metadata against the actual modulus before running.
    (void)ecm_results::target_exponent(r,N);return r;
}

static void validate_saved_batch(const ecm_save::Range &range,const Stage1Options &opt,mpz_srcptr expected,double bound,const ExecutionResult &result) {
    if(!range.valid)return;
    const bool ed=opt.execution.selection.curve==CurveModel::Edwards;
    const bool legacy_ed=ed && result.curve_kinds.empty();
    std::vector<uint64_t> sigmas;
    for(uint32_t i=0;i<result.curves;++i)
        if(!ed || result.curve_kinds.empty() || result.curve_kinds[i]!=ExecutionResult::CurveKind::CompletedWithoutPoint)
            sigmas.push_back(result.sigmas[i]);
    if(!legacy_ed && range.records.size()!=sigmas.size())throw std::runtime_error("saved batch count differs from completed point count");
    ecm_save::validate_batch(range,expected,ed?uint64_t(bound):uint64_t(std::llround(bound)),
        ed?-1:opt.execution.selection.target==ExecutionTarget::Cpu?0:opt.execution.selection.gpu_param,sigmas,legacy_ed);
    if(legacy_ed)for(uint32_t i=0;i<result.curves;++i)if(result.array_found[i]!=ECM_NO_FACTOR_FOUND) {
        const auto token="SIGMA="+std::to_string(result.sigmas[i])+";";
        if(std::none_of(range.records.begin(),range.records.end(),[&](const auto &line){return line.find(token)!=line.npos;}))
            throw std::runtime_error("saved batch omitted a completed factor");
    }
}

bool stage1_publish_factor(const Stage1Options &opt,mpz_srcptr n,uint64_t b1,uint64_t sigma,mpz_srcptr factor) {
    static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);
    try {
        auto r=stage1_submission(opt,n,b1);r.stage=1;r.sigma=sigma;r.when=ecm_results::utc_timestamp();
        r.factors.push_back(ecm_results::decimal(factor));(void)ecm_results::submission_json(r,n);return true;
    } catch(const std::exception &e){ecm_ts_fprintf(stderr,"ERROR: Stage1 factor validation: %s\n",e.what());return false;}
}

// Owns batch state/lock lifetime only; task sequencing stays in run_stage1_task.
class Stage1TaskSession {
    bool active_ = false;
public:
    Stage1TaskSession() = default;
    Stage1TaskSession(const Stage1TaskSession &) = delete;
    Stage1TaskSession &operator=(const Stage1TaskSession &) = delete;
    ~Stage1TaskSession() { close(false); }
    void begin(const std::string &identity,uint32_t curves,bool fixed,uint64_t sigma,bool defer=false) {
        active_=true; // begin_batch may acquire its lock before throwing.
        ecm_stage1_begin_batch(identity,curves,fixed,sigma,defer);
    }
    void close(bool completed) {
        if(!active_)return;
        ecm_stage1_end_batch(completed);
        active_=false;
    }
};

TaskResult run_stage1_task(const Stage1Request &request,const TaskDependencies &deps) {
    const auto N=request.modulus;
    const double B1=request.b1,B2=request.b2;
    const uint32_t curves=request.curves;
    const auto &savefilename=request.save_file,&n_expr=request.expression;
    const bool saveappend=request.append_save;
    Stage1Options opt=request.options;
    TaskResult result;
    auto *out=result.data_.get(); // Private execution payload; caller receives an owner.
    Stage1TaskSession session;
    auto phase=FailureSource::Preparation;
    try {
        if(!opt.identity.target_kbnc) {
            // Direct CLI expressions retain k*b^n+c for naming and reporting.
            std::smatch m;
            const std::regex form(R"(^\s*\(?\s*(?:(\d+)\s*\*\s*)?(\d+)\s*\^\s*(\d+)\s*([+-])\s*(\d+)\s*\)?\s*$)");
            if(std::regex_match(n_expr,m,form)) {
                const auto b=std::stoull(m[2]),n=std::stoull(m[3]);const auto c=std::stoll(m[5])*(m[4]=="-"?-1:1);
                if(b<=UINT32_MAX && n<=UINT32_MAX && c>=INT32_MIN && c<=INT32_MAX) {
                    opt.identity.target_kbnc=true;opt.identity.handoff_k_str=m[1].matched?m[1].str():"1";
                    opt.identity.handoff_b=static_cast<uint32_t>(b);opt.identity.handoff_n=static_cast<uint32_t>(n);opt.identity.handoff_c=static_cast<int32_t>(c);
                }
            }
        }
        const auto base=stage1_submission(opt,N,uint64_t(B1));
        (void)ecm_results::submission_json(base,N);
        const auto same=[](const std::string &a,const std::string &b){if(a.empty() || b.empty())return false;std::error_code ec;
            return std::filesystem::absolute(a).lexically_normal()==std::filesystem::absolute(b).lexically_normal() || std::filesystem::equivalent(a,b,ec);};
        if(same(opt.persistence.results_file,opt.persistence.submission_file) || same(savefilename,opt.persistence.results_file) || same(savefilename,opt.persistence.submission_file))
            throw std::runtime_error("Stage1 save, internal results and submission paths must differ");
        ecm_results::check_stage1_outputs(opt.persistence.results_file,opt.persistence.submission_file);
        ecm_edwards::Integer plus;mpz_add_ui(plus.v,N,1);
        const auto exponent=opt.identity.target_kbnc?opt.identity.handoff_n:(mpz_popcount(plus.v)==1?mpz_scan1(plus.v,0):0);
        const auto name=std::string(base.edwards?".e":".m"+std::to_string(base.param))+"_s1_"+
            (exponent?"m"+std::to_string(exponent):"bit"+std::to_string(mpz_sizeinbase(N,2)))+"_"+std::to_string(uint64_t(B1))+".ckpt";
        const auto path=opt.persistence.tmp_dir.empty()?std::string{}:(std::filesystem::path(opt.persistence.tmp_dir)/name).lexically_normal().make_preferred().string();
        if(!path.empty())std::filesystem::create_directories(opt.persistence.tmp_dir);
        ecm_stage1_set_backup_limit(opt.persistence.ckpt_backups);ecm_stage1_set_checkpoint_path(path);
        ExecutionOptions execution(opt);
        {
            auto identity=ecm_edwards::hex(N)+"|"+(base.edwards?"ed":"mont"+std::to_string(base.param))+"|"+
                std::to_string(uint64_t(B1))+"|"+std::to_string(int(opt.execution.selection.scalar_construction))+"|"+
                ecm_stage1::execution_target_name(opt.execution.selection.target);
            if(opt.execution.selection.target==ecm_stage1::ExecutionTarget::Cpu)
                identity+="|"+std::to_string(opt.execution.naf_w?opt.execution.naf_w:deps.default_naf_window())+"|"+std::to_string(int(opt.execution.selection.cpu_implementation));
            else if(!base.edwards) {
                identity+="|"+execution.gpu_algorithm;
            }
            session.begin(identity,curves,opt.identity.sigma_fixed,opt.identity.fixed_sigma64,
                deps.defer_gpu_checkpoint_validation && opt.execution.selection.target==ExecutionTarget::Gpu);
        }
        stage1_copy_save_history(opt,savefilename);
        const auto begin=std::chrono::steady_clock::now();
        phase=FailureSource::Execution;
        ecm_save::Capture publication;
        const int rc=deps.execute(N,B1,B2,curves,savefilename,saveappend,n_expr,execution,out);
        if(rc==ECM_ERROR){if(out->completion==ExecutionResult::Completion::Legacy && !ecm_stage1_stop_requested() && !out->publish_failed)out->prepare_failed=true;session.close(false);out->finish_task(rc,ecm_stage1_stop_requested()!=0);return result;}
        if(!out->sigmas || !out->factors || !out->array_found)throw std::runtime_error("Stage1 completed without per-curve result identity");
        out->validate_curves(N);
        out->save_range=std::move(publication.range);
        phase=FailureSource::Publication;
        validate_saved_batch(out->save_range,opt,N,B1,*out);
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::vector<ecm_results::Stage1BatchRecord> records;
        std::string task_identity=ecm_edwards::hex(N)+"|"+std::to_string(base.b1)+(base.edwards?"ed":"mont"+std::to_string(base.param));
        for(uint32_t i=0;i<out->curves;++i)task_identity+="|"+std::to_string(out->sigmas[i]);
        const auto publication_task=std::to_string(ecm_edwards::hash(task_identity));
        const auto publication_when=ecm_results::utc_timestamp();
        for(uint32_t i=0;i<out->curves;++i) {
            auto r=base;r.publication_task=publication_task;r.sigma=out->sigmas[i];r.when=publication_when;
            if(out->array_found[i]!=ECM_NO_FACTOR_FOUND){r.stage=1;r.factors.push_back(ecm_results::decimal(out->factors[i]));}
            std::ostringstream extra;extra<<",\"stage1-engine\":{\"curve\":"<<i<<",\"target\":"<<ecm_results::quoted(ecm_stage1::execution_target_name(opt.execution.selection.target))
                <<",\"batch-seconds\":"<<seconds<<",\"coverage\":\"B1\"}";
            records.push_back({r,ecm_results::internal_json(r,N,0,deps.build_identity(),extra.str())});
        }
        ecm_results::publish_stage1_batch(opt.persistence.results_file,opt.persistence.submission_file,
                                        std::move(records),N,task_identity);
        if(opt.persistence.retain_completed_checkpoint)out->completed_checkpoint=path;
        session.close(!opt.persistence.retain_completed_checkpoint);
        ecm_ts_fprintf(stdout,"Stage1 results: internal=%s submission=%s\n",opt.persistence.results_file.c_str(),opt.persistence.submission_file.c_str());
        out->finish_task(out->ret,false);
        return result;
    } catch(const std::exception &e) {
        if(const auto *checkpoint=dynamic_cast<const ecm_stage1_checkpoint::Error *>(&e))
            phase=checkpoint->state==ecm_stage1_checkpoint::RestoreState::IoFailure?
                FailureSource::Persistence:FailureSource::Preparation;
        if(out->completion!=ExecutionResult::Completion::Failed)out->fail(phase);
        session.close(false);ecm_ts_fprintf(stderr,"ERROR: Stage1: %s\n",e.what());
        out->finish_task(ECM_ERROR,ecm_stage1_stop_requested()!=0);return result;
    }
}

} // namespace ecm_stage1
