#include "../../../../io/files.h"
// Standalone CPU Edwards continuation. This target's <ecm.h> is libecm's API.
#include <ecm.h>
#include "ecm/io/save.h"
#include "ecm/io/results.h"
#include "ecm/curve/edwards.h"
#include "ecm/platform/stop.h"
#include "../../../application/configuration/generated/ecm_config_generated.h"
#include "ecm/platform/program_version.h"
#include <chrono>
#include <charconv>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <optional>
#include <exception>
#include <sstream>
#include <omp.h>
#include "../../cpu/gwpoly.h"
#include "../../cpu/giant_range.h"
#include "../../cpu/runtime_tune.h"
#include "../../queue_paths.h"
#include "../../launch.h"
#include "../../../platform/compute_log.h"
#include "ecm/platform/fingerprint.h"
#include "../../cpu/thread_layout.h"
#include <fstream>

#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID ecm_program::version
#endif
#ifndef ECM_LIBECM_NTT_THRESHOLD
#define ECM_LIBECM_NTT_THRESHOLD 30
#endif
namespace {
using namespace ecm_edwards;
namespace fs=std::filesystem;
std::string quote(const std::string &s) {return ecm_results::quoted(s);}
std::string number(const mpz_t x) {return ecm_results::decimal(x);}
uint64_t decimal(const std::string& text) {
    uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("expected unsigned decimal integer: " + text);
    return value;
}
void set64(mpz_t x,uint64_t n) {mpz_import(x,1,1,sizeof n,0,0,&n);}
struct Options {
    std::string save,results="results_cpu_stage2.jsonl",submission,ini,backend="auto",user,computer,aid;
    std::vector<std::string> known;uint64_t b2=0,skip=0,count=UINT64_MAX,exponent=0,memory=512,threads=1;
    std::string worktodo,log,cpu_affinity;
    fs::path tune_profile;
    std::optional<ecm_cpu_tune::ThreadLayout> thread_layout;
    bool tune_runtime=true,round_b2=false;
    uint64_t round_limit=ecm_gwpoly::max_b2;
    uint64_t tune_min_repeats=2,tune_sample_limit=10;
    int log_level=-1;
    bool gui=false,once=false,explicit_results=false;
    bool dry=false,export_only=false,gw_selftest=false,gw_check=false;uint64_t d=0;uintptr_t stop_handle=0;int worker=1;
};
void check_tune_path(const Options &o,std::initializer_list<fs::path> resources,
                     std::initializer_list<fs::path> results={}) {
    if(o.tune_profile.empty())return;
    const auto canonical=[](const fs::path &path) {
        std::error_code error;
        const auto resolved=fs::weakly_canonical(fs::absolute(path),error);
        return error?fs::absolute(path).lexically_normal():resolved;
    };
    const auto profile=canonical(o.tune_profile).u8string();
    const fs::path files[]={fs::u8path(profile),fs::u8path(profile+".publish.tmp"),fs::u8path(profile+".publish.lock")};
    const auto equal=[](const std::string &a,const std::string &b) {
        return !ecm_io::file_identity_less(a,b) && !ecm_io::file_identity_less(b,a);
    };
    const auto conflict=[] {throw std::runtime_error("tune profile conflicts with a CPU Stage2 input or output");};
    for(const auto &path:resources) {
        if(path.empty())continue;
        const auto resource=canonical(path).u8string();
        for(const auto &file:files) {
            std::error_code error;
            const auto writer=file.u8string();
            if(fs::equivalent(file,path,error) || equal(writer,resource) ||
               equal(writer,resource+".publish.lock") || equal(writer,resource+".edit.lock") ||
               equal(writer,resource+".lock") || equal(writer,resource+".publish.tmp"))conflict();
        }
    }
    for(const auto &path:results) {
        if(path.empty())continue;
        const auto ledger=canonical(ecm_results::submission_ledger(path)).u8string();
        const auto legacy=canonical(ecm_results::legacy_submission_ledger(path)).u8string();
        for(auto dir=fs::u8path(profile);;) {
            if(equal(dir.u8string(),ledger) || equal(dir.u8string(),legacy))conflict();
            auto parent=dir.parent_path();
            if(parent.empty() || parent==dir)break;
            dir=std::move(parent);
        }
    }
}
Options options(int argc,char **argv,const ecm_stage2::LaunchConfig& launch) {
    Options o;
    o.worker=launch.worker;
    const auto& ini=launch.path;
    o.ini=ini.string();
    const auto settings=ecm_config::read_stage2(ecm_config::cli_entries(launch.lines,o.worker,true));
    o.threads=settings.cpu_threads;o.memory=settings.cpu_memory_mib;o.backend=settings.cpu_backend;
    o.tune_runtime=settings.tune_runtime;o.round_b2=settings.b2_round_chain;
    bool explicit_rounding=settings.has_b2_round_chain;
    if(settings.auto_b2 && settings.auto_max)o.round_limit=std::min(o.round_limit,settings.auto_max);
    o.tune_min_repeats=settings.tune_min_repeats;
    o.tune_sample_limit=settings.runtime_samples;
    const auto &tune_name=settings.tune_profile.empty()?settings.shared_tune_file:settings.tune_profile;
    if(!tune_name.empty())o.tune_profile=s2prod::absolute_from(fs::u8path(o.ini).parent_path(),tune_name);
    o.cpu_affinity=settings.shared_cpu_affinity;
    o.user=settings.result_user;o.computer=settings.result_computer;
    if(settings.has_submission && settings.submission.empty())throw std::runtime_error("empty submission path");
    const auto filename=settings.has_submission?settings.submission:ecm_config::worker_file(settings.submission,o.worker);
    o.submission=(ini.parent_path()/filename).string();
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--version") {
            std::cout << ecm_program::name << ' ' << ecm_program::version << " CPU Stage2 " << ECM_BUILD_ID << '\n';
            if (const char* error = ecm_gwpoly::load_error()) std::cout << error << '\n';
            else std::cout << ecm_gwpoly::identity() << " GWNUM=" << ecm_gwpoly::version()
                           << " CPU=" << ecm_gwpoly::cpu() << '\n';
            std::exit(0);
        }
        if(key=="--help" || key=="-h") {
            std::cout<<"CPU Edwards Stage2\n[--config FILE] [--worker N] [--worktodo FILE] [--once] [--stage2-stop-handle HANDLE]\n--save FILE --b2 DECIMAL [--backend auto|ntt|plain|gwpoly] [--memory-mib 512]\n"
                "[--threads 1] [--b2-round-chain 0|1] [--results INTERNAL.jsonl] [--submission results.json.txt] [--config FILE] [--worker N]\n"
                "[--skip N] [--curves N] [--dry-run] [--d D (gwpoly only)] [--gw-check] [--selftest-gwpoly]\n"
                "[--tune-runtime 0|1] [--tune-profile FILE] [--tune-min-repeats N] [--tune-runtime-samples N]\n"
                "--export-results --submission FILE regenerates owned submission output without arithmetic\n"
                "[--exponent P] [--known-factor DECIMAL (repeatable)] [--user NAME] [--computer NAME] [--aid HEX32]\n"
                "B2 limit: libecm=1000000000000, gwpoly=1000000000000000\n";
            std::cout<<"--version shows loaded GW build and CPU; auto selects available gwpoly or libecm\n";

            std::exit(0);
        }
        if(key=="--gui-events"){o.gui=true;continue;}
        if(key=="--once"){o.once=true;continue;}
        if(key=="--dry-run"){o.dry=true;continue;}
        if(key=="--export-results"){o.export_only=true;continue;}
        if(key=="--selftest-gwpoly"){o.gw_selftest=true;continue;}
        if(key=="--gw-check"){o.gw_check=true;continue;}
        if(++i>=argc)throw std::runtime_error("missing value for "+key);std::string value=argv[i];
        if(key=="--save")o.save=value;else if(key=="--results"){o.results=value;o.explicit_results=true;}
        else if(key=="--worktodo")o.worktodo=value;
        else if(key=="--log")o.log=value;
        else if(key=="--log-level")o.log_level=stage2_log::parse(value);
        else if(key=="--submission")o.submission=value;
        else if(key=="--config" || key=="--worker")continue;
        else if(key=="--backend")o.backend=value;else if(key=="--b2")o.b2=decimal(value);
        else if(key=="--memory-mib")o.memory=decimal(value);else if(key=="--threads")o.threads=decimal(value);
        else if(key=="--skip")o.skip=decimal(value);else if(key=="--curves")o.count=decimal(value);
        else if(key=="--b2-round-chain"){
            const auto mode=decimal(value);
            if(mode>1)throw std::runtime_error("b2-round-chain must be 0 or 1");
            o.round_b2=mode!=0;explicit_rounding=true;
        }
        else if(key=="--d")o.d=decimal(value);
        else if(key=="--tune-profile")o.tune_profile=value.empty()?fs::path{}:s2prod::absolute_from(fs::current_path(),value);
        else if(key=="--tune-runtime"){
            const auto mode=decimal(value);
            if(mode>1)throw std::runtime_error("tune-runtime must be 0 or 1");
            o.tune_runtime=mode!=0;
        }
        else if(key=="--tune-min-repeats")o.tune_min_repeats=decimal(value);
        else if(key=="--tune-runtime-samples")o.tune_sample_limit=decimal(value);
        else if(key=="--exponent")o.exponent=decimal(value);else if(key=="--known-factor")o.known.push_back(value);
        else if(key=="--user")o.user=value;else if(key=="--computer")o.computer=value;else if(key=="--aid")o.aid=value;
        else if((key=="--stop-handle" || key=="--stage2-stop-handle"))o.stop_handle=decimal(value);
        else throw std::runtime_error("unknown option: "+key);
    }
    uint64_t max_b2=1000000000000ull;
    if ((o.backend=="gwpoly" || o.gw_selftest) && ecm_gwpoly::load_error())
        throw std::runtime_error(ecm_gwpoly::load_error());
    if(o.backend=="gwpoly" || (o.backend=="auto" && !ecm_gwpoly::load_error()))max_b2=ecm_gwpoly::max_b2;
    if((!o.export_only && !o.gw_selftest && (!o.save.empty() && !o.b2)) || o.b2>max_b2 || !o.count || !o.threads || o.threads>1024 || !o.memory || o.memory>1048576 ||
       (o.backend!="auto" && o.backend!="ntt" && o.backend!="plain" && o.backend!="gwpoly"))throw std::runtime_error("invalid options; see --help");
    if(o.round_b2 && (o.backend=="ntt" || o.backend=="plain" || ecm_gwpoly::stage2_abi()<4)){
        if(explicit_rounding && !o.export_only && !o.gw_selftest)
            throw std::runtime_error("CPU B2 rounding requires gwpoly with Stage2 ABI 4");
        o.round_b2=false;
    }
    if((o.d || o.gw_check) && o.backend!="gwpoly")throw std::runtime_error("--d/--gw-check require --backend gwpoly");
    if(!o.tune_sample_limit || o.tune_sample_limit>1000 || !o.tune_min_repeats || o.tune_min_repeats>o.tune_sample_limit)
        throw std::runtime_error("tune repeats and sample limit must be in 1..1000, with repeats <= limit");
    if(!o.aid.empty() && (o.aid.size()!=32 || o.aid.find_first_not_of("0123456789abcdefABCDEF")!=o.aid.npos))throw std::runtime_error("AID must contain 32 hex digits");
    if(o.exponent>1000000000 || (!o.known.empty() && !o.exponent))throw std::runtime_error("known factors require --exponent (max 1000000000)");
    if(!o.save.empty() && !o.export_only && !o.gw_selftest && std::filesystem::absolute(o.save).lexically_normal()==std::filesystem::absolute(o.results).lexically_normal())throw std::runtime_error("results must not overwrite save");
    std::error_code error;
    if(!o.save.empty() && std::filesystem::equivalent(o.save,o.results,error))throw std::runtime_error("results alias the input save");
    if(o.submission.empty())throw std::runtime_error("empty submission path");
    const auto same=[](const std::string &a,const std::string &b){if(a.empty() || b.empty())return false;std::error_code e;return std::filesystem::absolute(a).lexically_normal()==std::filesystem::absolute(b).lexically_normal() || std::filesystem::equivalent(a,b,e);};
    if(same(o.save,o.submission) || same(o.results,o.submission))throw std::runtime_error("submission path conflicts with save/internal results");
    if(!o.save.empty() && !o.worktodo.empty())throw std::runtime_error("--worktodo conflicts with --save");
    if(o.save.empty() && (o.skip || o.count!=UINT64_MAX || o.exponent || !o.known.empty() || !o.aid.empty()))
        throw std::runtime_error("queue curve selection and target identity come from worktodo");
    if(o.d || o.dry || o.gw_check || o.export_only || o.gw_selftest ||
       o.backend=="ntt" || o.backend=="plain" || ecm_gwpoly::load_error())o.tune_profile.clear();
    if(o.log_level<0)o.log_level=settings.log_level>=0?settings.log_level:stage2_log::phases;
    check_tune_path(o,{fs::u8path(o.ini),fs::u8path(o.save),fs::u8path(o.results),
                      fs::u8path(o.submission),fs::u8path(o.log),s2prod::executable()},
                    {fs::u8path(o.results),fs::u8path(o.submission)});
    return o;
}
void validate_target(const Options &o,const SavedCurve &r) {
    if(o.b2<=r.b1)throw std::runtime_error("B2 must exceed every selected B1");
    if(!o.exponent)return;
    Integer m,n,f;mpz_set_ui(m.v,1);mpz_mul_2exp(m.v,m.v,static_cast<mp_bitcnt_t>(o.exponent));mpz_sub_ui(m.v,m.v,1);
    for(const auto &v:o.known) {
        if(v.find_first_not_of("0123456789")!=v.npos || mpz_set_str(f.v,v.c_str(),10) || mpz_cmp_ui(f.v,1)<=0 || !mpz_divisible_p(m.v,f.v))throw std::runtime_error("invalid known factor for target");
        mpz_divexact(m.v,m.v,f.v);
    }
    mpz_set_str(n.v,r.n.c_str(),16);
    if(mpz_cmp(m.v,n.v))throw std::runtime_error("exponent/known-factors do not match saved N");
}
int stopped(){
    s2prod::poll_inherited_stop();
    return s2prod::stop_requests || ecm_stage1_stop_requested();
}
struct Params {ecm_params p;Params(){ecm_init(p);}~Params(){ecm_clear(p);}};
struct Details {uint64_t d=0,poly_size=0,fft_length=0,checked_values=0;std::string fft_description,polymult_isa;double init=0,main=0,maxerr=0;};
std::string result(const Options &o,const SavedCurve &r,const mpz_t n,const mpz_t f,int stage,double seconds,const Details &details,uint64_t requested_b2) {
    ecm_results::Submission common;common.exponent=o.exponent;common.sigma=r.sigma;common.b1=r.b1;common.b2=o.b2;
    common.stage=stage;common.known=o.known;common.user=o.user;common.computer=o.computer;common.aid=o.aid;
    if(stage)common.factors.push_back(number(f));
    if(stage!=1){common.d=details.d;common.poly_size=details.poly_size;}
    auto text=ecm_results::internal_json(common,n,details.fft_length,ECM_BUILD_ID);text.resize(text.size()-2);
    std::ostringstream out;out<<text
       <<",\"stage2-engine\":{\"name\":"<<quote(o.backend=="gwpoly"?"gwnum-polymult":"libecm")<<",\"version\":"<<quote(
           o.backend=="gwpoly"?ecm_gwpoly::version():
           ecm_version())<<",\"backend\":"<<quote(o.backend)
       <<",\"seconds\":"<<std::setprecision(9)<<seconds<<",\"save-line\":"<<r.line
       <<",\"coverage\":"<<quote(o.b2>requested_b2?"rounded-B2":"requested-B2");
    if(o.round_b2)out<<",\"requested-B2\":"<<requested_b2<<",\"actual-B2\":"<<o.b2;
    if(details.fft_length)out<<",\"gw-build\":"<<quote(ecm_gwpoly::identity())<<",\"fft-description\":"<<quote(details.fft_description)<<",\"polymult-isa\":"<<quote(details.polymult_isa)<<",\"fft-length\":"<<details.fft_length<<",\"init-seconds\":"<<details.init
        <<",\"main-seconds\":"<<details.main<<",\"max-roundoff\":"<<details.maxerr<<",\"checked-values\":"<<details.checked_values;
    out<<"}}";
    return out.str();
}

// In queue mode the first stop drains an admitted curve, matching the GPU owner.
int abort_queue_curve(){s2prod::poll_inherited_stop();return s2prod::stop_requests>1;}
struct TuneObservation {
    ecm_cpu_tune::Key key;
    ecm_cpu_tune::Candidate plan;
    double seconds = 0;
};
// Keep the most recent exact geometry probe. Scope changes are common between
// worktodo assignments and uncommon between curves in the same save.
struct TunePlans {
    std::string modulus;
    uint64_t b1=0,b2=0,exponent=0;
    std::vector<ecm_cpu_tune::Candidate> candidates;
    const std::vector<ecm_cpu_tune::Candidate>& get(const SavedCurve &r,mpz_srcptr n,uint64_t bound,
                                                   const ecm_gwpoly::Options &gw,const Options &options) {
        if(modulus!=r.n || b1!=r.b1 || b2!=bound || exponent!=gw.exponent) {
            const auto plans=ecm_gwpoly::tune_candidates(n,r.b1,bound,gw);
            std::vector<ecm_cpu_tune::Candidate> next;
            for(const auto &p:plans) {
                const auto actual=options.round_b2?ecm_gwpoly::full_block_b2(r.b1,bound,p.d,p.poly_size,options.round_limit):bound;
                if(actual)next.push_back({p.d,p.poly_size,p.fft_length,p.fft_exponent,
                    p.max_exponent,p.f_tree_splits,p.planned_mib,p.fft_layout,p.polymult_isa,actual});
            }
            modulus=r.n;
            candidates=std::move(next);
            b1=r.b1;b2=bound;exponent=gw.exponent;
        }
        return candidates;
    }
    void reject(uint64_t d) {
        candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](const auto &p){return p.d==d;}),
                         candidates.end());
    }
};
struct CurveProgress {
    const s2prod::QueueReport &report;
    const SavedCurve &curve;
    uint64_t b2;
    size_t bits;
    bool batches;
    double phase_start=0;
    std::exception_ptr error;

    static void receive(void *context,const ecm_gw_stage2_progress *event) noexcept {
        auto &stream=*static_cast<CurveProgress*>(context);
        if(stream.error)return;
        try { stream.emit(*event); }
        catch(...) { stream.error=std::current_exception(); }
    }

    void emit(const ecm_gw_stage2_progress &e) {
        std::ostringstream out;
        if(e.kind==ECM_GW_BATCH) {
            if(!batches || !e.batch || e.batch>e.batches)return;
            const double time=std::max(0.,e.elapsed-phase_start);
            out<<"batched_progress: batch="<<e.batch<<'/'<<e.batches<<" t="<<time
               <<" s left="<<time*(e.batches-e.batch)/e.batch<<" s backend=gwpoly";
        } else {
            const char *phase=nullptr;
            switch(e.kind) {
            case ECM_GW_PLAN: phase="Plan Stage2"; break;
            case ECM_GW_BABY: phase="Generate and normalize baby points"; break;
            case ECM_GW_FTREE: phase="Build baby-point F tree"; break;
            case ECM_GW_INVERSE: phase="Prepare polynomial inverse"; break;
            case ECM_GW_GIANT: phase="Giant points, G tree and fold"; break;
            case ECM_GW_DESCENT: phase="Normalize fold and descend F tree"; break;
            case ECM_GW_GCD: phase="Accumulate leaf product and GCD"; break;
            case ECM_GW_REPLAY: phase="Plan Stage2 (full-range replay)"; break;
            }
            if(!phase)return;
            if(e.kind==ECM_GW_BABY) {
                out<<"stage2_real: B1="<<curve.b1<<" B2="<<b2<<" D="<<e.d<<" P="<<e.poly_size
                   <<" N_bits="<<bits<<" backend=gwpoly";
                report(out.str(),false);
                out.str("");
            }
            out<<"stage2_phase: "<<phase<<" previous="<<std::max(0.,e.elapsed-phase_start)
               <<" s elapsed="<<e.elapsed<<" s";
            phase_start=e.elapsed;
        }
        report(out.str(),false);
    }
};

void persist_tune(const Options &o,const std::optional<TuneObservation> &pending) {
    if(!pending)return;
    try {
        ecm_cpu_tune::observe(o.tune_profile,pending->key,pending->plan,pending->seconds,o.tune_sample_limit);
    } catch(const std::exception &error) {
        std::cerr<<"gwpoly-tune: cannot save observation: "<<error.what()<<'\n';
    }
}
bool calculate(const Options &o,const SavedCurve &r,const std::string *prepared_a24,
               const std::string &publication_task,uint64_t order,ecm_results::Submission &report,
               std::string &json,double &elapsed,TunePlans &tune,std::optional<TuneObservation> &pending,
               const s2prod::QueueReport &log) {
    const auto cancel=prepared_a24?abort_queue_curve:stopped;
    if(cancel())return false;
    Integer n,x,d,px,py,a24,f;
    mpz_set_str(n.v,r.n.c_str(),16);mpz_set_str(x.v,r.x.c_str(),16);
    auto selected=o;
    if(selected.backend=="auto"){
        selected.backend=ecm_gwpoly::load_error() ? (mpz_size(n.v)<=ECM_LIBECM_NTT_THRESHOLD?"ntt":"plain") : "gwpoly";

    }
    const bool verbose=!prepared_a24 || stage2_log::enabled(stage2_log::curve);
    if(verbose) {
        std::cout << "selected=" << selected.backend;
        if(selected.backend=="gwpoly")std::cout<<" "<<ecm_gwpoly::identity()<<" GWNUM="<<ecm_gwpoly::version()<<" CPU="<<ecm_gwpoly::cpu();
        std::cout<<std::endl;
    }
    int stage=0;Details details;const auto begin=std::chrono::steady_clock::now();
    if(r.factor){mpz_set(f.v,x.v);stage=1;}
    else {
        auto setup=Setup::Ready;
        if(prepared_a24)mpz_set_str(a24.v,prepared_a24->c_str(),16);
        else setup=construct(d.v,px.v,py.v,a24.v,f.v,r.sigma,n.v);
        if(setup==Setup::Degenerate)throw std::runtime_error("degenerate saved curve");
        if(setup==Setup::Factor)stage=1;
        else {
            if(selected.backend=="gwpoly"){
                ecm_gwpoly::Options gw;gw.threads=static_cast<unsigned>(o.threads);gw.memory_mib=o.memory;
                gw.exponent=o.exponent;gw.d=o.d;gw.stop=cancel;gw.check=o.gw_check;
                CurveProgress progress{log,r,o.b2,mpz_sizeinbase(n.v,2),o.gui || stage2_log::enabled(stage2_log::batches)};
                if(o.gui || stage2_log::enabled(stage2_log::phases)) {
                    gw.progress=CurveProgress::receive;
                    gw.progress_context=&progress;
                }
                ecm_gwpoly::Stats stats{};
                std::optional<TuneObservation> observation;
                const auto apply_plan=[&](const ecm_cpu_tune::Candidate &plan) {
                    gw.d=plan.d;selected.b2=plan.actual_b2;progress.b2=selected.b2;
                    if(o.round_b2)log("gwpoly-bounds: requested_B2="+std::to_string(o.b2)+
                        " actual_B2="+std::to_string(selected.b2)+" D="+std::to_string(gw.d)+
                        " P="+std::to_string(plan.poly_size),false);
                };
                if(o.round_b2 && tune.get(r,n.v,o.b2,gw,o).empty())
                    throw std::runtime_error("No admitted whole G block endpoint within CPU B2 limit "+std::to_string(o.round_limit));
                std::error_code tune_error;
                if(ecm_gwpoly::stage2_abi()>=4 && o.thread_layout && !o.tune_profile.empty() &&
                   (o.tune_runtime || fs::is_regular_file(o.tune_profile,tune_error))) {
                    try {
                        const auto &plans=tune.get(r,n.v,o.b2,gw,o);
                        if(!plans.empty()) {
                            ecm_cpu_tune::Key key{ecm_cpu_tune::cpu_identity(ecm_gwpoly::cpu()),ecm_gwpoly::version(),*o.thread_layout,
                                gw.exponent != 0,gw.exponent ? gw.exponent : mpz_sizeinbase(n.v,2),r.b1,o.b2};
                            const auto d=ecm_cpu_tune::select(o.tune_profile,key,plans,o.tune_min_repeats,o.tune_runtime);
                            const auto plan=std::find_if(plans.begin(),plans.end(),[&](const auto &p){return p.d==d;});
                            if(plan!=plans.end()) {
                                key.b2=plan->actual_b2;
                                observation=TuneObservation{std::move(key),*plan};
                                apply_plan(*plan);
                                if(verbose)log("gwpoly-tune: D="+std::to_string(d)+" P="+std::to_string(plan->poly_size)+
                                    " FFT="+std::to_string(plan->fft_length)+(o.tune_runtime?" runtime":" read-only"),false);
                            }
                        }
                    } catch(const std::exception &error) {
                        if(cancel())return false;
                        std::cerr<<"gwpoly-tune: using automatic D: "<<error.what()<<'\n';
                    }
                }
                if(cancel())return false;
                if(o.round_b2 && !observation)apply_plan(tune.candidates.front());
                const auto trial_start=std::chrono::steady_clock::now();
                try {
                    if(ecm_gwpoly::stage2(f.v,n.v,x.v,a24.v,r.b1,selected.b2,gw,stats))stage=2;
                } catch(const std::exception &error) {
                    if(cancel())return false;
                    if(!observation)throw;
                    std::cerr<<"gwpoly-tune: D="<<gw.d<<" failed, retrying automatic D: "<<error.what()<<'\n';
                    tune.reject(gw.d);
                    observation.reset();
                    if(o.round_b2){
                        if(tune.candidates.empty())throw;
                        apply_plan(tune.candidates.front());
                    }else gw.d=0;
                    stats={};
                    try {if(ecm_gwpoly::stage2(f.v,n.v,x.v,a24.v,r.b1,selected.b2,gw,stats))stage=2;}
                    catch(...){if(cancel())return false;throw;}
                }
                if(progress.error)std::rethrow_exception(progress.error);
                if(o.round_b2 && !stage && (stats.d!=gw.d || !stats.poly_size ||
                   stats.giants%stats.poly_size || stats.giants/stats.poly_size!=stats.blocks))
                    throw std::runtime_error("GWNUM returned an incomplete rounded G block geometry");
                const double trial_seconds=std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-trial_start).count();
                if(o.tune_runtime && !stage && observation &&
                   stats.d==observation->plan.d && stats.poly_size==observation->plan.poly_size &&
                   stats.fft_length==observation->plan.fft_length &&
                   stats.fft_description==observation->plan.fft_layout &&
                   stats.polymult_isa==observation->plan.isa &&
                   stats.estimated_memory_mib==observation->plan.planned_mib) {
                    observation->seconds=trial_seconds;
                    pending=std::move(observation);
                }
                details.fft_description=stats.fft_description;details.polymult_isa=stats.polymult_isa;
                details.d=stats.d;details.poly_size=stats.poly_size;details.fft_length=stats.fft_length;
                details.init=stats.init_seconds;details.main=stats.main_seconds;details.maxerr=stats.max_roundoff;details.checked_values=stats.checked_values;
                if(verbose)std::cout<<"gwpoly: D="<<stats.d<<" P="<<stats.poly_size<<" fft_length="<<stats.fft_length
                         <<" fft=\""<<stats.fft_description<<"\" polymult_isa="<<stats.polymult_isa<<" blocks="<<stats.blocks<<" giants="<<stats.giants<<" polymults="<<stats.polymults
                         <<" init="<<stats.init_seconds<<" main="<<stats.main_seconds<<" maxerr="<<stats.max_roundoff
                         <<" checked_values="<<stats.checked_values<<" planned_mib="<<stats.estimated_memory_mib
                         <<" points="<<stats.point_seconds<<" trees="<<stats.tree_seconds<<" folds="<<stats.fold_seconds
                         <<" descent="<<stats.descent_seconds<<std::endl;
            }else {
                Params params;auto p=params.p;p->sigma_is_A=1;
                mpz_mul_ui(p->sigma,a24.v,4);mpz_sub_ui(p->sigma,p->sigma,2);mpz_mod(p->sigma,p->sigma,n.v);
                mpz_set(p->x,x.v);p->B1done=static_cast<double>(r.b1);set64(p->B2,o.b2);
                p->maxmem=static_cast<double>(o.memory)*1048576.0;p->use_ntt=selected.backend=="ntt";p->S=1;p->verbose=verbose?2:0;p->stop_asap=cancel;
                const int rc=ecm_factor(f.v,n.v,static_cast<double>(r.b1),p);
                if(cancel())return false; // libecm may return NF on interruption.
                if(rc<0)throw std::runtime_error("libecm failed; no result published for current curve");
                if((rc==0 || rc==ECM_FACTOR_FOUND_STEP2) &&
                   (p->B1done!=static_cast<double>(r.b1) || mpz_cmp(p->x,x.v)))
                    throw std::runtime_error("libecm changed the completed Stage1 point");
                if(rc>0)stage=rc==ECM_FACTOR_FOUND_STEP2?2:1;
            }
        }
    }
    if(stage && (mpz_cmp_ui(f.v,1)<=0 || mpz_cmp(f.v,n.v)>=0 || !mpz_divisible_p(n.v,f.v)))throw std::runtime_error("saturated or invalid factor; no NF published");
    elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    json=result(selected,r,n.v,f.v,stage,elapsed,details,o.b2);
    report.exponent=o.exponent;report.sigma=r.sigma;report.b1=r.b1;report.b2=selected.b2;
    report.publication_task=publication_task;report.stage=stage;report.known=o.known;report.user=o.user;report.computer=o.computer;report.aid=o.aid;
    report.publication_order=order;
    report.save_name=std::filesystem::path(o.save).stem().u8string();
    report.d=details.d;report.poly_size=details.poly_size;report.stage2_fft_length=details.fft_length;
    if(stage)report.factors.push_back(number(f.v));
    return true;
}

int run_queue(const Options &cpu,const std::vector<ecm_config::TomlLine> &lines) {
    s2prod::Options o;
    o.worker=cpu.worker;o.worktodo=cpu.worktodo;o.b2=cpu.b2;o.d=cpu.d;
    o.results=cpu.explicit_results?cpu.results:"";o.submission=cpu.submission;o.log=cpu.log;
    o.dry=cpu.dry;o.once=cpu.once;o.gui_events=cpu.gui;o.parallel=s2prod::ParallelMode::serial;
    const auto settings=ecm_config::read_stage2(ecm_config::cli_entries(lines,cpu.worker,true));
    EcmQueueConfig cfg;
    std::vector<const char*> inherited={"finished"};
    if(settings.save_dir.empty())inherited.push_back("tmp_dir");
    ecm_queue_config_project(lines,cpu.ini,cpu.worker,cfg,inherited);
    const bool shared=ecm_config::toml_document_version(lines)>=3;
    const auto paths=s2prod::task_paths(o,s2prod::fs::current_path(),cpu.ini,settings,cfg,shared);
    check_tune_path(cpu,{paths.worktodo,paths.finished,paths.results,paths.log,paths.legacy_progress,
                        fs::path(paths.worktodo.string()+".worker"+std::to_string(cpu.worker)+".stage2.lock")},
                    {paths.results,fs::u8path(cpu.submission)});
    TunePlans tune;
    if(cpu.stop_handle) {
        s2prod::inherited_stop_event=reinterpret_cast<HANDLE>(cpu.stop_handle);
        if(WaitForSingleObject(s2prod::inherited_stop_event,0)==WAIT_FAILED)
            throw std::runtime_error("invalid inherited Stage2 stop event");
    }
    if(stage2_log::enabled(stage2_log::curve))std::cout<<"CPU Edwards Stage2 backend="<<cpu.backend<<" threads="<<cpu.threads
             <<" memory_mib="<<cpu.memory<<" libecm="<<ecm_version()<<std::endl;
    s2prod::TaskEngine engine;
    engine.name="CPU Stage2";
    engine.validate=[&](const s2prod::Record &r,uint64_t b2) {
        if(!r.edwards)throw std::runtime_error("CPU Stage2 requires Edwards saves; set device = \"gpu\" for Montgomery");
        const bool gw=cpu.backend=="gwpoly" || (cpu.backend=="auto" && !ecm_gwpoly::load_error());
        if(b2>(gw?ecm_gwpoly::max_b2:1000000000000ull))
            throw std::runtime_error("requested B2 exceeds the CPU backend limit");
    };
    engine.run=[&](const s2prod::CurveQueue &queue,s2prod::Options &task,uint64_t b2,uint64_t) {
        check_tune_path(cpu,{queue.save,queue.progress,queue.results,queue.log},{queue.results});
        auto selected=cpu;
        selected.save=queue.save.string();selected.b2=b2;selected.exponent=task.submission_exponent;
        selected.known=task.submission_known;selected.aid=task.submission_aid;
        const bool events=task.gui_events || stage2_log::enabled(stage2_log::curve);
        auto &state=queue.state;
        std::ifstream save(queue.save,std::ios::binary);
        while(state.done<queue.plan.size()) {
            if(stopped())return false;
            const auto position=state.done;
            const auto &record=queue.plan[position];
            // The queue already verified the checksum and reconstructed A24 during planning.
            // Bind that snapshot to the current on-disk record before admitting the curve.
            save.clear();save.seekg(record.offset);
            std::string line;
            if(!std::getline(save,line) || s2prod::fingerprint(line)!=record.hash)
                throw std::runtime_error("save changed after planning");
            const auto ordinal=std::to_string(position+1)+"/"+std::to_string(queue.plan.size());
            const auto identity=" record="+std::to_string(record.index)+" sigma="+std::to_string(record.sigma);
            if(events)queue.report("curve_start: "+ordinal+identity+" B1="+std::to_string(record.b1)+" curve_model=ed",false);
            Integer n,x,f;
            mpz_set_str(n.v,record.n.c_str(),16);mpz_set_str(x.v,record.x.c_str(),16);
            mpz_gcd(f.v,x.v,n.v);
            const bool hit=mpz_cmp_ui(f.v,1)>0;
            SavedCurve row{record.n,hit?s2prod::number(f.v):record.x,record.sigma,record.b1,record.index,hit};
            std::string json;ecm_results::Submission report;double elapsed=0;
            std::optional<TuneObservation> pending;
            if(!calculate(selected,row,&record.a24,state.publication_task,record.index,report,json,elapsed,tune,pending,queue.report))return false;
            const auto outcome=s2prod::checkpoint_outcome(record,report.b2,ecm_results::submission_json(report,n.v));
            if(state.reserved.empty())state.reserved.push_back({position});
            queue.commit(position,outcome);
            ++queue.completed;
            if(!queue.results.empty())ecm_results::append_text_record(queue.results,json);
            persist_tune(selected,pending);
            if(events)queue.report("stage2_result:"+identity+" B1="+std::to_string(record.b1)+" curve_model=ed seconds="+
                std::to_string(elapsed)+" "+json,false);
            if(events)queue.report("curve_done: "+ordinal+identity+" completed="+std::to_string(state.done)+
                " wall="+std::to_string(elapsed)+" active="+std::to_string(elapsed),false);
        }
        return true;
    };
    return s2prod::run_task_queue(engine,o,paths,{cfg.b2_ratio,cfg.worktodo_priority,cfg.worktodo_skip,cfg.finished,shared},
                                  {settings.b2,0,0});
}
}
int ecm_stage2::run_cpu_stage2(int argc,char **argv,const ecm_stage2::LaunchConfig* launch) {
    try {
        for(int i=1;i<argc;++i)if(std::string(argv[i])=="--gui-queue-protocol"){
            std::cout<<"ECM_QUEUE_OWNER=1\n";return 0;
        }
        LaunchConfig standalone;
        if (!launch) {
            std::string config;
            int worker = 1;
            for(int i=1;i<argc;++i) {
                const std::string key=argv[i];
                if(key!="--config" && key!="--worker")continue;
                if(++i>=argc)throw std::runtime_error("missing value for "+key);
                if(key=="--config")config=argv[i];
                else worker=launch_worker(argv[i]);
            }
            standalone=read_launch_config(config,worker,s2prod::executable());
            launch=&standalone;
        }
        auto o=options(argc,argv,*launch);
        stage2_log::level=o.log_level;
        if((o.backend=="gwpoly" || o.backend=="auto") && !ecm_gwpoly::load_error() && !o.dry &&
           !o.export_only && !o.gw_selftest) {
            const auto abi=ecm_gwpoly::stage2_abi();
            if(abi<4 && !o.tune_profile.empty())
                std::cerr<<"gwpoly-tune: gwnum.dll Stage2 ABI "<<abi
                         <<" cannot tune; rebuild the DLL with -Stage2 (ABI 4).\n";
            if(abi<3 && (o.gui || stage2_log::enabled(stage2_log::phases)))
                std::cerr<<"gwpoly-progress: gwnum.dll Stage2 ABI "<<abi
                         <<" cannot stream progress; rebuild the DLL with -Stage2 (ABI 3).\n";
        }
        if (!o.export_only && !o.dry)
            o.thread_layout=ecm_cpu_tune::bind_worker(o.cpu_affinity,o.threads);
        if(o.gw_selftest){ecm_gwpoly::selftest(static_cast<unsigned>(o.threads));return 0;}
        if(o.export_only) {
            if(!ecm_results::has_result_ledger(o.submission))throw std::runtime_error("no owned submission ledger");
            ecm_results::seal_submissions(o.submission,2,{},true);std::cout<<"submission output regenerated\n";return 0;
        }
        omp_set_num_threads(static_cast<int>(o.threads));
        if(o.save.empty())return run_queue(o,launch->lines);
        auto rows=ecm_edwards::read_saved_curves(o.save);
        if(o.skip>=rows.size())throw std::runtime_error("skip exceeds saved curve count");
        const auto end=o.skip+std::min<uint64_t>(o.count,rows.size()-o.skip);
        // Validate the entire selected batch before any computation or publication.
        for(uint64_t i=o.skip;i<end;++i)validate_target(o,rows[i]);
        if(o.dry){std::cout<<"validated "<<end-o.skip<<" curves; B2="<<o.b2<<'\n';return 0;}
        std::string task_key=std::to_string(o.b2);
        for(uint64_t i=o.skip;i<end;++i)task_key+="|"+rows[i].n+"|"+std::to_string(rows[i].sigma)+"|"+std::to_string(rows[i].b1);
        const auto publication_task=std::to_string(ecm_io::fingerprint(task_key));
        ecm_stage1_install_stop_handler();if(o.stop_handle && !ecm_stage1_set_stop_handle(o.stop_handle))throw std::runtime_error("invalid stop handle");
        std::cout<<"CPU Edwards Stage2 backend="<<o.backend<<" threads="<<o.threads<<" memory_mib="<<o.memory<<" libecm="<<ecm_version()<<std::endl;
        TunePlans tune;
        const s2prod::QueueReport log=[](const std::string &message,bool error) {
            (error?std::cerr:std::cout)<<message<<std::endl;
        };
        for(uint64_t i=o.skip;i<end;++i) {
            ecm_results::check_stage2_outputs(o.results,o.submission,i>o.skip);
            const auto &r=rows[i];
            Integer n;mpz_set_str(n.v,r.n.c_str(),16);
            std::string json;ecm_results::Submission report;double elapsed=0;
            std::optional<TuneObservation> pending;
            if(!calculate(o,r,nullptr,publication_task,i+1,report,json,elapsed,tune,pending,log))return 130;
            ecm_results::check_stage2_outputs(o.results,o.submission,i>o.skip);
            ecm_results::publish_submission(o.submission,report,n.v);
            ecm_results::append_text_record(o.results,json);
            persist_tune(o,pending);
            std::cout<<"completed save-line="<<r.line<<" sigma="<<r.sigma<<" status="<<(report.stage?"F":"NF")<<" seconds="<<elapsed<<std::endl;
        }
        ecm_results::check_stage2_outputs(o.results,o.submission,end>o.skip);
        ecm_results::seal_submissions(o.submission,2,publication_task,true);
        return 0;
    }catch(const std::exception &e){std::cerr<<"CPU Stage2: "<<e.what()<<'\n';try{s2prod::append_fatal_log(e.what());}catch(...){}return 1;}
}

#ifndef ECM_STAGE2_COMBINED
// The GUI also supports existing CPU-only executables.
extern "C" __declspec(dllexport) void ecm_cpu_stage2_worker() {}

int main(int argc,char **argv) {
    return ecm_stage2::run_cpu_stage2(argc,argv);
}
#endif
