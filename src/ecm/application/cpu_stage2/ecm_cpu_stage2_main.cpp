#include "io/files.h"
// Standalone CPU Edwards continuation. This target's <ecm.h> is libecm's API.
#include <ecm.h>
#include "ecm/io/save.h"
#include "ecm/io/results.h"
#include "ecm/curve/edwards.h"
#include "ecm/platform/stop.h"
#include "../configuration/generated/ecm_config_generated.h"
#include <chrono>
#include <charconv>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <omp.h>
#ifdef ECM_CPU_HAVE_GWPOLY
#include "../../stage2/cpu/gwpoly.h"
#endif

#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_LIBECM_NTT_THRESHOLD
#define ECM_LIBECM_NTT_THRESHOLD 30
#endif
namespace {
using namespace ecm_edwards;
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
    bool dry=false,export_only=false,gw_selftest=false,gw_check=false;uint64_t d=0;uintptr_t stop_handle=0;int worker=1;
};
Options options(int argc,char **argv) {
    Options o;
    // Read local identity first so explicit CLI values, including empty values,
    // always override the selected global/worker TOML projection.
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if((key=="--config" || key=="--worker") && i+1<argc) {
            const std::string value=argv[++i];
            if(key=="--worker"){const auto n=decimal(value);if(!n || n>1000000)throw std::runtime_error("invalid worker");o.worker=int(n);}
            else o.ini=value;
        }
    }
    const auto ini=o.ini.empty()?std::filesystem::path(argv[0]).parent_path()/"ecm.toml":std::filesystem::path(o.ini);
    std::vector<ecm_config::TomlLine> lines;
    std::vector<std::string> ignored_gui;
    const bool read=ecm_config::read_toml(ini.string(),lines,ecm_config::TomlReadMode::Computation,&ignored_gui);
    for(const auto &key:ignored_gui)std::cerr<<"WARNING: ignoring unknown GUI-only TOML key: "<<key<<'\n';
    if(!read && (!o.ini.empty() || std::filesystem::exists(ini)))throw std::runtime_error("cannot read local TOML");
    const auto settings=ecm_config::read_stage2(ecm_config::cli_entries(lines,o.worker,true));
    o.user=settings.result_user;o.computer=settings.result_computer;
    if(settings.has_submission && settings.submission.empty())throw std::runtime_error("empty submission path");
    const auto filename=settings.has_submission?settings.submission:ecm_config::worker_file(settings.submission,o.worker);
    o.submission=(ini.parent_path()/filename).string();
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {
            std::cout<<"CPU Edwards Stage2\n--save FILE --b2 DECIMAL [--backend auto|ntt|plain|gwpoly] [--memory-mib 512]\n"
                "[--threads 1] [--results INTERNAL.jsonl] [--submission results.json.txt] [--config FILE] [--worker N]\n"
                "[--skip N] [--curves N] [--dry-run] [--d D (gwpoly only)] [--gw-check] [--selftest-gwpoly]\n"
                "--export-results --submission FILE regenerates owned submission output without arithmetic\n"
                "[--exponent P] [--known-factor DECIMAL (repeatable)] [--user NAME] [--computer NAME] [--aid HEX32]\n"
                "B2 limit: libecm=1000000000000, gwpoly=1000000000000000\n";
#ifdef ECM_CPU_HAVE_GWPOLY
            std::cout<<"auto selects gwpoly\n";
#else
            std::cout<<"auto selects libecm NTT/plain; gwpoly requires -GwPoly build\n";
#endif
            std::exit(0);
        }
        if(key=="--dry-run"){o.dry=true;continue;}
        if(key=="--export-results"){o.export_only=true;continue;}
        if(key=="--selftest-gwpoly"){o.gw_selftest=true;continue;}
        if(key=="--gw-check"){o.gw_check=true;continue;}
        if(++i>=argc)throw std::runtime_error("missing value for "+key);std::string value=argv[i];
        if(key=="--save")o.save=value;else if(key=="--results")o.results=value;
        else if(key=="--submission")o.submission=value;
        else if(key=="--config" || key=="--worker")continue;
        else if(key=="--backend")o.backend=value;else if(key=="--b2")o.b2=decimal(value);
        else if(key=="--memory-mib")o.memory=decimal(value);else if(key=="--threads")o.threads=decimal(value);
        else if(key=="--skip")o.skip=decimal(value);else if(key=="--curves")o.count=decimal(value);
        else if(key=="--d")o.d=decimal(value);
        else if(key=="--exponent")o.exponent=decimal(value);else if(key=="--known-factor")o.known.push_back(value);
        else if(key=="--user")o.user=value;else if(key=="--computer")o.computer=value;else if(key=="--aid")o.aid=value;
        else if(key=="--stop-handle")o.stop_handle=decimal(value);
        else throw std::runtime_error("unknown option: "+key);
    }
    uint64_t max_b2=1000000000000ull;
#ifdef ECM_CPU_HAVE_GWPOLY
    if(o.backend=="gwpoly" || o.backend=="auto")max_b2=ecm_gwpoly::max_b2;
#endif
    if((!o.export_only && !o.gw_selftest && (o.save.empty() || !o.b2)) || o.b2>max_b2 || !o.count || !o.threads || o.threads>1024 || !o.memory || o.memory>1048576 ||
       (o.backend!="auto" && o.backend!="ntt" && o.backend!="plain" && o.backend!="gwpoly"))throw std::runtime_error("invalid options; see --help");
    if((o.d || o.gw_check) && o.backend!="gwpoly")throw std::runtime_error("--d/--gw-check require --backend gwpoly");
#ifndef ECM_CPU_HAVE_GWPOLY
    if(o.backend=="gwpoly" || o.gw_selftest)throw std::runtime_error("GW polynomial backend not compiled; build with -GwPoly");
#endif
    if(!o.aid.empty() && (o.aid.size()!=32 || o.aid.find_first_not_of("0123456789abcdefABCDEF")!=o.aid.npos))throw std::runtime_error("AID must contain 32 hex digits");
    if(o.exponent>1000000000 || (!o.known.empty() && !o.exponent))throw std::runtime_error("known factors require --exponent (max 1000000000)");
    if(!o.export_only && !o.gw_selftest && std::filesystem::absolute(o.save).lexically_normal()==std::filesystem::absolute(o.results).lexically_normal())throw std::runtime_error("results must not overwrite save");
    std::error_code error;
    if(!o.save.empty() && std::filesystem::equivalent(o.save,o.results,error))throw std::runtime_error("results alias the input save");
    if(o.submission.empty())throw std::runtime_error("empty submission path");
    const auto same=[](const std::string &a,const std::string &b){if(a.empty() || b.empty())return false;std::error_code e;return std::filesystem::absolute(a).lexically_normal()==std::filesystem::absolute(b).lexically_normal() || std::filesystem::equivalent(a,b,e);};
    if(same(o.save,o.submission) || same(o.results,o.submission))throw std::runtime_error("submission path conflicts with save/internal results");
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
int stopped(){return ecm_stage1_stop_requested()?1:0;}
struct Params {ecm_params p;Params(){ecm_init(p);}~Params(){ecm_clear(p);}};
struct Details {uint64_t d=0,poly_size=0,fft_length=0,checked_values=0;double init=0,main=0,maxerr=0;};
std::string result(const Options &o,const SavedCurve &r,const mpz_t n,const mpz_t f,int stage,double seconds,const Details &details) {
    ecm_results::Submission common;common.exponent=o.exponent;common.sigma=r.sigma;common.b1=r.b1;common.b2=o.b2;
    common.stage=stage;common.known=o.known;common.user=o.user;common.computer=o.computer;common.aid=o.aid;
    if(stage)common.factors.push_back(number(f));
    if(stage!=1){common.d=details.d;common.poly_size=details.poly_size;}
    auto text=ecm_results::internal_json(common,n,details.fft_length,ECM_BUILD_ID);text.resize(text.size()-2);
    std::ostringstream out;out<<text
       <<",\"stage2-engine\":{\"name\":"<<quote(o.backend=="gwpoly"?"gwnum-polymult":"libecm")<<",\"version\":"<<quote(
#ifdef ECM_CPU_HAVE_GWPOLY
           o.backend=="gwpoly"?ecm_gwpoly::version():
#endif
           ecm_version())<<",\"backend\":"<<quote(o.backend)
       <<",\"seconds\":"<<std::setprecision(9)<<seconds<<",\"save-line\":"<<r.line<<",\"coverage\":\"requested-B2\"";
    if(details.fft_length)out<<",\"fft-length\":"<<details.fft_length<<",\"init-seconds\":"<<details.init
        <<",\"main-seconds\":"<<details.main<<",\"max-roundoff\":"<<details.maxerr<<",\"checked-values\":"<<details.checked_values;
    out<<"}}\n";
    return out.str();
}
}
int main(int argc,char **argv) {
    try {
        const auto o=options(argc,argv);
#ifdef ECM_CPU_HAVE_GWPOLY
        if(o.gw_selftest){ecm_gwpoly::selftest(static_cast<unsigned>(o.threads));return 0;}
#endif
        if(o.export_only) {
            if(!ecm_results::has_result_ledger(o.submission))throw std::runtime_error("no owned submission ledger");
            ecm_results::seal_submissions(o.submission,2,{},true);std::cout<<"submission output regenerated\n";return 0;
        }
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
        omp_set_num_threads(static_cast<int>(o.threads));
        std::cout<<"CPU Edwards Stage2 backend="<<o.backend<<" threads="<<o.threads<<" memory_mib="<<o.memory<<" libecm="<<ecm_version()<<std::endl;
        for(uint64_t i=o.skip;i<end;++i) {
            ecm_results::check_stage2_outputs(o.results,o.submission,i>o.skip);
            if(stopped())return 130;const auto &r=rows[i];Integer n,x,d,px,py,a24,f;
            mpz_set_str(n.v,r.n.c_str(),16);mpz_set_str(x.v,r.x.c_str(),16);
            auto selected=o;
            if(selected.backend=="auto"){
#ifdef ECM_CPU_HAVE_GWPOLY
                selected.backend="gwpoly";
#else
                selected.backend=mpz_size(n.v)<=ECM_LIBECM_NTT_THRESHOLD?"ntt":"plain";
#endif
            }
            int stage=0;Details details;const auto begin=std::chrono::steady_clock::now();
            if(r.factor){mpz_set(f.v,x.v);stage=1;}
            else {
                const auto setup=construct(d.v,px.v,py.v,a24.v,f.v,r.sigma,n.v);
                if(setup==Setup::Degenerate)throw std::runtime_error("degenerate saved curve");
                if(setup==Setup::Factor)stage=1;
                else {
#ifdef ECM_CPU_HAVE_GWPOLY
                    if(selected.backend=="gwpoly"){
                        ecm_gwpoly::Options gw;gw.threads=static_cast<unsigned>(o.threads);gw.memory_mib=o.memory;
                        gw.exponent=o.exponent;gw.d=o.d;gw.stop=stopped;gw.check=o.gw_check;ecm_gwpoly::Stats stats;
                        try {if(ecm_gwpoly::stage2(f.v,n.v,x.v,a24.v,r.b1,o.b2,gw,stats))stage=2;}
                        catch(...){if(stopped())return 130;throw;}
                        details.d=stats.d;details.poly_size=stats.poly_size;details.fft_length=stats.fft_length;
                        details.init=stats.init_seconds;details.main=stats.main_seconds;details.maxerr=stats.max_roundoff;details.checked_values=stats.checked_values;
                        std::cout<<"gwpoly: D="<<stats.d<<" P="<<stats.poly_size<<" fft_length="<<stats.fft_length
                                 <<" blocks="<<stats.blocks<<" giants="<<stats.giants<<" polymults="<<stats.polymults
                                 <<" init="<<stats.init_seconds<<" main="<<stats.main_seconds<<" maxerr="<<stats.max_roundoff
                                 <<" checked_values="<<stats.checked_values<<" planned_mib="<<stats.estimated_memory_mib
                                 <<" points="<<stats.point_seconds<<" trees="<<stats.tree_seconds<<" folds="<<stats.fold_seconds
                                 <<" descent="<<stats.descent_seconds<<std::endl;
                    }else
#endif
                    {
                    Params params;auto p=params.p;p->sigma_is_A=1;
                    mpz_mul_ui(p->sigma,a24.v,4);mpz_sub_ui(p->sigma,p->sigma,2);mpz_mod(p->sigma,p->sigma,n.v);
                    mpz_set(p->x,x.v);p->B1done=static_cast<double>(r.b1);set64(p->B2,o.b2);
                    p->maxmem=static_cast<double>(o.memory)*1048576.0;p->use_ntt=selected.backend=="ntt";p->S=1;p->verbose=2;p->stop_asap=stopped;
                    const int rc=ecm_factor(f.v,n.v,static_cast<double>(r.b1),p);
                    if(stopped())return 130; // libecm may return NF on interruption.
                    if(rc<0)throw std::runtime_error("libecm failed; no result published for current curve");
                    if((rc==0 || rc==ECM_FACTOR_FOUND_STEP2) &&
                       (p->B1done!=static_cast<double>(r.b1) || mpz_cmp(p->x,x.v)))
                        throw std::runtime_error("libecm changed the completed Stage1 point");
                    if(rc>0)stage=rc==ECM_FACTOR_FOUND_STEP2?2:1;
                    }
                }
            }
            if(stage && (mpz_cmp_ui(f.v,1)<=0 || mpz_cmp(f.v,n.v)>=0 || !mpz_divisible_p(n.v,f.v)))throw std::runtime_error("saturated or invalid factor; no NF published");
            const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            const auto json=result(selected,r,n.v,f.v,stage,elapsed,details);
            ecm_results::Submission report;report.exponent=o.exponent;report.sigma=r.sigma;report.b1=r.b1;report.b2=o.b2;
            report.publication_task=publication_task;report.stage=stage;report.known=o.known;report.user=o.user;report.computer=o.computer;report.aid=o.aid;
            report.publication_order=i+1;
            report.save_name=std::filesystem::path(o.save).stem().u8string();
            report.d=details.d;report.poly_size=details.poly_size;report.stage2_fft_length=details.fft_length;
            if(stage)report.factors.push_back(number(f.v));
            ecm_results::check_stage2_outputs(o.results,o.submission,i>o.skip);
            ecm_results::publish_submission(o.submission,report,n.v);
            ecm_results::append_text_record(o.results,json);
            std::cout<<"completed save-line="<<r.line<<" sigma="<<r.sigma<<" status="<<(stage?"F":"NF")<<" seconds="<<elapsed<<std::endl;
        }
        ecm_results::check_stage2_outputs(o.results,o.submission,end>o.skip);
        ecm_results::seal_submissions(o.submission,2,publication_task,true);
        return 0;
    }catch(const std::exception &e){std::cerr<<"CPU Stage2: "<<e.what()<<'\n';return 1;}
}
