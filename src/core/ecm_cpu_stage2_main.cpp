// Standalone CPU Edwards continuation. This target's <ecm.h> is libecm's API.
#include <ecm.h>
#include "ecm_edwards_save_reader.h"
#include "ecm_result_submission.h"
#include "ecm_submission_writer.h"
#include "generated/ecm_config_generated.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <omp.h>

#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_LIBECM_NTT_THRESHOLD
#define ECM_LIBECM_NTT_THRESHOLD 30
#endif
namespace {
using namespace ecm_edwards;
std::string quote(const std::string &s) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<c;
    }
    out<<'"';return out.str();
}
std::string number(const mpz_t x) {
    std::vector<char> b(mpz_sizeinbase(x,10)+3);mpz_get_str(b.data(),10,x);return b.data();
}
void set64(mpz_t x,uint64_t n) {mpz_import(x,1,1,sizeof n,0,0,&n);}
uint32_t crc32(const std::string &s) {
    uint32_t c=0xffffffffu;
    for(unsigned char b:s){c^=b;for(int i=0;i<8;++i)c=(c>>1)^((c&1)?0xedb88320u:0);}
    return c^0xffffffffu;
}
std::string crc_hex(const std::string &s) {
    std::ostringstream out;out<<std::uppercase<<std::hex<<std::setw(8)<<std::setfill('0')<<crc32(s);return out.str();
}
std::string timestamp() {
    const auto now=std::time(nullptr);std::tm t{};
#ifdef _WIN32
    gmtime_s(&t,&now);
#else
    gmtime_r(&now,&t);
#endif
    std::ostringstream out;out<<std::put_time(&t,"%Y-%m-%d %H:%M:%S");return out.str();
}
struct Options {
    std::string save,results="results_cpu_stage2.jsonl",submission,ini,backend="auto",user,computer,aid;
    std::vector<std::string> known;uint64_t b2=0,skip=0,count=UINT64_MAX,exponent=0,memory=512,threads=1;
    bool dry=false,export_only=false;uintptr_t stop_handle=0;int worker=1;
};
Options options(int argc,char **argv) {
    Options o;
    // Read local identity first so explicit CLI values, including empty values,
    // always override the selected global/worker INI projection.
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if((key=="--ini" || key=="-ini" || key=="--worker") && i+1<argc) {
            const std::string value=argv[++i];
            if(key=="--worker"){const auto n=decimal(value);if(!n || n>1000000)throw std::runtime_error("invalid worker");o.worker=int(n);}
            else o.ini=value;
        }
    }
    const auto ini=o.ini.empty()?std::filesystem::path(argv[0]).parent_path()/"ecm.ini":std::filesystem::path(o.ini);
    std::vector<ecm_config::IniLine> lines;
    const bool read=ecm_config::read_ini(ini.string(),lines);
    if(!read && (!o.ini.empty() || std::filesystem::exists(ini)))throw std::runtime_error("cannot read local INI");
    const auto settings=ecm_config::read_stage2(ecm_config::cli_entries(lines,o.worker,true));
    o.user=settings.result_user;o.computer=settings.result_computer;
    if(settings.has_submission && settings.submission.empty())throw std::runtime_error("empty submission path");
    const auto filename=settings.has_submission?settings.submission:ecm_config::worker_file(settings.submission,o.worker);
    o.submission=(ini.parent_path()/filename).string();
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {
            std::cout<<"CPU Edwards Stage2 (libecm)\n--save FILE --b2 DECIMAL (<=1000000000000) [--backend auto|ntt|plain] [--memory-mib 512]\n"
                "[--threads 1] [--results INTERNAL.jsonl] [--submission results.json.txt] [--ini FILE] [--worker N]\n"
                "[--skip N] [--curves N] [--dry-run]\n"
                "--export-results --submission FILE regenerates owned submission output without arithmetic\n"
                "[--exponent P] [--known-factor DECIMAL (repeatable)] [--user NAME] [--computer NAME] [--aid HEX32]\n";
            std::exit(0);
        }
        if(key=="--dry-run"){o.dry=true;continue;}
        if(key=="--export-results"){o.export_only=true;continue;}
        if(++i>=argc)throw std::runtime_error("missing value for "+key);std::string value=argv[i];
        if(key=="--save")o.save=value;else if(key=="--results")o.results=value;
        else if(key=="--submission")o.submission=value;
        else if(key=="--ini" || key=="-ini" || key=="--worker")continue;
        else if(key=="--backend")o.backend=value;else if(key=="--b2")o.b2=decimal(value);
        else if(key=="--memory-mib")o.memory=decimal(value);else if(key=="--threads")o.threads=decimal(value);
        else if(key=="--skip")o.skip=decimal(value);else if(key=="--curves")o.count=decimal(value);
        else if(key=="--exponent")o.exponent=decimal(value);else if(key=="--known-factor")o.known.push_back(value);
        else if(key=="--user")o.user=value;else if(key=="--computer")o.computer=value;else if(key=="--aid")o.aid=value;
        else if(key=="--stop-handle")o.stop_handle=decimal(value);
        else throw std::runtime_error("unknown option: "+key);
    }
    if((!o.export_only && (o.save.empty() || !o.b2)) || o.b2>1000000000000ull || !o.count || !o.threads || o.threads>1024 || !o.memory || o.memory>1048576 ||
       (o.backend!="auto" && o.backend!="ntt" && o.backend!="plain"))throw std::runtime_error("invalid options; see --help");
    if(!o.aid.empty() && (o.aid.size()!=32 || o.aid.find_first_not_of("0123456789abcdefABCDEF")!=o.aid.npos))throw std::runtime_error("AID must contain 32 hex digits");
    if(o.exponent>1000000000 || (!o.known.empty() && !o.exponent))throw std::runtime_error("known factors require --exponent (max 1000000000)");
    if(!o.export_only && std::filesystem::absolute(o.save).lexically_normal()==std::filesystem::absolute(o.results).lexically_normal())throw std::runtime_error("results must not overwrite save");
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
std::string result(const Options &o,const SavedCurve &r,const mpz_t n,const mpz_t f,int stage,double seconds) {
    const bool hit=stage!=0;const auto when=timestamp();const auto fac=hit?number(f):"";
    std::string known;for(const auto &v:o.known){if(!known.empty())known+=',';known+=v;}
    // PrMers JsonBuilder's public ECM canonical layout, without optional torsion.
    // Generic-N extension uses N=<decimal> in its first slot; no server claim.
    const auto target=o.exponent?std::to_string(o.exponent):"N="+number(n);
    const auto b2=stage==1?std::string():std::to_string(o.b2);
    const std::string canonical=target+";ECM;"+fac+";"+known+";"+std::to_string(r.b1)+";"+b2+";E"+
        std::to_string(r.sigma)+";0;invariant:0;"+ecm_program::name+";"+ecm_program::version+";;;Windows;x86_64;"+when;
    std::ostringstream out;out<<"{\"status\":"<<quote(hit?"F":"NF");
    if(o.exponent)out<<",\"exponent\":"<<o.exponent;
    else {const auto decimal_n=number(n);out<<",\"N\":"<<(decimal_n.size()>2000?quote(decimal_n):decimal_n);}
    out<<",\"worktype\":\"ECM\",\"b1\":"<<r.b1;
    if(stage!=1)out<<",\"b2\":"<<o.b2;
    out<<",\"curves\":1,\"Edwards\":{\"sigma\":"<<r.sigma<<"},\"fft-length\":0";
    if(hit)out<<",\"stage\":"<<stage<<",\"factors\":["<<quote(fac)<<"]";
    if(!o.known.empty()) {out<<",\"known-factors\":[";for(size_t i=0;i<o.known.size();++i){if(i)out<<',';out<<quote(o.known[i]);}out<<']';}
    out<<",\"errors\":{\"invariant\":0},\"program\":{\"name\":"<<quote(ecm_program::name)<<",\"version\":"<<quote(ecm_program::version)<<",\"port\":\"Windows x86_64\"}"
       <<",\"source-build\":"<<quote(ECM_BUILD_ID)
       <<",\"os\":{\"os\":\"Windows\",\"architecture\":\"x86_64\"},\"timestamp\":"<<quote(when);
    if(!o.user.empty())out<<",\"user\":"<<quote(o.user);
    if(!o.computer.empty())out<<",\"computer\":"<<quote(o.computer);
    if(!o.aid.empty())out<<",\"aid\":"<<quote(o.aid);
    out<<",\"checksum\":{\"version\":1,\"checksum\":"<<quote(crc_hex(canonical))<<"}"
       <<",\"stage2-engine\":{\"name\":\"libecm\",\"version\":"<<quote(ecm_version())<<",\"backend\":"<<quote(o.backend)
       <<",\"seconds\":"<<std::setprecision(9)<<seconds<<",\"save-line\":"<<r.line<<",\"coverage\":\"requested-B2\"}}\n";
    return out.str();
}
}
int main(int argc,char **argv) {
    try {
        const auto o=options(argc,argv);
        if(o.export_only) {
            if(!std::filesystem::exists(ecm_results::submission_ledger(o.submission)/"format"))throw std::runtime_error("no owned submission ledger");
            ecm_results::seal_submissions(o.submission);std::cout<<"submission output regenerated\n";return 0;
        }
        auto rows=ecm_edwards::read_saved_curves(o.save);
        if(o.skip>=rows.size())throw std::runtime_error("skip exceeds saved curve count");
        const auto end=o.skip+std::min<uint64_t>(o.count,rows.size()-o.skip);
        // Validate the entire selected batch before any computation or publication.
        for(uint64_t i=o.skip;i<end;++i)validate_target(o,rows[i]);
        if(o.dry){std::cout<<"validated "<<end-o.skip<<" curves; B2="<<o.b2<<'\n';return 0;}
        ecm_results::refresh_submission_if_owned(o.submission);
        ecm_stage1_install_stop_handler();if(o.stop_handle && !ecm_stage1_set_stop_handle(o.stop_handle))throw std::runtime_error("invalid stop handle");
        omp_set_num_threads(static_cast<int>(o.threads));
        std::cout<<"CPU Edwards Stage2 libecm="<<ecm_version()<<" backend="<<o.backend<<" threads="<<o.threads<<" memory_mib="<<o.memory<<std::endl;
        for(uint64_t i=o.skip;i<end;++i) {
            if(stopped()){ecm_results::seal_submissions(o.submission);return 130;}const auto &r=rows[i];Integer n,x,d,px,py,a24,f;
            mpz_set_str(n.v,r.n.c_str(),16);mpz_set_str(x.v,r.x.c_str(),16);
            auto selected=o;
            if(selected.backend=="auto")selected.backend=mpz_size(n.v)<=ECM_LIBECM_NTT_THRESHOLD?"ntt":"plain";
            int stage=0;const auto begin=std::chrono::steady_clock::now();
            if(r.factor){mpz_set(f.v,x.v);stage=1;}
            else {
                const auto setup=construct(d.v,px.v,py.v,a24.v,f.v,r.sigma,n.v);
                if(setup==Setup::Degenerate)throw std::runtime_error("degenerate saved curve");
                if(setup==Setup::Factor)stage=1;
                else {
                    Params params;auto p=params.p;p->sigma_is_A=1;
                    mpz_mul_ui(p->sigma,a24.v,4);mpz_sub_ui(p->sigma,p->sigma,2);mpz_mod(p->sigma,p->sigma,n.v);
                    mpz_set(p->x,x.v);p->B1done=static_cast<double>(r.b1);set64(p->B2,o.b2);
                    p->maxmem=static_cast<double>(o.memory)*1048576.0;p->use_ntt=selected.backend=="ntt";p->S=1;p->verbose=2;p->stop_asap=stopped;
                    const int rc=ecm_factor(f.v,n.v,static_cast<double>(r.b1),p);
                    if(stopped()){ecm_results::seal_submissions(o.submission);return 130;} // libecm may return NF on interruption.
                    if(rc<0)throw std::runtime_error("libecm failed; no result published for current curve");
                    if((rc==0 || rc==ECM_FACTOR_FOUND_STEP2) &&
                       (p->B1done!=static_cast<double>(r.b1) || mpz_cmp(p->x,x.v)))
                        throw std::runtime_error("libecm changed the completed Stage1 point");
                    if(rc>0)stage=rc==ECM_FACTOR_FOUND_STEP2?2:1;
                }
            }
            if(stage && (mpz_cmp_ui(f.v,1)<=0 || mpz_cmp(f.v,n.v)>=0 || !mpz_divisible_p(n.v,f.v)))throw std::runtime_error("saturated or invalid factor; no NF published");
            const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            const auto json=result(selected,r,n.v,f.v,stage,elapsed);
            ecm_results::Submission report;report.exponent=o.exponent;report.sigma=r.sigma;report.b1=r.b1;report.b2=o.b2;
            report.stage=stage;report.known=o.known;report.user=o.user;report.computer=o.computer;report.aid=o.aid;
            if(stage)report.factors.push_back(number(f.v));
            ecm_results::publish_submission(o.submission,report,n.v);
            if(!publish_save(o.results,json,true))throw std::runtime_error("result publication failed");
            std::cout<<"completed save-line="<<r.line<<" sigma="<<r.sigma<<" status="<<(stage?"F":"NF")<<" seconds="<<elapsed<<std::endl;
        }
        ecm_results::seal_submissions(o.submission);
        return 0;
    }catch(const std::exception &e){std::cerr<<"CPU Stage2: "<<e.what()<<'\n';return 1;}
}
