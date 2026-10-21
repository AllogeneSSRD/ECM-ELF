#pragma once
#include "ecm_stage2_tune_ecm.h"
#include "ecm_stage2_tune_ntt_profile.h"
#include "ecm/core/stage2/probability.h"
#include "../../storage/profile/ecm_stage1_cost_csv.h"
#include "ecm_stage2_d_search.h"
#include <chrono>
#include <charconv>
#include <ctime>
#include "ecm/platform/program_version.h"

namespace ecm_stage2 { namespace tune { namespace portable {
constexpr unsigned format=8;
inline constexpr const char *matrix_column_order[]={
#define ECM_STAGE2_TUNE_COLUMN(name) #name,
#define ECM_STAGE2_TUNE_NTT_COLUMN(name)
#include "ecm_stage2_tune_columns.def"
#undef ECM_STAGE2_TUNE_COLUMN
#undef ECM_STAGE2_TUNE_NTT_COLUMN
};
inline constexpr const char *ntt_matrix_column_order[]={
#define ECM_STAGE2_TUNE_COLUMN(name)
#define ECM_STAGE2_TUNE_NTT_COLUMN(name) #name,
#include "ecm_stage2_tune_columns.def"
#undef ECM_STAGE2_TUNE_COLUMN
#undef ECM_STAGE2_TUNE_NTT_COLUMN
};
enum class MatrixSortKind {integer,optional_integer,string};
struct MatrixSortKey {const char *name;MatrixSortKind kind;};
inline constexpr MatrixSortKey ecm_row_order[]={
#define ECM_STAGE2_TUNE_ECM_ROW(name,kind) {#name,MatrixSortKind::kind},
#define ECM_STAGE2_TUNE_NTT_ROW(name,kind)
#include "ecm_stage2_tune_rows.def"
#undef ECM_STAGE2_TUNE_ECM_ROW
#undef ECM_STAGE2_TUNE_NTT_ROW
};
inline constexpr MatrixSortKey ntt_row_order[]={
#define ECM_STAGE2_TUNE_ECM_ROW(name,kind)
#define ECM_STAGE2_TUNE_NTT_ROW(name,kind) {#name,MatrixSortKind::kind},
#include "ecm_stage2_tune_rows.def"
#undef ECM_STAGE2_TUNE_ECM_ROW
#undef ECM_STAGE2_TUNE_NTT_ROW
};
constexpr Word default_b2_max=2600000000000ull;
struct Grid {std::vector<unsigned> widths;std::vector<Word> ds,b2s;unsigned repeats=2,basins=2;};
inline std::vector<size_t> neighbors(size_t center,size_t count) {
    if(center>=count)throw std::runtime_error("invalid tune neighborhood");
    std::vector<size_t> result;
    for(size_t i=center?center-1:0;i<=std::min(center+1,count-1);++i)result.push_back(i);
    return result;
}
// Discrete descent: measure the starting cell and its immediate neighbors.
// Move only on a significant measured improvement, then expose one new neighbor.
// Invalid candidates close the local direction; budget deferrals are unfinished.
struct LocalSearch {
    struct Timing {double seconds=0,stddev=0;};
    size_t center,count;std::map<size_t,Timing> tested;std::set<size_t> rejected,deferred;
    explicit LocalSearch(size_t start,size_t size):center(start),count(size) {
        if(start>=size)throw std::runtime_error("invalid local search start");
    }
    static bool improves(Timing a,Timing b) {
        return d_search_improves(a.seconds,a.stddev,b.seconds,b.stddev);
    }
    size_t next() {
        if(!tested.count(center) && !rejected.count(center) && !deferred.count(center))return center;
        for(;;) {
            const auto adjacent=neighbors(center,count);
            for(auto i:adjacent)if(!tested.count(i) && !rejected.count(i) && !deferred.count(i))return i;
            if(!tested.count(center))return count;
            auto winner=center;
            for(auto i:adjacent)if(tested.count(i) && improves(tested.at(i),tested.at(winner)))winner=i;
            if(winner==center)return count;
            center=winner;
        }
    }
    bool complete() {return next()==count && deferred.empty();}
};
struct UnsupportedFormat:std::runtime_error {
    UnsupportedFormat():std::runtime_error("unsupported tune format; regenerate this development profile"){}
};
inline Grid grid(unsigned level,Word upper=default_b2_max) {
    if(level<1 || level>10 || upper<2 || upper>Word(INT64_MAX)-8192)
        throw std::runtime_error("invalid portable tune grid");
    const unsigned widths[]={3,4,5,5,5,8,8,10,15,15};
    const unsigned steps[]={4096,3072,2048,2048,2048,1536,1536,1024,1024,1024};
    const unsigned basins[]={2,2,3,3,3,4,4,5,6,6};
    const unsigned bounds[]={3,3,4,4,6,6,6,8,8,8};
    const unsigned ratios[]={10,10,10,10,5,5,5,5,5,5};
    const unsigned repeats[]={2,2,2,2,3,3,3,3,5,5};
    const auto i=level-1;Grid out;out.widths.push_back(512);out.repeats=repeats[i];
    out.basins=basins[i];
    for(unsigned n=1;n<=widths[i];++n)out.widths.push_back(n*steps[i]);
    for(const auto &entry:d_catalog::entries())out.ds.push_back(entry.d);
    out.b2s.push_back(upper);
    for(unsigned n=1;n<bounds[i];++n)out.b2s.push_back(std::max(Word(2),out.b2s.back()/ratios[i]));
    std::sort(out.b2s.begin(),out.b2s.end());
    out.b2s.erase(std::unique(out.b2s.begin(),out.b2s.end()),out.b2s.end());return out;
}
inline std::string numeric(double x) {
    if(!std::isfinite(x) || x<0)throw std::runtime_error("invalid portable tune number");
    std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();
}
inline std::string quoted(const std::string &s) {
    std::string out="\"";for(char c:s) {
        if(static_cast<unsigned char>(c)<32)throw std::runtime_error("control character in portable tune string");
        if(c=='"' || c=='\\')out+='\\';out+=c;
    }return out+'"';
}
inline std::string text(const Fields &f,const char *key) {
    const auto &s=required(f,key);
    if(s.size()<2 || s.front()!='"' || s.back()!='"')throw std::runtime_error("expected portable tune string");
    std::string out;for(size_t i=1;i<s.size()-1;++i) {
        if(s[i]=='\\'){if(++i>=s.size()-1 || (s[i]!='\\' && s[i]!='"'))throw std::runtime_error("unsupported tune escape");}
        out+=s[i];
    }return out;
}
inline Word optional_uint(const Fields &f,const char *key,Word fallback=0){return f.count(key)?uint(f,key):fallback;}
inline double optional_real(const Fields &f,const char *key,double fallback=0){return f.count(key)?real(f,key):fallback;}
template<size_t N> inline bool matrix_row_less(const Fields &a,const Fields &b,const MatrixSortKey (&keys)[N]) {
    for(const auto &key:keys) {
        if(key.kind==MatrixSortKind::string) {
            const auto x=text(a,key.name),y=text(b,key.name);if(x!=y)return x<y;
        } else {
            const auto x=key.kind==MatrixSortKind::optional_integer?optional_uint(a,key.name):uint(a,key.name);
            const auto y=key.kind==MatrixSortKind::optional_integer?optional_uint(b,key.name):uint(b,key.name);
            if(x!=y)return x<y;
        }
    }
    return false;
}
inline std::string utc_updated_at(Word nanoseconds) {
    const auto seconds=std::time_t(nanoseconds/1000000000);std::tm utc{};
#ifdef _WIN32
    if(gmtime_s(&utc,&seconds))throw std::runtime_error("invalid measurement time");
#else
    if(!gmtime_r(&seconds,&utc))throw std::runtime_error("invalid measurement time");
#endif
    std::ostringstream out;out<<std::put_time(&utc,"%Y-%m-%dT%H:%M:%S")<<'.'
        <<std::setfill('0')<<std::setw(9)<<nanoseconds%1000000000<<'Z';return out.str();
}
inline void stamp_provenance(Fields &s,Word completed_ns=0) {
    if(!completed_ns)completed_ns=Word(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    s["program_version"]=ecm_stage2::tune::portable::quoted(ecm_program::version);
    s["updated_at"]=ecm_stage2::tune::portable::quoted(utc_updated_at(completed_ns));
}
inline std::string updated_at(const Fields &s) {return s.count("updated_at")?text(s,"updated_at"):std::string{};}
inline void validate_provenance(const Fields &s) {
    if(s.count("program_version") && text(s,"program_version").size()>256)throw std::runtime_error("invalid source version");
    const auto time=updated_at(s);if(time.empty())return;
    const std::string pattern="0000-00-00T00:00:00.000000000Z";
    if(time.size()!=pattern.size())throw std::runtime_error("updated_at requires UTC nanosecond format");
    for(size_t i=0;i<time.size();++i)if(pattern[i]=='0'?time[i]<'0' || time[i]>'9':time[i]!=pattern[i])
        throw std::runtime_error("invalid updated_at UTC timestamp");
    std::tm parsed{};std::istringstream in(time.substr(0,19));in>>std::get_time(&parsed,"%Y-%m-%dT%H:%M:%S");
    if(in.fail())throw std::runtime_error("invalid updated_at calendar time");
}
inline void merge_provenance(Fields &to,const Fields &from) {
    // Unknown legacy times stay unknown. Reading or formatting is not a measurement.
    if(updated_at(from)>updated_at(to) ||
       (updated_at(from)==updated_at(to) && !to.count("program_version"))) {
        for(const char *key:{"program_version","updated_at"}) {
            if(from.count(key))to[key]=from.at(key);else to.erase(key);
        }
    }
}
inline Word ntt_option(const Fields &f,const char *key) {
    const auto at=f.find(key);if(at==f.end())return 0;
    return at->second=="true"?1:at->second=="false"?0:uint(f,key);
}
inline bool measured(const Fields &sample){return required(sample,"source")=="\"measured\"";}
inline bool runtime_observation(const Fields &sample) {
    return sample.count("benchmark_kind") && text(sample,"benchmark_kind")=="runtime_save";
}
inline double average(const std::vector<double> &values) {
    double mean=0;size_t n=0;for(double x:values)mean+=(x-mean)/double(++n);return mean;
}
inline double deviation(const std::vector<double> &values) {
    double mean=0,m2=0;size_t n=0;
    for(double x:values){const double delta=x-mean;mean+=delta/double(++n);m2+=delta*(x-mean);}
    return n>1?std::sqrt(std::max(0.,m2/double(n-1))):0;
}
// Length-prefixed scalar strings keep the matrix flat and identities unambiguous.
inline std::set<std::string> measurement_ids(const Fields &s) {
    std::set<std::string> ids;
    if(s.count("measurement_ids")) {
        const auto value=text(s,"measurement_ids");size_t pos=0;
        while(pos<value.size()) {
            const auto sep=value.find(':',pos);if(sep==value.npos)throw std::runtime_error("invalid measurement identity set");
            const auto length=cost::integer(value.substr(pos,sep-pos));pos=sep+1;
            if(!length || length>value.size()-pos || !ids.insert(value.substr(pos,size_t(length))).second)
                throw std::runtime_error("invalid/duplicate measurement identity");
            pos+=size_t(length);
        }
    } else if(s.count("runtime_sample_id"))ids.insert(text(s,"runtime_sample_id"));
    else if(measured(s))ids.insert("summary:"+required(s,"mean_seconds")+":"+required(s,"stddev_seconds")+":"+required(s,"repeats"));
    return ids;
}
inline std::string encoded_ids(const std::set<std::string> &ids) {
    std::string value;for(const auto &id:ids)value+=std::to_string(id.size())+":"+id;return ecm_stage2::tune::portable::quoted(value);
}
inline void validate_ntt_summary(const Fields &s) {
    validate_provenance(s);
    const auto n=uint(s,"length"),b=uint(s,"batch");
    if(n<8 || n>(1ull<<27) || (n&(n-1)) || !b || b>65535 || uint(s,"bad") ||
       text(s,"source")!="measured" || text(s,"status")!="measured" ||
       text(s,"unit")!="field_convolution" || text(s,"reference_kind")!="gmp_3x3_distinct_constant_v1" ||
       uint(s,"verified_words_per_sample")!=n*b || !uint(s,"repeats") || uint(s,"repeats")>1000000000 ||
       !(real(s,"mean_seconds")>0))throw std::runtime_error("unchecked portable NTT summary");
    Word log=0;for(auto x=n;x>1;x>>=1)++log;
    const double seconds=real(s,"mean_seconds");
    auto close=[](double a,double b){return std::abs(a-b)<=1e-9*std::max(std::abs(a),std::abs(b));};
    if(uint(s,"log2_length")!=log || !close(real(s,"conv_iter_per_s"),b/seconds) ||
       !close(real(s,"batch_iter_per_s"),1/seconds))throw std::runtime_error("inconsistent portable NTT summary");
    if(s.count("iter_per_s") && !close(real(s,"iter_per_s"),b/seconds))
        throw std::runtime_error("inconsistent NTT convolution throughput");
    if(s.count("passes_forward") && (!uint(s,"passes_forward") || uint(s,"passes_forward")>log+1))
        throw std::runtime_error("invalid NTT forward pass count");
    if(s.count("tile") && s.count("max_radix_bits") && uint(s,"tile")<=log && uint(s,"max_radix_bits")) {
        const auto remaining=log-uint(s,"tile"),radix=uint(s,"max_radix_bits");
        const auto passes=remaining/radix+(remaining%radix!=0)+(uint(s,"tile")!=0);
        if(s.count("passes_forward") && uint(s,"passes_forward")!=passes)
            throw std::runtime_error("inconsistent NTT forward pass count");
    }
    const auto sd=real(s,"stddev_seconds");uint(s,"condition");
    if(uint(s,"repeats")==1 && sd!=0)throw std::runtime_error("single NTT observation has dispersion");
    for(const auto &f:s)if(!f.second.empty() && f.second.front()=='[')
        throw std::runtime_error("portable NTT contains sampling arrays");
}
inline void strip_ntt_auxiliary_timing(Fields &s) {
    for(const char *key:{"setup_seconds","warmup_seconds","verification_seconds"})s.erase(key);
}
inline std::string runtime_group(const Fields &s) {
    std::string key;
    for(const char *field:{"condition","target_bits","arithmetic_bits","carrier_exponent","modulus_kind",
                          "b1","b2","d","fold_resident","frontier_resident","execution_path"})
        key+=std::string(field)+'='+required(s,field)+';';
    for(const char *field:{"giant_chunk_points","giant_chain_min","giant_force_ladder"}) {
        const auto at=s.find(field);key+=std::string(field)+'='+(at==s.end()?"absent":at->second)+';';
    }
    return key;
}
inline std::string scope(const Fields &s) {
    // Formal calibration and production timing remain distinct populations.
    if(runtime_observation(s))return "runtime;"+runtime_group(s);
    std::string key="formal;";
    for(const char *field:{"condition","target_bits","arithmetic_bits","carrier_exponent","modulus_kind",
                          "b1","b2","d","fold_resident","frontier_resident","execution_path"}) {
        const auto &value=required(s,field);key+=std::to_string(value.size())+":"+value;
    }
    for(const char *field:{"giant_chunk_points","giant_chain_min","giant_force_ladder"})
        key+=std::string(field)+'='+std::to_string(optional_uint(s,field, std::string(field)=="giant_chain_min"?32768:0))+';';
    return key;
}
inline void validate(const Fields &s) {
    validate_provenance(s);
    const auto bits=uint(s,"target_bits"),arithmetic=uint(s,"arithmetic_bits"),carrier=uint(s,"carrier_exponent");
    const auto b1=uint(s,"b1"),b2=uint(s,"b2"),d=uint(s,"d");
    if(bits<2 || bits>16384 || arithmetic<bits || arithmetic>16384 || (carrier && carrier!=arithmetic) ||
       (!carrier && bits!=arithmetic) || b1<2 || b2<=b1 || b2>Word(INT64_MAX)-8192 ||
       d<6 || d%2 || d>200000000 || uint(s,"p")!=cost::phi(d)/2 ||
       uint(s,"fold_resident")>1 || uint(s,"frontier_resident")>1 ||
       !(real(s,"mean_seconds")>0) || uint(s,"repeats")>1000000000)
        throw std::runtime_error("invalid portable ECM sample shape/cost");
    const auto kind=text(s,"modulus_kind"),source=text(s,"source");
    if((kind!="generic" && kind!="mersenne") || (carrier && kind!="mersenne") ||
       (source!="measured" && source!="model" && source!="calibrated" && source!="interpolated" && source!="extrapolated"))
        throw std::runtime_error("unsupported portable ECM source/arithmetic");
    if(measured(s) && (!uint(s,"repeats") ||
       (s.count("verified")?uint(s,"verified")!=1:
        (uint(s,"clean")!=1 || uint(s,"bad") || uint(s,"hits") || !uint(s,"selftest_cases") || !uint(s,"checked")))))
        throw std::runtime_error("portable measurement lacks completed arithmetic evidence");
    if(runtime_observation(s) && !measured(s))
        throw std::runtime_error("invalid runtime aggregate count");
    if(runtime_observation(s) && s.count("runtime_sample_id") &&
       (text(s,"runtime_sample_id").empty() || text(s,"runtime_sample_id").size()>256))
        throw std::runtime_error("invalid runtime observation identity");
    if(optional_uint(s,"runtime_probe")>1)throw std::runtime_error("invalid runtime probe marker");
    const auto spread=real(s,"stddev_seconds");
    if(uint(s,"repeats")==1 && spread!=0)throw std::runtime_error("single observation has nonzero standard deviation");
    uint(s,"condition");text(s,"execution_path");
    if(measured(s) && s.count("verified") &&
       (!s.count("measurement_ids") || measurement_ids(s).empty() ||
        (runtime_observation(s) && measurement_ids(s).size()!=uint(s,"repeats"))))
        throw std::runtime_error("invalid aggregate measurement identities");
    for(const auto &field:s)if(!field.second.empty() && field.second.front()=='[')
        throw std::runtime_error("portable summaries must not contain sampling arrays");
    if(optional_uint(s,"validation_count"))real(s,"validation_max_relative_error");
    else if(s.count("validation_max_relative_error"))throw std::runtime_error("portable error without independent validation");
    if(s.count("measured_model_ratio") || s.count("predicted_seconds") || s.count("prediction_model") || s.count("prediction_basis")) {
        const auto predicted=real(s,"predicted_seconds"),ratio=real(s,"measured_model_ratio");
        const auto basis=text(s,"prediction_basis");text(s,"prediction_model");
        if(!measured(s) || !(predicted>0 && ratio>0) ||
           (basis!="before_measurement" && basis!="leave_one_out") ||
           std::abs(ratio-real(s,"mean_seconds")/predicted)>1e-9*std::max(1.,ratio))
            throw std::runtime_error("inconsistent measured/model time ratio");
    }
}
struct Ignore {
    bool gpu=true,driver=true,cuda=true,backend=true,memory=false,environment=true;
    static Ignore parse(const std::string &list) {
        Ignore out;out.gpu=out.driver=out.cuda=out.backend=out.environment=false;
        std::set<std::string> seen;std::istringstream in(list);std::string item;
        while(std::getline(in,item,',')) {
            item=stage1_cost::trim(item);if(item.empty() || !seen.insert(item).second)throw std::runtime_error("invalid tune ignore list");
            if(item=="gpu")out.gpu=true;else if(item=="driver")out.driver=true;else if(item=="cuda")out.cuda=true;
            else if(item=="backend")out.backend=true;else if(item=="memory")out.memory=true;
            else if(item=="environment")out.environment=true;else throw std::runtime_error("unknown tune ignore item: "+item);
        }return out;
    }
};
inline bool matches(const Fields &old,const Fields &current,const Ignore &ignore) {
    auto same=[&](const char *key){auto a=old.find(key),b=current.find(key);return a!=old.end() && b!=current.end() && a->second==b->second;};
    if(!ignore.gpu && (!same("uuid_hex") || !same("sm_major") || !same("sm_minor")))return false;
    if(!ignore.driver && !same("cuda_driver"))return false;
    if(!ignore.cuda && !same("cuda_runtime"))return false;
    if(!ignore.backend && (!same("gl_fixed_mode") || !same("outer_unroll_u") || !same("add_sub_mask")))return false;
    if(!ignore.environment) {
        for(const auto &f:old)if(f.first.compare(0,4,"env_")==0 && !same(f.first.c_str()))return false;
        for(const auto &f:current)if(f.first.compare(0,4,"env_")==0 && !same(f.first.c_str()))return false;
    }
    // Memory equality is checked against each actual candidate's execution_path,
    // not raw batch/arena/fold budget numbers stored in the condition.
    return true;
}
inline bool ntt_matches(Fields old,Fields current,const Ignore &ignore) {
    // S4 scheduling changes do not change the standalone field benchmark.
    for(const char *key:{"env_s4_short_batch_tune","env_s4_short_batch_hash"}) {old.erase(key);current.erase(key);}
    return matches(old,current,ignore);
}
// Restricted TOML arrays contain scalar table cells, never trial samples.
inline std::vector<std::string> scalar_array(const std::string &input) {
    auto s=stage1_cost::trim(input);
    if(s.size()<2 || s.front()!='[' || s.back()!=']')throw std::runtime_error("invalid tune cell array");
    std::vector<std::string> result;size_t i=1;
    while(i<s.size()-1) {
        while(i<s.size()-1 && std::isspace(static_cast<unsigned char>(s[i])))++i;
        if(i==s.size()-1)break;
        const auto begin=i;
        if(s[i]=='"') {
            bool closed=false;++i;
            while(i<s.size()-1) {
                if(s[i]=='\\'){i+=2;continue;}
                if(s[i++]=='"'){closed=true;break;}
            }
            if(!closed)throw std::runtime_error("unterminated tune cell");
        } else while(i<s.size()-1 && s[i]!=',')++i;
        auto value=stage1_cost::trim(s.substr(begin,i-begin));
        if(value.empty() || value.front()=='[')throw std::runtime_error("nested/empty tune cell");
        fields("{\"v\":"+value+"}");result.push_back(value);
        while(i<s.size()-1 && std::isspace(static_cast<unsigned char>(s[i])))++i;
        if(i<s.size()-1 && s[i++]!=',')throw std::runtime_error("invalid tune cell separator");
    }
    return result;
}
inline std::string compact_literal(const std::string &value) {
    if(value.empty() || value.front()=='[')throw std::runtime_error("cannot persist trial arrays");
    if(value.front()=='"' || value=="true" || value=="false" || value.find_first_of(".eE")==value.npos)return value;
    const auto number=real(Fields{{"v",value}},"v");char buffer[64];
    const auto written=std::to_chars(buffer,buffer+sizeof(buffer),number);
    if(written.ec!=std::errc{})throw std::runtime_error("cannot format tune cell");
    return std::string(buffer,written.ptr);
}

inline Fields compact_sample(Fields s) {
    validate(s);
    if(measured(s)) {
        s["verified"]="1";s["measurement_ids"]=encoded_ids(measurement_ids(s));
        if(!optional_uint(s,"runtime_probe",1))s["preferred_order"]=std::to_string(optional_uint(s,"runtime_completed_ns"));
    }
    static const std::set<std::string> keep={
        "condition","target_bits","arithmetic_bits","carrier_exponent","modulus_kind","b1","b2","d","p",
        "fold_resident","frontier_resident","execution_path","giant_chunk_points","giant_chain_min","giant_force_ladder",
        "source","benchmark_kind","program_version","updated_at","verified","repeats","mean_seconds","stddev_seconds",
        "measurement_ids","preferred_order","validation_count","validation_max_relative_error",
        "phase_shape_seconds","phase_setup_seconds","phase_baby_seconds","phase_ftree_seconds","phase_main_setup_seconds",
        "phase_inverse_setup_seconds","phase_giant_loop_seconds","phase_descent_seconds","phase_accum_seconds","phase_finalize_seconds",
        "giant_seconds","gtrees_seconds","fold_seconds","model_relative_error_allowance"};
    for(auto i=s.begin();i!=s.end();)if(!keep.count(i->first))i=s.erase(i);else ++i;
    return s;
}
// Merge sufficient statistics, preserving slow observations and sample variance.
inline bool combine(Fields &old,const Fields &sample,Word limit=1000000000) {
    const auto before=measurement_ids(old),added=measurement_ids(sample);
    if(std::includes(before.begin(),before.end(),added.begin(),added.end()))return false;
    if(std::includes(added.begin(),added.end(),before.begin(),before.end())) {
        if(uint(sample,"repeats")<uint(old,"repeats"))throw std::runtime_error("measurement snapshot count regressed");
        if(uint(sample,"repeats")>limit)return false;
        const auto order=std::max(optional_uint(old,"preferred_order"),optional_uint(sample,"preferred_order"));
        const auto previous=old;old=sample;merge_provenance(old,previous);
        if(order)old["preferred_order"]=std::to_string(order);return true;
    }
    for(const auto &id:added)if(before.count(id))
        throw std::runtime_error("partially overlapping tune snapshots; recover deduplicated original JSONL instead");
    const auto n=uint(old,"repeats"),m=uint(sample,"repeats");
    if(n>=limit || m>limit-n)return false;
    const double a=real(old,"mean_seconds"),b=real(sample,"mean_seconds"),delta=b-a;
    const double x=real(old,"stddev_seconds"),y=real(sample,"stddev_seconds");
    const double m2=double(n-1)*x*x+double(m-1)*y*y+delta*delta*double(n)*double(m)/double(n+m);
    for(auto &f:old)if(f.first!="stddev_seconds" && f.first.size()>=8 &&
        f.first.compare(f.first.size()-8,8,"_seconds")==0 && sample.count(f.first))
        f.second=numeric(real(old,f.first.c_str())+(real(sample,f.first.c_str())-real(old,f.first.c_str()))*double(m)/double(n+m));
    old["mean_seconds"]=numeric(a+delta*double(m)/double(n+m));
    old["stddev_seconds"]=numeric(std::sqrt(std::max(0.,m2/double(n+m-1))));old["repeats"]=std::to_string(n+m);
    auto ids=before;ids.insert(added.begin(),added.end());old["measurement_ids"]=encoded_ids(ids);
    const auto order=std::max(optional_uint(old,"preferred_order"),optional_uint(sample,"preferred_order"));
    if(order)old["preferred_order"]=std::to_string(order);
    const auto checks=optional_uint(old,"validation_count")+optional_uint(sample,"validation_count");
    if(checks){old["validation_count"]=std::to_string(checks);old["validation_max_relative_error"]=numeric(std::max(optional_real(old,"validation_max_relative_error"),optional_real(sample,"validation_max_relative_error")));}
    merge_provenance(old,sample);
    return true;
}
inline void hydrate(Fields &s,const Fields &condition,bool ntt=false) {
    if(!s.count("program_version") && condition.count("program_version"))s["program_version"]=condition.at("program_version");
    if(ntt) {
        const auto n=uint(s,"length"),b=uint(s,"batch");Word log=0;for(auto x=n;x>1;x>>=1)++log;
        if(s.count("log2_length") && uint(s,"log2_length")!=log)
            throw std::runtime_error("inconsistent NTT length logarithm");
        s["log2_length"]=std::to_string(log);s["source"]=s["status"]="\"measured\"";s["unit"]="\"field_convolution\"";
        s["reference_kind"]="\"gmp_3x3_distinct_constant_v1\"";s["verified_words_per_sample"]=std::to_string(n*b);s["bad"]="0";
        if(s.count("batch_iter_per_s") && std::abs(real(s,"batch_iter_per_s")-1/real(s,"mean_seconds"))>
            1e-9*std::max(real(s,"batch_iter_per_s"),1/real(s,"mean_seconds")))
            throw std::runtime_error("inconsistent NTT batch throughput");
        s["conv_iter_per_s"]=numeric(b/real(s,"mean_seconds"));s["batch_iter_per_s"]=numeric(1/real(s,"mean_seconds"));
        if(s.count("iter_per_s") && std::abs(real(s,"iter_per_s")-b/real(s,"mean_seconds"))>
            1e-9*std::max(real(s,"iter_per_s"),b/real(s,"mean_seconds")))
            throw std::runtime_error("inconsistent NTT convolution throughput");
        s["iter_per_s"]=s["conv_iter_per_s"];
        if(!s.count("passes_forward") && s.count("tile") && s.count("max_radix_bits") &&
            uint(s,"tile")<=log && uint(s,"max_radix_bits")) {
            const auto remaining=log-uint(s,"tile"),radix=uint(s,"max_radix_bits");
            s["passes_forward"]=std::to_string(remaining/radix+(remaining%radix!=0)+(uint(s,"tile")!=0));
        }
    } else {
        const auto d=uint(s,"d");if(d<6 || d>200000000 || d%2)throw std::runtime_error("invalid tune D");
        const Fields derived={{"arithmetic_bits",std::to_string(optional_uint(s,"carrier_exponent")?uint(s,"carrier_exponent"):uint(s,"target_bits"))},
            {"p",std::to_string(cost::phi(uint(s,"d"))/2)},
            {"execution_path",ecm_stage2::tune::portable::quoted(uint(s,"fold_resident")?(uint(s,"frontier_resident")?"resident":"host_frontier"):"host_fold_frontier")}};
        for(const auto &f:derived){if(s.count(f.first) && s.at(f.first)!=f.second)throw std::runtime_error("inconsistent derived tune field");s[f.first]=f.second;}
    }
}
struct Profile {
    Fields metadata;
    std::map<Word,Fields> conditions;
    std::vector<Fields> samples,ntt;
    Word revision=0;
    Word condition(Fields fields) {
        fields.erase("program_version");fields.erase("updated_at");
        for(const auto &old:conditions)if(old.second==fields)return old.first;
        if(conditions.size()>=256)throw std::runtime_error("too many portable tune conditions");
        const Word id=conditions.empty()?0:conditions.rbegin()->first+1;conditions.emplace(id,fields);return id;
    }
    bool update(Fields sample,Word runtime_limit=10) {
        if(!conditions.count(uint(sample,"condition")))throw std::runtime_error("unknown measurement condition");
        sample=compact_sample(std::move(sample));
        const auto key=scope(sample);
        if(runtime_observation(sample)) {
            if(runtime_limit<2 || runtime_limit>1000)throw std::runtime_error("runtime sample limit must be 2..1000");
        }
        for(auto &old:samples)if(scope(old)==key) {
            if(measured(old) && !measured(sample))return false;
            if(measured(old) && measured(sample)) {
                const Word limit=runtime_observation(sample)?std::max(runtime_limit,uint(old,"repeats")):1000000000;
                if(!combine(old,sample,limit))return false;
            } else {if(old==sample)return false;old=std::move(sample);}
            ++revision;return true;
        }
        if(runtime_observation(sample) && uint(sample,"repeats")>runtime_limit)return false;
        samples.push_back(std::move(sample));++revision;return true;
    }
    bool update_ntt(Fields sample) {
        validate_ntt_summary(sample);
        strip_ntt_auxiliary_timing(sample);
        if(!conditions.count(uint(sample,"condition")))throw std::runtime_error("unknown NTT measurement condition");
        hydrate(sample,conditions.at(uint(sample,"condition")),true);
        sample["measurement_ids"]=encoded_ids(measurement_ids(sample));sample["verified"]="1";
        for(auto &old:ntt)if(uint(old,"condition")==uint(sample,"condition") &&
            uint(old,"length")==uint(sample,"length") && uint(old,"batch")==uint(sample,"batch")) {
            bool same=true;
            for(const char *k:{"tile","max_radix_bits","coop_outer","warp_tail","compact_scratch"})
                if(ntt_option(old,k)!=ntt_option(sample,k))same=false;
            if(!same)continue;
            if(!combine(old,sample))return false;old.erase("iter_per_s");old.erase("batch_iter_per_s");
            hydrate(old,conditions.at(uint(old,"condition")),true);++revision;return true;
        }
        ntt.push_back(std::move(sample));++revision;return true;
    }
    static Profile load(const std::filesystem::path &path) {
        std::error_code error;const auto bytes=std::filesystem::file_size(path,error);
        if(error || bytes>64*1048576)throw std::runtime_error("tune summary missing or exceeds 64MiB");
        std::ifstream in(path,std::ios::binary);Profile out;Fields *table=nullptr;std::string line;
        Fields common;std::vector<std::string> columns;bool parameter_table=false,ntt_table=false,rows=false;
        Word table_key=0,row_count=0;std::set<std::string> sections;bool first=true,columns_seen=false,rows_seen=false;
        auto finish=[&]() {
            if(parameter_table && (!columns_seen || !rows_seen || !row_count || rows))
                throw std::runtime_error("incomplete tune parameter table");
            parameter_table=false;common.clear();columns.clear();row_count=0;columns_seen=rows_seen=false;
        };
        while(std::getline(in,line)) {
            if(first && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);first=false;
            line=stage1_cost::trim(line);if(line.empty() || line[0]=='#')continue;
            if(rows) {
                if(line=="]"){rows=false;continue;}
                if(line.back()==',')line.pop_back();
                auto values=scalar_array(line);
                if(!columns_seen) {
                    std::set<std::string> names;
                    for(const auto &atom:values) {
                        const auto name=text(Fields{{"v",atom}},"v");
                        fields("{\""+name+"\":0}");
                        if(!names.insert(name).second)throw std::runtime_error("duplicate tune column");
                        columns.push_back(name);
                    }
                    if(columns.empty() || columns.size()>256)throw std::runtime_error("invalid tune matrix header");
                    columns_seen=true;continue;
                }
                if(values.size()!=columns.size())throw std::runtime_error("tune row/column size mismatch");
                Fields record=common;
                for(size_t i=0;i<columns.size();++i)if(values[i]!="\"\"" && !record.emplace(columns[i],values[i]).second)
                    throw std::runtime_error("tune column duplicates a common field");
                if(ntt_table) {
                    if(uint(record,"condition")!=table_key)
                        throw std::runtime_error("NTT table condition/count mismatch");
                    out.ntt.push_back(std::move(record));
                } else {
                    if(uint(record,"target_bits")!=table_key)
                        throw std::runtime_error("width table target/count mismatch");
                    out.samples.push_back(std::move(record));
                }
                ++row_count;continue;
            }
            if(line[0]=='[') {
                finish();if(line.back()!=']')throw std::runtime_error("invalid tune table");
                const auto section=line.substr(1,line.size()-2);
                if(!sections.insert(section).second)throw std::runtime_error("duplicate tune table");
                if(section=="profile")table=&out.metadata;
                else if(section.compare(0,10,"condition.")==0) {
                    if(out.conditions.size()>=256)throw std::runtime_error("too many tune conditions");
                    table=&out.conditions[cost::integer(section.substr(10))];
                } else if(section.compare(0,6,"width.")==0 || section.compare(0,4,"ntt.")==0) {
                    ntt_table=section.compare(0,4,"ntt.")==0;
                    const size_t begin=ntt_table?4u:6u;table_key=cost::integer(section.substr(begin));
                    if(!ntt_table && (table_key<2 || table_key>16384))throw std::runtime_error("invalid width knot");
                    parameter_table=true;table=&common;common[ntt_table?"condition":"target_bits"]=std::to_string(table_key);
                } else throw std::runtime_error("unsupported tune table; regenerate this development profile");
            } else {
                const auto eq=line.find('=');if(!table || eq==line.npos)throw std::runtime_error("invalid tune row");
                const auto key=stage1_cost::trim(line.substr(0,eq));
                auto value=stage1_cost::trim(line.substr(eq+1));
                if(parameter_table && key=="matrix") {
                    if(rows_seen || value!="[")throw std::runtime_error("invalid tune matrix");
                    rows=rows_seen=true;continue;
                }
                if(parameter_table && rows_seen)throw std::runtime_error("common field after tune rows");
                if(!value.empty() && value.front()=='[')throw std::runtime_error("sampling arrays are not tune summaries");
                const auto parsed=fields("{\""+key+"\":"+value+"}");
                if(!table->emplace(key,parsed.at(key)).second)throw std::runtime_error("duplicate tune field");
                if(table==&out.metadata && key=="format" && uint(out.metadata,"format")!=format)
                    throw UnsupportedFormat();
            }
        }
        finish();
        if(!in.eof() || uint(out.metadata,"format")!=format || required(out.metadata,"unit")!="\"full_stage2\"" ||
           out.conditions.empty())
            throw std::runtime_error("invalid tune header");
        std::set<std::string> scopes;
        for(auto &s:out.samples) {
            if(!out.conditions.count(uint(s,"condition")))throw std::runtime_error("unknown tune condition");
            hydrate(s,out.conditions.at(uint(s,"condition")));validate(s);
            out.revision=std::max(out.revision,optional_uint(s,"preferred_order"));if(!out.conditions.count(uint(s,"condition")) || !scopes.insert(scope(s)).second)
                throw std::runtime_error("duplicate/unknown tune measurement scope");
        }
        std::set<std::array<Word,8>> ntt_scopes;
        for(auto &s:out.ntt) {
            if(!out.conditions.count(uint(s,"condition")) || optional_uint(s,"verified")!=1)throw std::runtime_error("unchecked NTT summary");
            hydrate(s,out.conditions.at(uint(s,"condition")),true);validate_ntt_summary(s);
            strip_ntt_auxiliary_timing(s);
            if(!ntt_scopes.insert({uint(s,"condition"),uint(s,"length"),uint(s,"batch"),ntt_option(s,"tile"),ntt_option(s,"max_radix_bits"),
                ntt_option(s,"coop_outer"),ntt_option(s,"warp_tail"),ntt_option(s,"compact_scratch")}).second)
                throw std::runtime_error("duplicate/unknown tune NTT scope");
        }
        // Legacy format-8 conditions included a program version. Transfer it
        // to rows, then merge conditions that now describe the same hardware.
        Profile normalized;normalized.metadata=out.metadata;std::map<Word,Word> remap;
        for(const auto &c:out.conditions)remap[c.first]=normalized.condition(c.second);
        std::map<std::string,size_t> merged;
        for(auto s:out.samples) {
            s["condition"]=std::to_string(remap.at(uint(s,"condition")));const auto key=scope(s);
            const auto at=merged.find(key);
            if(at==merged.end()){merged[key]=normalized.samples.size();normalized.samples.push_back(std::move(s));}
            else {
                auto &old=normalized.samples[at->second];
                if(measured(old) && measured(s))combine(old,s);
                else if(!measured(old))old=std::move(s);
            }
        }
        for(auto s:out.ntt){s["condition"]=std::to_string(remap.at(uint(s,"condition")));normalized.update_ntt(std::move(s));}
        normalized.revision=out.revision;return normalized;
    }
    std::string serialize()const {
        std::ostringstream out;out<<"# Stage2 mean timings; identical parameters share one repeated measurement row.\n[profile]\nformat = "<<format<<"\nunit = \"full_stage2\"\n";
        if(metadata.count("error_limit"))out<<"error_limit = "<<compact_literal(metadata.at("error_limit"))<<'\n';
        for(const auto &c:conditions) {
            out<<"\n[condition."<<c.first<<"]\n";
            for(const auto &f:c.second)if(f.first!="program_version" && f.first!="updated_at")out<<f.first<<" = "<<compact_literal(f.second)<<'\n';
        }
        auto emit=[&](const std::string &section,std::vector<Fields> rows,bool ntt) {
            std::set<std::string> names;
            const std::set<std::string> omitted=ntt?std::set<std::string>{"condition","source","status","unit","reference_kind","verified_words_per_sample","bad","conv_iter_per_s"}:
                std::set<std::string>{"target_bits","p","arithmetic_bits","execution_path"};
            for(const auto &s:rows)for(const auto &f:s)if(!omitted.count(f.first))names.insert(f.first);
            names.insert("program_version");names.insert("updated_at");
            std::vector<std::string> columns;
            if(ntt) {
                for(const char *k:ntt_matrix_column_order)names.insert(k);
                for(const char *k:ntt_matrix_column_order)if(names.erase(k))columns.push_back(k);
            } else for(const char *k:matrix_column_order)if(names.erase(k))columns.push_back(k);
            columns.insert(columns.end(),names.begin(),names.end());
            std::stable_sort(rows.begin(),rows.end(),[&](const Fields &a,const Fields &b){
                return ntt?matrix_row_less(a,b,ntt_row_order):matrix_row_less(a,b,ecm_row_order);
            });
            std::vector<std::vector<std::string>> matrix(1);
            for(const auto &c:columns)matrix[0].push_back(ecm_stage2::tune::portable::quoted(c));
            for(const auto &s:rows){std::vector<std::string> row;for(const auto &c:columns){const auto at=s.find(c);row.push_back(at==s.end()?"\"\"":compact_literal(at->second));}matrix.push_back(std::move(row));}
            std::vector<size_t> widths(columns.size());for(const auto &row:matrix)for(size_t i=0;i<row.size();++i)widths[i]=std::max(widths[i],row[i].size());
            out<<"\n["<<section<<"]\nmatrix = [\n";
            for(const auto &row:matrix){out<<"    [";for(size_t i=0;i<row.size();++i){if(i)out<<", ";out<<std::setw(int(widths[i]))<<row[i];}out<<"],\n";}out<<"]\n";
        };
        std::map<Word,std::vector<Fields>> groups;
        for(const auto &s:samples){auto row=compact_sample(s);groups[uint(s,"target_bits")].push_back(std::move(row));}
        for(auto &g:groups)emit("width."+std::to_string(g.first),std::move(g.second),false);
        groups.clear();
        static const std::set<std::string> ntt_fields={"log2_length","length","iter_per_s","batch_iter_per_s","batch","tile","max_radix_bits","passes_forward","coop_outer","warp_tail","compact_scratch","repeats","mean_seconds","stddev_seconds","verified","measurement_ids","program_version","updated_at"};
        for(auto s:ntt){hydrate(s,conditions.at(uint(s,"condition")),true);validate_ntt_summary(s);Fields row;for(const auto &f:s)if(ntt_fields.count(f.first))row.insert(f);row["verified"]="1";groups[uint(s,"condition")].push_back(std::move(row));}
        for(auto &g:groups)emit("ntt."+std::to_string(g.first),std::move(g.second),true);
        return out.str();
    }
};
class Budget {
    std::chrono::steady_clock::time_point start;
    double seconds;
public:
    explicit Budget(double value):start(std::chrono::steady_clock::now()),seconds(value) {
        if(!std::isfinite(value) || value<=0)throw std::runtime_error("invalid tune budget");
    }
    double elapsed()const{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
    double remaining()const{return std::max(0.,seconds-elapsed());}
    bool expired()const{return elapsed()>=seconds;}
};
} } }
