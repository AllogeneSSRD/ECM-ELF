#pragma once
#include "ecm_stage2_tune_ecm.h"
#include "ecm_stage2_tune_ntt_profile.h"
#include "ecm_stage2_probability.h"
#include "ecm_stage1_cost_csv.h"
#include <chrono>

namespace ecm_stage2 { namespace tune { namespace portable {
constexpr unsigned format=5;
constexpr Word default_b2_max=2600000000000ull;
struct Grid {std::vector<unsigned> widths;std::vector<Word> ds,b2s;unsigned repeats=2;};
inline Grid grid(unsigned level,Word upper=default_b2_max) {
    if(level<1 || level>10 || upper<2 || upper>Word(INT64_MAX)-8192)
        throw std::runtime_error("invalid portable tune grid");
    const unsigned widths[]={3,4,5,5,5,8,8,10,15,15};
    const unsigned steps[]={4096,3072,2048,2048,2048,1536,1536,1024,1024,1024};
    const unsigned counts[]={4,6,8,10,10,12,12,12,12,14};
    const unsigned bounds[]={3,3,4,4,6,6,6,8,8,8};
    const unsigned ratios[]={10,10,10,10,5,5,5,5,5,5};
    const unsigned repeats[]={2,2,2,2,3,3,3,3,5,5};
    const Word catalog[]={30030,60060,120120,180180,210210,360360,570570,
        690690,810810,1021020,1141140,1381380,1711710,2282280};
    const auto i=level-1;Grid out;out.widths.push_back(512);out.repeats=repeats[i];
    for(unsigned n=1;n<=widths[i];++n)out.widths.push_back(n*steps[i]);
    for(unsigned n=0;n<counts[i];++n) {
        const unsigned index=(n*13+(counts[i]-1)/2)/(counts[i]-1);
        out.ds.push_back(catalog[index]);
    }
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
inline bool measured(const Fields &sample){return required(sample,"source")=="\"measured\"";}
inline void validate_ntt_summary(const Fields &s) {
    const auto n=uint(s,"length"),b=uint(s,"batch");
    if(n<8 || n>(1ull<<27) || (n&(n-1)) || !b || b>65535 || uint(s,"bad") ||
       text(s,"source")!="measured" || text(s,"status")!="measured" ||
       text(s,"unit")!="field_convolution" || text(s,"reference_kind")!="gmp_3x3_distinct_constant_v1" ||
       uint(s,"verified_words_per_sample")!=n*b || !uint(s,"repeats") || uint(s,"repeats")>1000 ||
       !(real(s,"median_seconds")>0))throw std::runtime_error("unchecked portable NTT summary");
    Word log=0;for(auto x=n;x>1;x>>=1)++log;
    const double seconds=real(s,"median_seconds");
    auto close=[](double a,double b){return std::abs(a-b)<=1e-9*std::max(std::abs(a),std::abs(b));};
    if(uint(s,"log2_length")!=log || !close(real(s,"conv_iter_per_s"),b/seconds) ||
       !close(real(s,"batch_iter_per_s"),1/seconds))throw std::runtime_error("inconsistent portable NTT summary");
    real(s,"mad_seconds");uint(s,"condition");
    for(const auto &f:s)if(!f.second.empty() && f.second.front()=='[')
        throw std::runtime_error("portable NTT contains sampling arrays");
}
inline std::string scope(const Fields &s) {
    std::string key;
    for(const char *field:{"condition","target_bits","arithmetic_bits","carrier_exponent","modulus_kind",
                          "b1","b2","d","fold_resident","frontier_resident","execution_path"}) {
        const auto &value=required(s,field);key+=std::to_string(value.size())+":"+value;
    }return key;
}
inline void validate(const Fields &s) {
    const auto bits=uint(s,"target_bits"),arithmetic=uint(s,"arithmetic_bits"),carrier=uint(s,"carrier_exponent");
    const auto b1=uint(s,"b1"),b2=uint(s,"b2"),d=uint(s,"d");
    if(bits<2 || bits>16384 || arithmetic<bits || arithmetic>16384 || (carrier && carrier!=arithmetic) ||
       (!carrier && bits!=arithmetic) || b1<2 || b2<=b1 || b2>Word(INT64_MAX)-8192 ||
       d<6 || d%2 || d>200000000 || uint(s,"p")!=cost::phi(d)/2 ||
       uint(s,"fold_resident")>1 || uint(s,"frontier_resident")>1 ||
       !(real(s,"median_seconds")>0) || uint(s,"repeats")>1000)
        throw std::runtime_error("invalid portable ECM sample shape/cost");
    const auto kind=text(s,"modulus_kind"),source=text(s,"source");
    if((kind!="generic" && kind!="mersenne") || (carrier && kind!="mersenne") ||
       (source!="measured" && source!="model" && source!="calibrated" && source!="interpolated" && source!="extrapolated"))
        throw std::runtime_error("unsupported portable ECM source/arithmetic");
    if(measured(s) && (!uint(s,"repeats") || uint(s,"clean")!=1 || uint(s,"bad") ||
        uint(s,"hits") || !uint(s,"selftest_cases") || !uint(s,"checked")))
        throw std::runtime_error("portable measurement lacks completed arithmetic evidence");
    real(s,"mad_seconds");uint(s,"condition");text(s,"execution_path");
    for(const auto &field:s)if(!field.second.empty() && field.second.front()=='[')
        throw std::runtime_error("portable summaries must not contain sampling arrays");
    if(optional_uint(s,"validation_count"))real(s,"validation_max_relative_error");
    else if(s.count("validation_max_relative_error"))throw std::runtime_error("portable error without independent validation");
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
struct Profile {
    Fields metadata;
    std::map<Word,Fields> conditions;
    std::vector<Fields> samples,ntt;
    Word revision=0;
    Word condition(const Fields &fields) {
        for(const auto &old:conditions)if(old.second==fields)return old.first;
        if(conditions.size()>=256)throw std::runtime_error("too many portable tune conditions");
        const Word id=conditions.empty()?0:conditions.rbegin()->first+1;conditions.emplace(id,fields);return id;
    }
    bool update(Fields sample) {
        validate(sample);if(!conditions.count(uint(sample,"condition")))throw std::runtime_error("unknown measurement condition");
        const auto key=scope(sample);
        for(auto &old:samples)if(scope(old)==key) {
            if(measured(old) && (!measured(sample) || real(sample,"median_seconds")>=real(old,"median_seconds")))return false;
            if(!measured(old) && !measured(sample) && (!sample.count("model_evidence") ||
               (old.count("model_evidence") && old.at("model_evidence")==sample.at("model_evidence"))))return false;
            old=std::move(sample);++revision;return true;
        }
        if(samples.size()>=65536)throw std::runtime_error("too many portable tune samples");
        samples.push_back(std::move(sample));++revision;return true;
    }
    bool update_ntt(Fields sample) {
        validate_ntt_summary(sample);
        if(!conditions.count(uint(sample,"condition")))throw std::runtime_error("unknown NTT measurement condition");
        for(auto &old:ntt)if(uint(old,"condition")==uint(sample,"condition") &&
            uint(old,"length")==uint(sample,"length") && uint(old,"batch")==uint(sample,"batch")) {
            if(real(sample,"median_seconds")>=real(old,"median_seconds"))return false;
            old=std::move(sample);++revision;return true;
        }
        if(ntt.size()>=65536)throw std::runtime_error("too many NTT summaries");
        ntt.push_back(std::move(sample));++revision;return true;
    }
    static Profile from_legacy(const EcmProfile &legacy) {
        Profile out;Fields condition=legacy.device;
        if(condition.count("gl_add_sub_mask")){condition["add_sub_mask"]=condition.at("gl_add_sub_mask");condition.erase("gl_add_sub_mask");}
        for(const auto &field:legacy.policy)if(field.first!="environment")condition[field.first]=field.second;
        for(const auto &field:legacy.environment)condition["env_"+field.first]=field.second;
        const auto id=out.condition(condition);
        for(auto s:legacy.samples) {
            for(auto i=s.begin();i!=s.end();)if(!i->second.empty() && i->second.front()=='[')i=s.erase(i);else ++i;
            s["condition"]=std::to_string(id);s["source"]="\"measured\"";
            s["execution_path"]="\"legacy_resident\"";out.update(std::move(s));
        }
        for(const auto &entry:legacy.ntt_samples) {
            auto s=entry.second;s["mad_seconds"]=numeric(mad(array(required(s,"seconds"))));
            for(auto i=s.begin();i!=s.end();)if(!i->second.empty() && i->second.front()=='[')i=s.erase(i);else ++i;
            s["condition"]=std::to_string(id);s["source"]="\"measured\"";out.update_ntt(std::move(s));
        }
        out.metadata["effort_level"]=required(legacy.profile,"effort_level");return out;
    }
    static Profile from_ntt(const NttProfile &legacy) {
        Profile out;auto fields=legacy.device;
        if(fields.count("gl_add_sub_mask")){fields["add_sub_mask"]=fields.at("gl_add_sub_mask");fields.erase("gl_add_sub_mask");}
        for(const auto &e:legacy.environment)fields["env_"+e.first]=e.second;
        const auto id=out.condition(fields);
        for(const auto &entry:legacy.samples) {
            auto s=entry.second;s["mad_seconds"]=numeric(mad(array(required(s,"seconds"))));
            for(auto i=s.begin();i!=s.end();)if(!i->second.empty() && i->second.front()=='[')i=s.erase(i);else ++i;
            s["condition"]=std::to_string(id);s["source"]="\"measured\"";out.update_ntt(std::move(s));
        }
        out.metadata["component_unit"]="\"field_convolution\"";return out;
    }
    static Profile load(const std::filesystem::path &path) {
        std::error_code error;const auto bytes=std::filesystem::file_size(path,error);
        if(error || bytes>64*1048576)throw std::runtime_error("portable tune missing or exceeds 64MiB");
        std::ifstream in(path,std::ios::binary);Profile out;Fields *table=nullptr;std::string line;
        std::set<std::string> sections;bool first=true;
        while(std::getline(in,line)) {
            if(first && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);first=false;
            line=stage1_cost::trim(line);if(line.empty() || line[0]=='#')continue;
            if(line[0]=='[') {
                if(line.back()!=']')throw std::runtime_error("invalid portable tune table");
                const auto section=line.substr(1,line.size()-2);
                if(!sections.insert(section).second)throw std::runtime_error("duplicate portable tune table");
                if(section=="profile")table=&out.metadata;
                else if(section.compare(0,10,"condition.")==0)table=&out.conditions[cost::integer(section.substr(10))];
                else if(section.compare(0,7,"sample.")==0) {
                    cost::integer(section.substr(7));if(out.samples.size()>=65536)throw std::runtime_error("too many portable samples");
                    out.samples.emplace_back();table=&out.samples.back();
                } else if(section.compare(0,4,"ntt.")==0) {
                    cost::integer(section.substr(4));if(out.ntt.size()>=65536)throw std::runtime_error("too many portable NTT samples");
                    out.ntt.emplace_back();table=&out.ntt.back();
                } else {
                    // Native legacy reader verifies arrays/arithmetic before import.
                    if(out.metadata.count("format") && uint(out.metadata,"format")<format) {
                        if(required(out.metadata,"unit")=="\"field_convolution\"")return from_ntt(NttProfile::load(path));
                        return from_legacy(EcmProfile::load(path));
                    }
                    throw std::runtime_error("unsupported portable tune table");
                }
            } else {
                const auto eq=line.find('=');if(!table || eq==line.npos)throw std::runtime_error("invalid portable tune row");
                const auto key=stage1_cost::trim(line.substr(0,eq)),value=stage1_cost::trim(line.substr(eq+1));
                const auto parsed=fields("{\""+key+"\":"+value+"}");
                if(!value.empty() && value.front()=='[')throw std::runtime_error("portable profile contains sampling array");
                if(!table->emplace(key,parsed.at(key)).second)throw std::runtime_error("duplicate portable tune field");
                // Legacy NTT has slices in [profile], before its first policy
                // table. Delegate once the header identifies it, so legacy
                // proof/array validation remains the sole importer boundary.
                if(table==&out.metadata && out.metadata.count("format") && out.metadata.count("unit") &&
                   uint(out.metadata,"format")<format) {
                    if(required(out.metadata,"unit")=="\"field_convolution\"")return from_ntt(NttProfile::load(path));
                    return from_legacy(EcmProfile::load(path));
                }
            }
        }
        if(!in.eof() || uint(out.metadata,"format")!=format || required(out.metadata,"unit")!="\"full_stage2\"" ||
           uint(out.metadata,"algorithm_revision")!=1 || out.conditions.empty() || out.conditions.size()>256)
            throw std::runtime_error("invalid portable tune header");
        out.revision=uint(out.metadata,"revision");
        std::set<std::string> scopes;
        for(const auto &s:out.samples) {
            validate(s);if(!out.conditions.count(uint(s,"condition")) || !scopes.insert(scope(s)).second)
                throw std::runtime_error("duplicate/unknown portable measurement scope");
        }
        std::set<std::tuple<Word,Word,Word>> ntt_scopes;
        for(const auto &s:out.ntt) {
            validate_ntt_summary(s);
            if(!out.conditions.count(uint(s,"condition")) || !ntt_scopes.emplace(uint(s,"condition"),uint(s,"length"),uint(s,"batch")).second)
                throw std::runtime_error("duplicate/unknown portable NTT scope");
        }
        return out;
    }
    std::string serialize()const {
        std::ostringstream out;out<<"# Stage2 final performance summaries; no per-run samples. Seconds and bytes.\n[profile]\n";
        Fields header=metadata;header["format"]=std::to_string(format);header["unit"]="\"full_stage2\"";
        header["algorithm_revision"]="1";header["revision"]=std::to_string(revision);
        for(const auto &f:header)out<<f.first<<" = "<<f.second<<'\n';
        auto emit=[&](const std::string &section,const Fields &fields) {
            out<<"\n["<<section<<"]\n";
            for(const auto &f:fields) {
                if(!f.second.empty() && f.second.front()=='[')throw std::runtime_error("cannot persist portable sampling arrays");
                out<<f.first<<" = "<<f.second<<'\n';
            }
        };
        for(const auto &c:conditions)emit("condition."+std::to_string(c.first),c.second);
        for(size_t i=0;i<samples.size();++i){validate(samples[i]);emit("sample."+std::to_string(i),samples[i]);}
        for(size_t i=0;i<ntt.size();++i){validate_ntt_summary(ntt[i]);emit("ntt."+std::to_string(i),ntt[i]);}return out.str();
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
