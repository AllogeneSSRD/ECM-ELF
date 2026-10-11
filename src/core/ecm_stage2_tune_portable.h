#pragma once
#include "ecm_stage2_tune_ecm.h"
#include "ecm_stage2_tune_ntt_profile.h"
#include "ecm_stage2_probability.h"
#include "ecm_stage1_cost_csv.h"
#include <chrono>
#include <charconv>

namespace ecm_stage2 { namespace tune { namespace portable {
constexpr unsigned format=6;
constexpr Word default_b2_max=2600000000000ull;
struct Grid {std::vector<unsigned> widths;std::vector<Word> ds,b2s;unsigned repeats=2;};
inline std::vector<size_t> neighbors(size_t center,size_t count) {
    if(center>=count)throw std::runtime_error("invalid tune neighborhood");
    std::vector<size_t> result;
    for(size_t i=center?center-1:0;i<=std::min(center+1,count-1);++i)result.push_back(i);
    return result;
}
struct UnsupportedFormat:std::runtime_error {
    UnsupportedFormat():std::runtime_error("unsupported tune format; regenerate this development profile"){}
};
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
inline void strip_ntt_auxiliary_timing(Fields &s) {
    for(const char *key:{"setup_seconds","warmup_seconds","verification_seconds"})s.erase(key);
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
        strip_ntt_auxiliary_timing(sample);
        if(!conditions.count(uint(sample,"condition")))throw std::runtime_error("unknown NTT measurement condition");
        for(auto &old:ntt)if(uint(old,"condition")==uint(sample,"condition") &&
            uint(old,"length")==uint(sample,"length") && uint(old,"batch")==uint(sample,"batch")) {
            if(real(sample,"median_seconds")>=real(old,"median_seconds"))return false;
            old=std::move(sample);++revision;return true;
        }
        if(ntt.size()>=65536)throw std::runtime_error("too many NTT summaries");
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
                if(values.size()!=columns.size())throw std::runtime_error("tune row/column size mismatch");
                Fields record=common;
                for(size_t i=0;i<columns.size();++i)if(!record.emplace(columns[i],values[i]).second)
                    throw std::runtime_error("tune column duplicates a common field");
                if(ntt_table) {
                    if(uint(record,"condition")!=table_key || out.ntt.size()>=65536)
                        throw std::runtime_error("NTT table condition/count mismatch");
                    out.ntt.push_back(std::move(record));
                } else {
                    if(uint(record,"target_bits")!=table_key || out.samples.size()>=65536)
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
                    const size_t begin=ntt_table?4u:6u;const auto sep=section.find(".table.",begin);
                    if(sep==section.npos)throw std::runtime_error("invalid parameter-table name");
                    table_key=cost::integer(section.substr(begin,sep-begin));cost::integer(section.substr(sep+7));
                    if(!ntt_table && (table_key<2 || table_key>16384))throw std::runtime_error("invalid width knot");
                    parameter_table=true;table=&common;
                } else throw std::runtime_error("unsupported tune table; regenerate this development profile");
            } else {
                const auto eq=line.find('=');if(!table || eq==line.npos)throw std::runtime_error("invalid tune row");
                const auto key=stage1_cost::trim(line.substr(0,eq));
                auto value=stage1_cost::trim(line.substr(eq+1));
                if(parameter_table && key=="columns") {
                    if(columns_seen || rows_seen)throw std::runtime_error("duplicate/late tune columns");
                    while(value.empty() || value.back()!=']') {
                        if(!std::getline(in,line))throw std::runtime_error("unterminated tune columns");
                        value+=' '+stage1_cost::trim(line);
                    }
                    std::set<std::string> names;
                    for(const auto &atom:scalar_array(value)) {
                        const auto name=text(Fields{{"v",atom}},"v");
                        fields("{\""+name+"\":0}");
                        if(!names.insert(name).second)throw std::runtime_error("duplicate tune column");
                        columns.push_back(name);
                    }
                    if(columns.size()>256)throw std::runtime_error("too many tune columns");
                    columns_seen=true;continue;
                }
                if(parameter_table && key=="rows") {
                    if(!columns_seen || rows_seen || value!="[")throw std::runtime_error("invalid tune rows");
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
           uint(out.metadata,"algorithm_revision")!=1 || out.conditions.empty())
            throw std::runtime_error("invalid tune header");
        out.revision=uint(out.metadata,"revision");std::set<std::string> scopes;
        for(const auto &s:out.samples) {
            validate(s);if(!out.conditions.count(uint(s,"condition")) || !scopes.insert(scope(s)).second)
                throw std::runtime_error("duplicate/unknown tune measurement scope");
        }
        std::set<std::tuple<Word,Word,Word>> ntt_scopes;
        for(auto &s:out.ntt) {
            validate_ntt_summary(s);
            strip_ntt_auxiliary_timing(s);
            if(!out.conditions.count(uint(s,"condition")) || !ntt_scopes.emplace(uint(s,"condition"),uint(s,"length"),uint(s,"batch")).second)
                throw std::runtime_error("duplicate/unknown tune NTT scope");
        }
        return out;
    }
    std::string serialize()const {
        std::ostringstream out;out<<"# Stage2 parameter tables. One final summary per row, not per-run samples.\n"
            <<"# Widths are calibration knots; values between knots are estimates.\n[profile]\n";
        Fields header=metadata;header["format"]=std::to_string(format);header["unit"]="\"full_stage2\"";
        header["algorithm_revision"]="1";header["revision"]=std::to_string(revision);
        auto emit_fields=[&](const Fields &values) {
            for(const auto &f:values)out<<f.first<<" = "<<compact_literal(f.second)<<'\n';
        };
        emit_fields(header);
        for(const auto &c:conditions){out<<"\n[condition."<<c.first<<"]\n";emit_fields(c.second);}
        using Group=std::vector<const Fields*>;
        std::map<std::pair<Word,std::string>,Group> groups;
        for(const auto &s:samples) {
            validate(s);std::string key;
            for(const char *name:{"condition","carrier_exponent","modulus_kind","execution_path","source"})key+=required(s,name)+';';
            for(const auto &f:s)key+=f.first+';';
            groups[{uint(s,"target_bits"),key}].push_back(&s);
        }
        auto emit_group=[&](const std::string &section,const Group &records,bool ntt) {
            Group ordered=records;
            if(ntt)std::stable_sort(ordered.begin(),ordered.end(),[](const Fields *a,const Fields *b) {
                return std::make_tuple(uint(*a,"log2_length"),uint(*a,"length"),uint(*a,"batch"),optional_uint(*a,"tile"))<
                       std::make_tuple(uint(*b,"log2_length"),uint(*b,"length"),uint(*b,"batch"),optional_uint(*b,"tile"));
            });
            Fields common=*records.front();
            for(auto it=common.begin();it!=common.end();) {
                bool same=true;for(auto row:records)if(row->at(it->first)!=it->second)same=false;
                const bool primary=ntt?(it->first=="log2_length" || it->first=="length" || it->first=="batch" || it->first=="tile" || it->first=="median_seconds"):
                    (it->first=="d" || it->first=="p" || it->first=="b2" || it->first=="median_seconds" || it->first=="mad_seconds");
                if(!same || primary)it=common.erase(it);else ++it;
            }
            std::vector<std::string> columns;
            for (const char *key : {
                "d", "p", "b2",                       // ECM 字段
                "log2_length", "length", "batch",       // NTT 规模
                "tile", "max_radix_bits",
                "passes_forward", "coop_outer",
                "warp_tail", "compact_scratch",       // 执行形状
                "conv_iter_per_s", "median_seconds", "repeats",
                "payload_bytes", "arena_payload_bytes",
                "external_arrays_bytes",
                "verified_words_per_sample",
                "batch_iter_per_s", "mad_seconds", "min_seconds", "max_seconds",
                "source", "validation_count",
                "validation_max_relative_error"
            })
                if (records.front()->count(key) && !common.count(key))
                    columns.push_back(key);
            for(const auto &f:*records.front())if(!common.count(f.first) &&
                std::find(columns.begin(),columns.end(),f.first)==columns.end())columns.push_back(f.first);
            out<<"\n["<<section<<"]\n";emit_fields(common);
            out<<"columns = [";size_t chars=11;
            for(size_t i=0;i<columns.size();++i) {
                const auto name=ecm_stage2::tune::portable::quoted(columns[i]);
                if(i){out<<", ";chars+=2;}
                if(chars+name.size()+2>112){out<<"\n    ";chars=4;}
                out<<name;chars+=name.size();
            }
            out<<"]\nrows = [\n";
            std::vector<size_t> widths(columns.size(),0);
            for(auto row:records)for(size_t i=0;i<columns.size();++i)
                widths[i]=std::max(widths[i],compact_literal(row->at(columns[i])).size());
            for(auto row:ordered) {
                out<<"    [";for(size_t i=0;i<columns.size();++i) {
                    if(i)out<<", ";const auto value=compact_literal(row->at(columns[i]));
                    out<<std::setw(int(widths[i]))<<value;
                }out<<"],\n";
            }out<<"]\n";
        };
        std::vector<Word> knots;
        for(const auto &g:groups)if(knots.empty() || knots.back()!=g.first.first)knots.push_back(g.first.first);
        Word width=0,index=0;
        for(const auto &g:groups){if(width!=g.first.first){width=g.first.first;index=0;
                const auto next=std::upper_bound(knots.begin(),knots.end(),width);
                out<<"\n# Width "<<width<<" bits; rows below describe this knot only.\n";
                if(next!=knots.end())out<<"# Interval ["<<width<<", "<<*next<<"): interior widths use model estimates.\n";
                else out<<"# Widths above "<<width<<" bits use model extrapolation.\n";
            }
            emit_group("width."+std::to_string(width)+".table."+std::to_string(index++),g.second,false);}
        groups.clear();
        for(const auto &s:ntt) {
            validate_ntt_summary(s);std::string key;for(const auto &f:s)key+=f.first+';';
            groups[{uint(s,"condition"),key}].push_back(&s);
        }
        Word condition=Word(-1);index=0;
        for(const auto &g:groups) {
            if(condition!=g.first.first){condition=g.first.first;index=0;}
            emit_group("ntt."+std::to_string(condition)+".table."+std::to_string(index++),g.second,true);
        }
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
