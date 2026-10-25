#pragma once
#include "../../gui/worker_controller.h"
#include "io/json.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <future>
#include <windows.h>

namespace ecmgui_v2::ui {
inline constexpr size_t stage2_phase_count=11;
inline constexpr const char* stage2_phase_keys[]{"read","plan","baby","ftree","inverse","giant","descent","gcd","check","factor","publish"};
inline int stage2_phase_index(const std::string& phase){
    const char* prefixes[]{"Read and validate","Plan Stage2","Generate and normalize","Build baby-point","Prepare polynomial inverse","Giant points","Normalize fold","Accumulate leaf","Validate arithmetic","Optional factor","Publish result"};
    for(int i=0;i<int(stage2_phase_count);++i)if(phase.compare(0,std::strlen(prefixes[i]),prefixes[i])==0)return i;
    if(phase=="read"||phase=="starting")return 0;if(phase=="calibration")return 1;return -1;
}
inline bool stage2_finished(const ecmgui::Stage2Status& s){return s.phase=="curve_done"||s.phase=="task_done"||(s.phase=="complete"&&s.current>0&&s.done>=s.current);}
inline float stage2_bounded_fraction(const ecmgui::Stage2Status& s,double estimated){
    if(stage2_finished(s))return 1;
    return std::isfinite(estimated)?float(std::clamp(estimated,0.,.99)):0;
}
using PhaseSeconds=std::array<double,stage2_phase_count>;
inline PhaseSeconds no_phase_seconds(){PhaseSeconds a;a.fill(-1);return a;}
struct Stage2Timing {
    PhaseSeconds seconds=no_phase_seconds();
    std::array<int,stage2_phase_count> sources{}; // 0 missing, 1 previous completed curve, 2 tune.
    bool any()const{for(double s:seconds)if(std::isfinite(s)&&s>0)return true;return false;}
    PhaseSeconds fractions()const{
        auto out=seconds;double sum=0;for(auto& n:out){if(!std::isfinite(n)||n<0)n=0;sum+=n;}
        if(sum<=0){out.fill(1./stage2_phase_count);return out;}for(auto& n:out)n/=sum;return out;
    }
};
inline Stage2Timing stage2_timing(const PhaseSeconds& previous,const PhaseSeconds& tune,int mode){
    Stage2Timing out;
    for(size_t i=0;i<stage2_phase_count;++i){
        if(mode!=1&&std::isfinite(tune[i])&&tune[i]>=0){out.seconds[i]=tune[i];out.sources[i]=2;}
        if(mode!=2&&std::isfinite(previous[i])&&previous[i]>=0){out.seconds[i]=previous[i];out.sources[i]=1;}
    }return out;
}
inline std::string stage2_scope(const ecmgui::WorkerView& w,const std::string& device,std::string engine_stamp={}){
    const auto& s=w.stage2_status;std::error_code error;
    if(engine_stamp.empty()){const auto stamp=std::filesystem::last_write_time(std::filesystem::u8path(w.exe),error);engine_stamp=error?"unknown":std::to_string(stamp.time_since_epoch().count());}
    // Sigma differs per curve; duration scope follows the engine, device and shape.
    return w.exe+"|"+engine_stamp+"|"+device+"|"+std::to_string(w.device)+"|"+s.curve_model+"|"+std::to_string(s.param)+"|"+std::to_string(s.bits)+"|"+std::to_string(s.d)+"|"+std::to_string(s.b1)+"|"+std::to_string(s.b2);
}

// Read-only view of this repository's scalar/matrix tune summary subset.
// It does not grant a profile computational calibration eligibility.
class Stage2Tune {
    using Value=ecm_results::json::Value;
    using Record=std::map<std::string,Value>;
    std::vector<Record> records_;
    std::map<std::string,Record> conditions_;
    std::string loaded_;
    static std::string trim(std::string s){const auto a=s.find_first_not_of(" \t\r\n");return a==s.npos?std::string{}:s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);}
    static std::string text(const Record& r,const std::string& key){auto i=r.find(key);return i==r.end()?"":i->second.scalar;}
    static double real(const Record& r,const std::string& key){try{const auto s=text(r,key);size_t n=0;double x=std::stod(s,&n);return n==s.size()&&std::isfinite(x)?x:-1;}catch(...){return -1;}}
    static std::string uuid(std::string s){std::string out;for(char c:s)if(std::isxdigit(static_cast<unsigned char>(c)))out+=char(std::tolower(static_cast<unsigned char>(c)));return out;}
public:
    std::string notice;
    void load(const std::filesystem::path& path){
        std::error_code e;auto stamp=std::filesystem::last_write_time(path,e);
        const auto signature=path.u8string()+"|"+(e?"missing":std::to_string(stamp.time_since_epoch().count()));
        if(signature==loaded_)return;loaded_=signature;records_.clear();conditions_.clear();notice.clear();
        if(e)return;
        try{
            if(std::filesystem::file_size(path)>64*1048576)throw std::runtime_error("tune exceeds 64 MiB");
            std::ifstream in(path);if(!in)throw std::runtime_error("cannot read tune");
            Record common,metadata;std::string section,line;bool matrix=false;std::vector<std::string> columns;
            auto finish=[&](){if(matrix)throw std::runtime_error("unfinished tune matrix");
                if(section.compare(0,10,"condition.")==0)conditions_[section.substr(10)]=common;
                else if(section=="device")conditions_["device"]=common;
                else if(section=="profile")metadata=common;
                else if(section.compare(0,11,"ecm.sample_")==0)records_.push_back(common);
                common.clear();columns.clear();};
            while(std::getline(in,line)){
                if(line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);line=trim(line);
                if(line.empty()||line[0]=='#')continue;
                if(matrix){
                    if(line=="]"){matrix=false;continue;}if(line.back()==',')line.pop_back();
                    auto values=ecm_results::json::parse(line);if(values.type!='a')throw std::runtime_error("invalid tune matrix row");
                    if(columns.empty()){for(const auto& v:values.items){if(v.type!='s'||std::find(columns.begin(),columns.end(),v.scalar)!=columns.end())throw std::runtime_error("invalid tune columns");columns.push_back(v.scalar);}if(columns.empty()||columns.size()>256)throw std::runtime_error("invalid tune columns");}
                    else{if(values.items.size()!=columns.size())throw std::runtime_error("tune row size mismatch");auto r=common;
                        for(size_t i=0;i<columns.size();++i)if(values.items[i].type!='s'||!values.items[i].scalar.empty())if(!r.emplace(columns[i],values.items[i]).second)throw std::runtime_error("duplicate tune field");
                        if(section.compare(0,6,"width.")==0)records_.push_back(std::move(r));}
                    if(records_.size()>65536)throw std::runtime_error("too many tune rows");continue;
                }
                if(line[0]=='['){finish();if(line.back()!=']')throw std::runtime_error("invalid tune section");section=line.substr(1,line.size()-2);continue;}
                const auto eq=line.find('=');if(eq==line.npos)throw std::runtime_error("invalid tune field");
                auto key=trim(line.substr(0,eq)),v=trim(line.substr(eq+1));
                if(key=="matrix"&&v=="["){matrix=true;continue;}
                if(!common.emplace(key,ecm_results::json::parse(v)).second)throw std::runtime_error("duplicate tune field");
            }finish();
            const double format=real(metadata,"format");
            if(format<3||format>8||text(metadata,"unit")!="full_stage2")throw std::runtime_error("unsupported tune summary");
        }catch(const std::exception& error){records_.clear();conditions_.clear();notice=error.what();}
    }
    PhaseSeconds lookup(const ecmgui::Stage2Status& s,const std::string& device)const{
        auto result=no_phase_seconds();if(!s.bits||!s.d||!s.b2)return result;double best=std::numeric_limits<double>::infinity();
        for(const auto& r:records_){
            if(text(r,"phase_accounting")!="exclusive_engine_v1"||real(r,"d")!=double(s.d)||real(r,"b2")!=double(s.b2))continue;
            // Portable summaries store checked as a positive word/sample count.
            if((r.count("checked")&&text(r,"checked")!="true"&&real(r,"checked")<=0)||(r.count("verified")&&text(r,"verified")!="true"&&real(r,"verified")<=0))continue;
            const auto condition=conditions_.find(r.count("condition")?text(r,"condition"):"device");
            if(condition==conditions_.end()||device.empty()||uuid(text(condition->second,"uuid_hex"))!=uuid(device))continue;
            const double bits=real(r,"arithmetic_bits")>0?real(r,"arithmetic_bits"):real(r,"bits");
            if(bits<=0)continue;const double score=std::abs(std::log(bits/double(s.bits)));if(score>=best)continue;
            auto candidate=no_phase_seconds();auto add=[&](int i,const char* key){const auto value=real(r,key);if(value>=0)candidate[i]=candidate[i]<0?value:candidate[i]+value;};
            add(1,"phase_shape_seconds");add(1,"phase_setup_seconds");add(2,"phase_baby_seconds");add(3,"phase_ftree_seconds");
            add(4,"phase_main_setup_seconds");add(4,"phase_inverse_setup_seconds");add(5,"phase_giant_loop_seconds");add(6,"phase_descent_seconds");add(7,"phase_accum_seconds");add(8,"phase_finalize_seconds");
            if(std::any_of(candidate.begin(),candidate.end(),[](double n){return n>0;})){best=score;result=candidate;}
        }return result;
    }
};

// Parsing large tune summaries never runs in the render loop or controller tick.
class AsyncStage2Tune {
    std::future<Stage2Tune> pending_;
    std::filesystem::path requested_,pending_path_;
    std::string signature_;
public:
    Stage2Tune value;
    unsigned revision=0;
    void refresh(const std::filesystem::path& path){
        if(requested_!=path){requested_=path;signature_.clear();value=Stage2Tune{};++revision;}
        if(pending_.valid()){
            if(pending_.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;
            auto loaded=pending_.get();if(pending_path_==requested_){value=std::move(loaded);++revision;}
        }
        std::error_code e;const auto stamp=std::filesystem::last_write_time(path,e);
        const auto signature=path.u8string()+"|"+(e?"missing":std::to_string(stamp.time_since_epoch().count()));
        if(signature==signature_)return;signature_=signature;pending_path_=path;
        pending_=std::async(std::launch::async,[path](){Stage2Tune result;result.load(path);return result;});
    }
};

class Stage2TimingHistory {
    using Value=ecm_results::json::Value;
    struct Cursor {std::vector<std::string> tail;size_t count=0;ecmgui::Stage2Status status;PhaseSeconds measured=no_phase_seconds();bool incomplete=false;};
    std::map<int,Cursor> cursors_;
    std::map<std::string,PhaseSeconds> previous_;
    std::filesystem::path file_;
    bool writable_=true;
public:
    std::string notice;
    void load(std::filesystem::path file){
        if(file==file_)return;file_=std::move(file);cursors_.clear();previous_.clear();writable_=true;notice.clear();
        try{if(!std::filesystem::exists(file_))return;if(std::filesystem::file_size(file_)>1048576)throw std::runtime_error("timing cache exceeds 1 MiB");
            std::ifstream in(file_);std::string data((std::istreambuf_iterator<char>(in)),{});auto doc=ecm_results::json::parse(data);
            if(doc.get("schema").text()!="1"||doc.get("records").type!='o')throw std::runtime_error("invalid timing cache");
            for(const auto& entry:doc.get("records").members){if(entry.second.type!='a'||entry.second.items.size()!=stage2_phase_count)throw std::runtime_error("invalid phase times");auto seconds=no_phase_seconds();
                for(size_t i=0;i<stage2_phase_count;++i){const auto& v=entry.second.items[i];size_t used=0;double n=std::stod(v.text(),&used);if(v.type!='d'||used!=v.scalar.size()||!std::isfinite(n)||n< -1)throw std::runtime_error("invalid duration");seconds[i]=n;}previous_[entry.first]=seconds;}
        }catch(const std::exception& e){previous_.clear();writable_=false;notice=e.what();}
    }
    PhaseSeconds lookup(const std::string& scope)const{auto found=previous_.find(scope);return found==previous_.end()?no_phase_seconds():found->second;}
    void observe(const ecmgui::WorkerView& worker,const std::string& device){
        auto& c=cursors_[worker.index];size_t start=0;
        if(!c.tail.empty()){
            bool found=false;
            for(size_t end=worker.raw.size();end>=c.tail.size();--end){if(std::equal(c.tail.begin(),c.tail.end(),worker.raw.begin()+end-c.tail.size())){start=end;found=true;break;}if(end==c.tail.size())break;}
            if(!found){c=Cursor{};c.incomplete=true;}
        }
        for(size_t i=start;i<worker.raw.size();++i){const auto line=ecmgui::parse_line(worker.raw[i]);const auto& event=line.stage2;using Kind=ecmgui::Stage2Event::Kind;
            if(event.kind==Kind::None)continue;
            if(event.kind==Kind::CurveStart){c.measured=no_phase_seconds();c.incomplete=false;}
            if(event.kind==Kind::Phase){const int old=stage2_phase_index(c.status.phase);const auto a=line.plain.find(" previous=");
                if(old>=0&&a!=line.plain.npos){char* end=nullptr;const char* begin=line.plain.c_str()+a+10;const double duration=std::strtod(begin,&end);
                    if(end!=begin&&std::isfinite(duration)&&duration>=0){if(c.measured[old]<0)c.measured[old]=0;c.measured[old]+=duration;}}}
            c.status.apply(event);
            if(event.kind==Kind::CurveDone&&!c.incomplete&&c.status.bits&&c.status.d&&c.status.b2&&std::any_of(c.measured.begin(),c.measured.end(),[](double n){return n>0;})){
                ecmgui::WorkerView scope;scope.exe=worker.exe;scope.device=worker.device;scope.stage2_status=c.status;previous_[stage2_scope(scope,device)]=c.measured;save();
            }
        }
        c.tail.assign(worker.raw.end()-std::min(size_t(8),worker.raw.size()),worker.raw.end());c.count=worker.raw.size();
    }
    void save(){
        if(!writable_||file_.empty())return;
        try{while(previous_.size()>128)previous_.erase(previous_.begin());Value records;records.type='o';
            for(const auto& entry:previous_){Value a;a.type='a';for(double s:entry.second){char n[64];std::snprintf(n,sizeof(n),"%.17g",s);a.items.push_back(Value::number(n));}records.set(entry.first,std::move(a));}
            Value doc;doc.type='o';doc.set("schema",Value::number("1"));doc.set("records",std::move(records));
            if(!file_.parent_path().empty())std::filesystem::create_directories(file_.parent_path());auto tmp=file_;tmp+=".tmp."+std::to_string(GetCurrentProcessId());
            std::ofstream out(tmp,std::ios::binary|std::ios::trunc);out<<doc.dump();out.close();if(!out)throw std::runtime_error("cannot write timing cache");
            if(!MoveFileExW(tmp.c_str(),file_.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("cannot replace timing cache");
        }catch(const std::exception& e){notice=e.what();}
    }
};
}
