#pragma once
#include "ecm_stage2_tune_ecm.h"

namespace ecm_stage2 { namespace tune {
// Full, completed CUDA Stage1 batches. This profile supplies T1 only; it does
// not choose Stage1 kernels or prove that a future Stage1 build has this cost.
struct Stage1Profile {
    Fields profile,device,policy,summary;
    std::vector<Fields> samples;
    static Stage1Profile load(const std::filesystem::path &path) {
        std::error_code error;const auto size=std::filesystem::file_size(path,error);
        if(error || size>16*1048576)throw std::runtime_error("Stage1 tune profile missing or exceeds 16MiB");
        std::ifstream in(path,std::ios::binary);Stage1Profile p;Fields *table=nullptr;
        std::set<std::string> sections,scopes;std::string line;bool first=true;
        auto trim=[](std::string s){const auto a=s.find_first_not_of(" \t\r\n");return a==s.npos?std::string{}:s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);};
        while(std::getline(in,line)) {
            if(first && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);first=false;
            line=trim(line);if(line.empty() || line[0]=='#')continue;
            if(line[0]=='[') {
                if(line.back()!=']')throw std::runtime_error("invalid Stage1 tune table");
                const auto name=line.substr(1,line.size()-2);
                if(!sections.insert(name).second)throw std::runtime_error("duplicate Stage1 tune table");
                if(name=="profile")table=&p.profile;else if(name=="device")table=&p.device;
                else if(name=="policy")table=&p.policy;else if(name=="summary")table=&p.summary;
                else if(name.compare(0,14,"stage1.sample_")==0) {
                    cost::integer(name.substr(14));if(p.samples.size()>=4096)throw std::runtime_error("too many Stage1 tune samples");
                    p.samples.emplace_back();table=&p.samples.back();
                } else throw std::runtime_error("unsupported Stage1 tune table");
            } else {
                const auto eq=line.find('=');if(!table || eq==line.npos)throw std::runtime_error("invalid Stage1 tune row");
                const auto key=trim(line.substr(0,eq)),value=trim(line.substr(eq+1));
                const auto parsed=fields("{\""+key+"\":"+value+"}");
                if(!table->emplace(key,parsed.at(key)).second)throw std::runtime_error("duplicate Stage1 tune key");
            }
        }
        if(!in.eof() || uint(p.profile,"format")!=1 || uint(p.profile,"algorithm_revision")!=1 ||
           required(p.profile,"unit")!="\"process_seconds_per_curve\"" ||
           uint(p.summary,"complete")!=1 || uint(p.summary,"failed") || p.samples.empty() ||
           uint(p.summary,"measured")!=p.samples.size())throw std::runtime_error("incomplete/unsupported Stage1 tune profile");
        if(uint(p.profile,"effort_level")<1 || uint(p.profile,"effort_level")>10 ||
           !uint(p.profile,"repeats") || uint(p.profile,"repeats")>1000 || uint(p.profile,"warmups")!=1)
            throw std::runtime_error("invalid Stage1 tune effort metadata");
        const auto uuid=required(p.device,"uuid_hex");
        if(uuid.size()!=34 || !cost::hex(uuid.substr(1,32),32))throw std::runtime_error("invalid Stage1 tune device");
        for(const char *key:{"sm_major","sm_minor","cuda_runtime","cuda_driver"})uint(p.device,key);
        if(required(p.policy,"algorithm")!="\"ladder\"" || required(p.policy,"backend")!="\"cgbn_montgomery\"" ||
           uint(p.policy,"param") || uint(p.policy,"requested_tpi") || required(p.policy,"exp_cache")!="\"off\"")
            throw std::runtime_error("unsupported Stage1 tune policy");
        for(const auto &s:p.samples) {
            const auto bits=uint(s,"target_bits"),b1=uint(s,"b1"),batch=uint(s,"batch"),repeats=uint(s,"repeats");
            const auto kind=required(s,"modulus_kind"),exponent=required(s,"exponent");
            if(bits<2 || bits>max_input_bits || b1<2 || b1>9007199254740991ull || !batch || batch>1048576 ||
               repeats!=uint(p.profile,"repeats") || uint(s,"checked_curves")<batch || uint(s,"hits") || uint(s,"bad") ||
               (kind!="\"mersenne\"" && kind!="\"generic\"") ||
               (exponent!="\"lcm\"" && exponent!="\"choose12\""))throw std::runtime_error("invalid/unchecked Stage1 tune sample");
            const auto times=array(required(s,"seconds"));
            if(times.size()!=repeats || *std::min_element(times.begin(),times.end())<=0 ||
               std::abs(median(times)-real(s,"median_seconds"))>1e-9*std::max(1.,median(times)) ||
               std::abs(mad(times)-real(s,"mad_seconds"))>1e-9*std::max(1.,mad(times)))
                throw std::runtime_error("inconsistent Stage1 tune statistics");
            if(!scopes.insert(std::to_string(bits)+":"+std::to_string(b1)+":"+std::to_string(batch)+":"+kind+":"+exponent).second)
                throw std::runtime_error("duplicate Stage1 tune scope");
        }
        return p;
    }
    double seconds(const EcmStage2DeviceInfo &d,Word bits,Word b1,Word batch,
                   const std::string &kind,const std::string &exponent)const {
        if(required(device,"uuid_hex")!=std::string("\"")+d.uuid_hex+"\"" || uint(device,"sm_major")!=d.major ||
           uint(device,"sm_minor")!=d.minor || uint(device,"cuda_runtime")!=d.runtime || uint(device,"cuda_driver")!=d.driver)
            throw std::runtime_error("Stage1 tune device/runtime mismatch");
        for(const auto &s:samples)if(uint(s,"target_bits")==bits && uint(s,"b1")==b1 && uint(s,"batch")==batch &&
            required(s,"modulus_kind")==kind && required(s,"exponent")==std::string("\"")+exponent+"\"")return real(s,"median_seconds");
        throw std::runtime_error("no matching Stage1 tune width/B1/batch/arithmetic/exponent scope");
    }
};
} }
