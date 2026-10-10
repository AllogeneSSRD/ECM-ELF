#pragma once
#include "ecm_stage2_tune_ecm.h"

namespace ecm_stage2 { namespace tune {
struct NttProfile {
    Fields profile,device,policy,environment,summary;
    std::map<std::pair<Word,Word>,Fields> samples;
    static NttProfile load(const std::filesystem::path &path) {
        std::error_code error;const auto size=std::filesystem::file_size(path,error);
        if(error || size>64*1048576)throw std::runtime_error("NTT profile missing or exceeds 64MiB");
        std::ifstream in(path,std::ios::binary);NttProfile out;Fields *current=nullptr;
        std::set<std::string> sections;std::map<std::string,Fields> tables;std::string line;bool first=true;
        auto trim=[](std::string s){auto a=s.find_first_not_of(" \t\r\n");return a==s.npos?std::string{}:s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);};
        while(std::getline(in,line)) {
            if(first && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);first=false;
            line=trim(line);if(line.empty() || line[0]=='#')continue;
            if(line[0]=='[') {
                if(line.back()!=']')throw std::runtime_error("invalid NTT table");
                const auto name=line.substr(1,line.size()-2);
                if(!sections.insert(name).second)throw std::runtime_error("duplicate NTT table");
                if(name=="profile")current=&out.profile;else if(name=="device")current=&out.device;
                else if(name=="policy")current=&out.policy;else if(name=="policy.environment")current=&out.environment;
                else if(name=="summary")current=&out.summary;
                else if(name.compare(0,11,"ntt.length_")==0) {
                    if(tables.size()>=65536)throw std::runtime_error("too many NTT shapes");current=&tables[name];
                } else throw std::runtime_error("unsupported NTT table");
            } else {
                const auto eq=line.find('=');if(!current || eq==line.npos)throw std::runtime_error("invalid NTT row");
                const auto key=trim(line.substr(0,eq)),value=trim(line.substr(eq+1));
                const auto parsed=fields("{\""+key+"\":"+value+"}");
                if(!current->emplace(key,parsed.at(key)).second)throw std::runtime_error("duplicate NTT field");
            }
        }
        if(!in.eof() || uint(out.profile,"format")!=2 || required(out.profile,"unit")!="\"field_convolution\"" ||
           uint(out.summary,"failed") || required(out.summary,"usable")!="true" ||
           required(out.policy,"accounting")!=std::string("\"")+ntt_cost_accounting+"\"")
            throw std::runtime_error("incomplete/unsupported NTT profile");
        const auto lo=uint(out.profile,"min_log2"),hi=uint(out.profile,"max_log2"),repeats=uint(out.profile,"repeats");
        if(lo<3 || hi>27 || lo>hi || !repeats || repeats>1000)throw std::runtime_error("invalid NTT grid");
        std::set<Word> slices;
        for(auto value:array(required(out.profile,"slices"))) {
            if(value<1 || value>65535 || value!=std::floor(value) || !slices.insert((Word)value).second)
                throw std::runtime_error("invalid/duplicate NTT slices");
        }
        if(slices.empty() || slices.size()>64 || tables.size()!=(hi-lo+1)*slices.size())
            throw std::runtime_error("incomplete NTT rectangular grid");
        Word skipped=0;
        for(auto &entry:tables) {
            auto &s=entry.second;const auto key=ntt_scope(s);Word log=0;for(auto n=key.first;n>1;n>>=1)++log;
            if(entry.first!=ntt_section(s) || key.first<8 || (key.first&(key.first-1)) || log<lo || log>hi ||
               uint(s,"log2_length")!=log || !slices.count(key.second))throw std::runtime_error("NTT shape outside declared grid");
            if(required(s,"status")=="\"skipped_memory\""){++skipped;continue;}
            if(s.count("repeats") && uint(s,"repeats")!=repeats)throw std::runtime_error("NTT repetition mismatch");
            s["repeats"]=std::to_string(repeats);validate_ntt_measurement(s);
            if(!out.samples.emplace(key,s).second)throw std::runtime_error("duplicate NTT measurement");
        }
        if(out.samples.empty() || uint(out.summary,"measured")!=out.samples.size() || uint(out.summary,"skipped")!=skipped)
            throw std::runtime_error("inconsistent NTT summary");
        return out;
    }
    bool matches(const EcmProfile &ecm)const {
        for(const char *key:{"uuid_hex","sm_major","sm_minor","cuda_runtime","cuda_driver","gl_fixed_mode","outer_unroll_u"})
            if(required(device,key)!=required(ecm.device,key))return false;
        if(uint(device,"gl_add_sub_mask")!=uint(ecm.device,"add_sub_mask") || environment.empty() || environment!=ecm.environment)return false;
        for(const auto &value:environment)cost::integer(value.second);
        return true;
    }
};
inline void attach_ntt_profiles(EcmProfile &ecm,const std::vector<NttProfile> &profiles) {
    if(profiles.empty())return;
    for(const auto &profile:profiles) {
        if(!profile.matches(ecm))throw std::runtime_error("NTT/full ECM device or backend/environment mismatch");
        for(const auto &entry:profile.samples) {
            if(ecm.ntt_samples.size()>=65536)throw std::runtime_error("embedded NTT shape limit exceeded");
            if(!ecm.ntt_samples.emplace(entry.first,entry.second).second)throw std::runtime_error("duplicate measured NTT shape across inputs");
        }
    }
    ecm.profile["format"]="4";ecm.profile["component_model"]=std::string("\"")+component_prediction_model+"\"";
    ecm.policy.erase("environment");if(!ecm.profile.count("max_batches"))ecm.profile["max_batches"]="0";
    ecm.ntt_policy=ntt_packing_policy(ecm.environment);ecm.summary["ntt_measured"]=std::to_string(ecm.ntt_samples.size());
}
} }
