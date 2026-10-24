#pragma once
#include "../../core/curve/ecm_edwards_curve.h"
#include "../checkpoint/ecm_stage1_checkpoint_runtime.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <algorithm>

namespace ecm_edwards {
inline std::string hex(const mpz_t x) {
    std::vector<char> buf(mpz_sizeinbase(x,16)+3);mpz_get_str(buf.data(),16,x);return buf.data();
}
inline uint64_t hash(const std::string &s) {
    uint64_t h=14695981039346656037ull;
    for(unsigned char ch:s){h^=ch;h*=1099511628211ull;}return h;
}
inline uint64_t save_checksum(const std::string &n,const std::string &x,
                               uint64_t sigma,uint64_t b1,bool factor) {
    return hash("ECM-EDWARDS-AM|1|48LCM|"+n+"|"+x+"|"+std::to_string(sigma)+"|"+
                std::to_string(b1)+"|"+(factor?"FACTOR":"POINT"));
}
inline std::string save_line(const mpz_t n,const mpz_t x,uint64_t sigma,uint64_t b1,bool factor) {
    const auto nh=hex(n),xh=hex(x);
    return "METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA="+std::to_string(sigma)+
        "; B1="+std::to_string(b1)+"; N=0x"+nh+"; X=0x"+xh+"; STATUS="+
        (factor?"FACTOR":"POINT")+"; CHECKSUM="+std::to_string(save_checksum(nh,xh,sigma,b1,factor))+";\n";
}
inline bool publish_save(const std::string &path,const std::string &lines,bool append) {
    const auto temp=ecm_stage1_checkpoint_temp_path(path);
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);if(!out)return false;
    std::error_code error;
    const bool exists=std::filesystem::exists(path,error);
    if(error)return false;
    std::set<std::string> seen;
    const auto key=[](const std::string &line){return line.substr(0,line.find("; PROGRAM="));};
    if(append && exists){
        std::ifstream old(path,std::ios::binary);if(!old)return false;
        std::string line;
        while(std::getline(old,line)){if(!line.empty() && line.back()=='\r')line.pop_back();out<<line<<'\n';seen.insert(key(line));}
        if(old.bad())return false;
    }
    std::istringstream incoming(lines);std::string line;
    std::vector<std::pair<uint64_t,std::string>> batch;bool point_records=true;
    while(std::getline(incoming,line)) {
        const auto at=line.compare(0,10,"METHOD=ECM")==0?line.find("SIGMA="):std::string::npos;uint64_t sigma=0;
        if(at==std::string::npos)point_records=false;
        else try {size_t used=0;sigma=std::stoull(line.substr(at+6),&used);if(!used || at+6+used>=line.size() || line[at+6+used]!=';')return false;}
             catch(const std::exception &){return false;}
        batch.emplace_back(sigma,std::move(line));
    }
    if(point_records)std::stable_sort(batch.begin(),batch.end(),[](const auto &a,const auto &b){return a.first<b.first;});
    for(const auto &record:batch)if(seen.insert(key(record.second)).second)out<<record.second<<'\n';
    out.close();return bool(out) && ecm_stage1_commit_checkpoint(path,temp);
}
// Validate raw extended coordinates; reducing corrupt input would hide errors.
} // namespace ecm_edwards
