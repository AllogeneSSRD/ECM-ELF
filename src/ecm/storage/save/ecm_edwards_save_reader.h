#pragma once
#include "ecm_edwards_io.h"
#include <algorithm>
#include <charconv>
#include <map>
#include <stdexcept>

namespace ecm_edwards {
inline std::string trim(std::string s) {
    const auto a=s.find_first_not_of(" \r\n\t"),b=s.find_last_not_of(" \r\n\t");
    return a==s.npos?"":s.substr(a,b-a+1);
}
inline uint64_t decimal(const std::string &s) {
    uint64_t n=0;const auto r=std::from_chars(s.data(),s.data()+s.size(),n);
    if(s.empty() || r.ec!=std::errc() || r.ptr!=s.data()+s.size())
        throw std::runtime_error("expected unsigned decimal integer: "+s);
    return n;
}
struct SavedCurve {
    std::string n,x;uint64_t sigma=0,b1=0,line=0;bool factor=false;
};
inline SavedCurve parse_saved_curve(std::string text) {
    if(text.compare(0,3,"\xef\xbb\xbf")==0)text.erase(0,3);
    if(text.find('\0')!=text.npos)throw std::runtime_error("binary saves are unsupported");
    std::map<std::string,std::string> fields;
    size_t start=0;
    while(start<text.size()) {
        const auto end=text.find(';',start);auto token=trim(text.substr(start,end==text.npos?end:end-start));
        start=end==text.npos?text.size():end+1;if(token.empty())continue;
        auto eq=token.find('=');if(eq==token.npos)throw std::runtime_error("missing '=' in save field");
        auto key=trim(token.substr(0,eq));
        if(!fields.emplace(key,trim(token.substr(eq+1))).second)throw std::runtime_error("duplicate save field: "+key);
    }
    for(const auto *key:{"METHOD","VERSION","SCALAR","SIGMA","B1","N","X","STATUS","CHECKSUM"})
        if(!fields.count(key) || fields[key].empty())throw std::runtime_error(std::string("missing save field: ")+key);
    if(fields["METHOD"]!="ECM-EDWARDS-AM" || fields["VERSION"]!="1" || fields["SCALAR"]!="48LCM" || fields.count("PARAM"))
        throw std::runtime_error("expected Edwards AM VERSION=1 SCALAR=48LCM, without PARAM");
    SavedCurve r;r.sigma=decimal(fields["SIGMA"]);r.b1=decimal(fields["B1"]);
    if(!r.sigma || r.b1<2 || r.b1>(1ull<<53))throw std::runtime_error("invalid SIGMA or B1 (B1 must be exactly representable by libecm)");
    if(fields["STATUS"]!="POINT" && fields["STATUS"]!="FACTOR")throw std::runtime_error("invalid save STATUS");
    r.factor=fields["STATUS"]=="FACTOR";
    Integer n,x,z;
    if(mpz_set_str(n.v,fields["N"].c_str(),0) || mpz_cmp_ui(n.v,3)<=0 || !mpz_odd_p(n.v))throw std::runtime_error("invalid odd N > 3");
    if(mpz_set_str(x.v,fields["X"].c_str(),0) || mpz_sgn(x.v)<0 || mpz_cmp(x.v,n.v)>=0)throw std::runtime_error("X outside [0,N)");
    if(fields.count("Z") && (mpz_set_str(z.v,fields["Z"].c_str(),0) || mpz_cmp_ui(z.v,1)))throw std::runtime_error("save must be normalized (Z=1)");
    r.n=hex(n.v);r.x=hex(x.v);
    if(decimal(fields["CHECKSUM"])!=save_checksum(r.n,r.x,r.sigma,r.b1,r.factor))throw std::runtime_error("save CHECKSUM mismatch");
    if(r.factor && (mpz_cmp_ui(x.v,1)<=0 || !mpz_divisible_p(n.v,x.v)))throw std::runtime_error("invalid saved factor");
    return r;
}
inline std::vector<SavedCurve> read_saved_curves(const std::string &path) {
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open save: "+path);
    std::vector<SavedCurve> result;std::string line;uint64_t number=0;
    while(std::getline(in,line)) {
        ++number;const auto t=trim(line);if(t.empty() || t[0]=='#' || t[0]==';')continue;
        try{auto r=parse_saved_curve(line);r.line=number;result.push_back(std::move(r));}
        catch(const std::exception &e){throw std::runtime_error(path+":"+std::to_string(number)+": "+e.what());}
    }
    if(in.bad())throw std::runtime_error("save read failed");
    if(result.empty())throw std::runtime_error("save contains no curves");return result;
}
} // namespace ecm_edwards
