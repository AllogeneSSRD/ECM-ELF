#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "../diagnostics/order_analysis.h"
#include <gmp.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace ecm_stage2 {
inline std::string json_quote(const std::string &s) {
    std::string out="\"";const char *hex="0123456789abcdef";
    for(unsigned char c:s) {
        if(c=='"' || c=='\\'){out+='\\';out+=(char)c;}
        else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=(char)c;
    }
    return out+'"';
}
inline std::wstring process_quote(const std::wstring &s) {
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:s){if(c==L'\\'){++slashes;continue;}out.append(slashes*(c==L'"'?2:1),L'\\');slashes=0;
        if(c==L'"')out+=L'\\';out+=c;}
    out.append(2*slashes,L'\\');return out+L'"';
}
struct NativeHandle {
    HANDLE value=INVALID_HANDLE_VALUE;
    ~NativeHandle(){if(value && value!=INVALID_HANDLE_VALUE)CloseHandle(value);}
};
inline std::vector<std::string> result_factors(const std::string &json) {
    const std::string key="\"factors\":[";size_t pos=json.find(key);
    if(pos==std::string::npos)throw std::runtime_error("engine result has no factor array");
    pos+=key.size();std::vector<std::string> out;
    for(;;) {
        while(pos<json.size() && std::isspace((unsigned char)json[pos]))++pos;
        if(pos<json.size() && json[pos]==']')return out;
        if(pos>=json.size() || json[pos++]!='"')throw std::runtime_error("invalid engine factor array");
        const size_t begin=pos;
        while(pos<json.size() && std::isdigit((unsigned char)json[pos]))++pos;
        if(pos==begin || pos>=json.size() || json[pos++]!='"')throw std::runtime_error("invalid engine factor");
        out.push_back(json.substr(begin,pos-begin-1));
        while(pos<json.size() && std::isspace((unsigned char)json[pos]))++pos;
        if(pos<json.size() && json[pos]==']')return out;
        if(pos>=json.size() || json[pos++]!=',')throw std::runtime_error("invalid engine factor separator");
    }
}

inline std::string factor_details(const std::string &engine_result,const mpz_t modulus,
                                   const std::filesystem::path &gp,unsigned timeout_seconds,
                                   const std::filesystem::path &directory) {
    using namespace ecm_group_order;
    ecm_diagnostics::Settings settings;settings.gp=gp.u8string();settings.gp_timeout_ms=uint64_t(timeout_seconds)*1000;
    ecm_diagnostics::Session session(settings);
    std::set<std::string> unique_primes;bool complete=true;std::string details="[";size_t ordinal=0;
    for(const auto &raw:result_factors(engine_result)) {
        Integer n(raw);
        if(mpz_cmp_ui(n.v,1)<=0 || mpz_cmp(n.v,modulus)>=0 || !mpz_divisible_p(modulus,n.v))
            throw std::runtime_error("engine reported an invalid factor");
        auto result=session.factor(n);const auto &value=result.value;
        const bool proven_complete=value.status==Status::Complete && value.all_proven;
        complete=complete && proven_complete;
        std::string status=proven_complete?"complete":value.factors.empty()?"unresolved":"partial";
        std::string reason=value.reason;
        if(reason.empty() && !proven_complete)reason="primality_unproven";
        // Diagnostic logging is best effort, never admission for publishing raw hits.
        std::string log;
        if(!result.gp_output.empty())try {
            std::filesystem::create_directories(directory);
            const auto path=directory/(std::to_string(GetCurrentProcessId())+"_"+std::to_string(GetTickCount64())+"_"+std::to_string(ordinal)+".log");
            std::ofstream file(path,std::ios::binary);file<<result.gp_output;if(file)log=path.string();
        }catch(const std::exception &){}
        if(ordinal++)details+=',';
        details+="{\"raw\":"+json_quote(raw)+",\"log\":"+json_quote(log)+",\"backend\":"+json_quote(result.backend)+
            ",\"pari_failure\":"+json_quote(result.pari_failure)+",\"status\":"+json_quote(status)+",\"reason\":"+json_quote(reason)+",\"gp_failure\":"+json_quote(result.gp_failure)+
            ",\"remaining_cofactor\":"+json_quote(value.cofactor.str())+",\"prime_powers\":[";
        for(size_t i=0;i<value.factors.size();++i){
            const auto &part=value.factors[i];if(i)details+=',';
            const bool proven=part.primality==Primality::Proven;
            if(proven)unique_primes.insert(part.prime.str());
            details+="{\"factor\":"+json_quote(part.prime.str())+",\"multiplicity\":"+std::to_string(part.exponent)+
                ",\"proven\":"+(proven?"true":"false")+",\"primality\":"+json_quote(name(part.primality))+"}";
        }
        details+="]}";
    }
    details+=']';std::string primes="[";
    for(const auto &p:unique_primes){if(primes.size()>1)primes+=',';primes+=json_quote(p);}primes+=']';
    return "\"factorization_complete\":"+std::string(complete ? "true" : "false")+
        ",\"prime_factors\":"+primes+",\"factor_analysis\":"+details;
}
} // namespace ecm_stage2
