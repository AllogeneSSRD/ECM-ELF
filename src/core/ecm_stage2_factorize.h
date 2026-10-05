#pragma once
#include <windows.h>
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
    std::set<std::string> unique_primes;bool complete=true;std::string details="[";
    size_t ordinal=0;
    for(const auto &raw:result_factors(engine_result)) {
        mpz_t value,reconstructed,prime,power;mpz_inits(value,reconstructed,prime,power,nullptr);
        if(mpz_set_str(value,raw.c_str(),10) || mpz_cmp_ui(value,1)<=0 || mpz_cmp(value,modulus)>=0 ||
           !mpz_divisible_p(modulus,value)) {
            mpz_clears(value,reconstructed,prime,power,nullptr);
            throw std::runtime_error("engine reported an invalid factor");
        }
        std::filesystem::create_directories(directory);
        const auto stem=std::to_string(GetCurrentProcessId())+"_"+std::to_string(GetTickCount64())+"_"+std::to_string(++ordinal);
        const auto script=directory/(stem+".gp"),log=directory/(stem+".log");
        std::ofstream out(script,std::ios::binary);
        out<<"ecm_split(n)={my(f=factor(n),v=1);for(i=1,matsize(f)[1],"
              "if(!isprime(f[i,1]),error(\"unproven factor\"));v*=f[i,1]^f[i,2];"
              "print(\"FACTOR \",f[i,1],\" \",f[i,2]));"
              "if(v!=n,error(\"factor product\"));print(\"OK\")};\necm_split("<<raw<<");\nquit();\n";
        out.close();if(!out)throw std::runtime_error("cannot write factor analysis script");
        SECURITY_ATTRIBUTES attributes{sizeof(attributes),nullptr,TRUE};NativeHandle output,input,process,thread;
        output.value=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&attributes,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        input.value=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&attributes,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        std::string failure;DWORD exit_code=1;
        if(output.value==INVALID_HANDLE_VALUE || input.value==INVALID_HANDLE_VALUE)failure="log_open_failed";
        else {
            STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES;
            startup.hStdOutput=startup.hStdError=output.value;startup.hStdInput=input.value;
            PROCESS_INFORMATION info{};auto command=process_quote(gp.wstring())+L" -q -f "+process_quote(script.wstring());
            if(!CreateProcessW(nullptr,command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,nullptr,&startup,&info))failure="gp_launch_failed";
            else {
                process.value=info.hProcess;thread.value=info.hThread;
                const auto wait=WaitForSingleObject(process.value,timeout_seconds*1000);
                if(wait!=WAIT_OBJECT_0) {
                    TerminateProcess(process.value,2);WaitForSingleObject(process.value,5000);
                    failure=wait==WAIT_TIMEOUT ? "gp_timeout" : "gp_wait_failed";
                } else if(!GetExitCodeProcess(process.value,&exit_code) || exit_code)failure="gp_exit_failed";
            }
        }
        if(output.value!=INVALID_HANDLE_VALUE){CloseHandle(output.value);output.value=INVALID_HANDLE_VALUE;}
        std::vector<std::pair<std::string,unsigned long>> parts;bool ok=false;
        mpz_set_ui(reconstructed,1);
        if(failure.empty()) {
            std::error_code error;const auto bytes=std::filesystem::file_size(log,error);
            if(error || bytes>2*1024*1024)failure="gp_output_invalid";
            else {
                std::ifstream input_log(log);std::string line;
                while(std::getline(input_log,line)) {
                    line.erase(std::remove(line.begin(),line.end(),'\r'),line.end());
                    if(line=="OK")ok=true;
                    if(line.compare(0,7,"FACTOR "))continue;
                    std::istringstream row(line.substr(7));std::string p,extra;unsigned long e=0;
                    if(!(row>>p>>e) || row>>extra || !e || e>8192 || p.empty() ||
                       p.size()>2500 || p.find_first_not_of("0123456789")!=p.npos ||
                       mpz_set_str(prime,p.c_str(),10) || mpz_cmp_ui(prime,2)<0 || mpz_cmp(prime,value)>0 ||
                       (mpz_sizeinbase(prime,2)-1)*e>mpz_sizeinbase(value,2)) {
                        failure="gp_factor_invalid";break;
                    }
                    for(const auto &old:parts)if(old.first==p)failure="gp_duplicate_factor";
                    if(!failure.empty())break;
                    mpz_pow_ui(power,prime,e);mpz_mul(reconstructed,reconstructed,power);
                    if(mpz_cmp(reconstructed,value)>0){failure="gp_factor_product_exceeded";break;}
                    parts.emplace_back(p,e);
                }
                if(!input_log.eof() && !input_log)failure="gp_output_read_failed";
                if(!ok || mpz_cmp(reconstructed,value))failure="gp_factorization_incomplete";
            }
        }
        if(ordinal>1)details+=',';
        details+="{\"raw\":"+json_quote(raw)+",\"log\":"+json_quote(log.string())+",\"status\":";
        if(!failure.empty()) {complete=false;details+="\"unresolved\",\"reason\":"+json_quote(failure)+",\"prime_powers\":[]}";}
        else {
            details+="\"complete\",\"prime_powers\":[";
            for(size_t i=0;i<parts.size();++i) {
                if(i)details+=',';unique_primes.insert(parts[i].first);
                details+="{\"factor\":"+json_quote(parts[i].first)+",\"multiplicity\":"+std::to_string(parts[i].second)+",\"proven\":true}";
            }
            details+="]}";
        }
        mpz_clears(value,reconstructed,prime,power,nullptr);
    }
    details+=']';std::string primes="[";
    for(const auto &p:unique_primes){if(primes.size()>1)primes+=',';primes+=json_quote(p);}primes+=']';
    return "\"factorization_complete\":"+std::string(complete ? "true" : "false")+
        ",\"prime_factors\":"+primes+",\"factor_analysis\":"+details;
}
} // namespace ecm_stage2
