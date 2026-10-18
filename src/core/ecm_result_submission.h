#pragma once
#include "ecm_edwards_curve.h"
#include "ecm_program_version.h"
#include "ecm_result_json.h"
#include "generated/p95_fft_reference.h"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ecm_results {
inline std::string quoted(const std::string &s) {
    return json::quote(s);
}
inline std::string decimal(const mpz_t n) {
    std::vector<char> b(mpz_sizeinbase(n,10)+3);mpz_get_str(b.data(),10,n);return b.data();
}
inline std::string normalized_integer(const std::string &text) {
    ecm_edwards::Integer v;
    if(mpz_set_str(v.v,text.c_str(),10))throw std::runtime_error("invalid decimal metadata");
    return decimal(v.v);
}
inline std::string utc_timestamp() {
    const auto now=std::time(nullptr);std::tm t{};
#ifdef _WIN32
    gmtime_s(&t,&now);
#else
    gmtime_r(&now,&t);
#endif
    std::ostringstream out;out<<std::put_time(&t,"%Y-%m-%d %H:%M:%S");return out.str();
}
inline uint32_t reference_fft(uint64_t exponent) {
    for(const auto &entry:p95_fft_capacities)if(exponent<=entry.max_exponent)return entry.length;
    throw std::runtime_error("target exceeds Prime95 FFT reference capacity");
}
struct Submission {
    uint64_t exponent=0,sigma=0,b1=0,b2=0,curves=1;
    uint64_t d=0,poly_size=0,stage2_fft_length=0; // Actual completed Stage2 geometry; zero means unavailable.
    int stage=0;bool edwards=true;
    int param=0;
    bool stage1_only=false,kbnc=false;
    std::string k="1";uint32_t b=2;int32_t c=-1;
    std::vector<std::string> factors,known;
    std::string user,computer,aid,when,publication_task;
};
// Verify original M metadata against the actual modulus. Only exact Mersenne
// inputs are inferred; a cofactor needs its original exponent and known factors.
inline uint64_t target_exponent(const Submission &r,const mpz_t n) {
    ecm_edwards::Integer m,f;
    if(r.kbnc && (r.k!="1" || r.b!=2 || r.c!=-1)) {
        if(!r.exponent || r.exponent>1000000000 || r.b<2 || mpz_set_str(f.v,r.k.c_str(),10) || mpz_sgn(f.v)<=0)
            throw std::runtime_error("invalid k*b^n+c submission target");
        mpz_ui_pow_ui(m.v,r.b,static_cast<unsigned long>(r.exponent));mpz_mul(m.v,m.v,f.v);
        if(r.c<0)mpz_sub_ui(m.v,m.v,static_cast<unsigned long>(-static_cast<int64_t>(r.c)));
        else mpz_add_ui(m.v,m.v,r.c);
        for(const auto &text:r.known) {
            if(text.empty() || text.find_first_not_of("0123456789")!=text.npos || mpz_set_str(f.v,text.c_str(),10) ||
               mpz_cmp_ui(f.v,1)<=0 || !mpz_divisible_p(m.v,f.v))throw std::runtime_error("invalid submission known factor");
            mpz_divexact(m.v,m.v,f.v);
        }
        if(mpz_cmp(m.v,n))throw std::runtime_error("submission target differs from saved modulus");
        return 0;
    }
    if(!r.exponent) {
        if(!r.known.empty())throw std::runtime_error("known factors require the original Mersenne exponent");
        mpz_add_ui(m.v,n,1);
        return mpz_popcount(m.v)==1?mpz_scan1(m.v,0):0;
    }
    if(r.exponent>1000000000)throw std::runtime_error("submission exponent exceeds supported range");
    mpz_set_ui(m.v,1);mpz_mul_2exp(m.v,m.v,static_cast<mp_bitcnt_t>(r.exponent));mpz_sub_ui(m.v,m.v,1);
    for(const auto &text:r.known) {
        if(text.empty() || text.find_first_not_of("0123456789")!=text.npos || mpz_set_str(f.v,text.c_str(),10) ||
           mpz_cmp_ui(f.v,1)<=0 || !mpz_divisible_p(m.v,f.v))throw std::runtime_error("invalid submission known factor");
        mpz_divexact(m.v,m.v,f.v);
    }
    if(mpz_cmp(m.v,n))throw std::runtime_error("submission target differs from saved modulus");
    return r.exponent;
}
inline uint32_t crc32(const std::string &s) {
    uint32_t c=0xffffffffu;
    for(unsigned char b:s){c^=b;for(int i=0;i<8;++i)c=(c>>1)^((c&1)?0xedb88320u:0);}
    return c^0xffffffffu;
}
inline std::string crc_hex(const std::string &s) {
    std::ostringstream out;out<<std::uppercase<<std::hex<<std::setw(8)<<std::setfill('0')<<crc32(s);return out.str();
}
// Compute from the final record, including its source version, model marker,
// actual/reference FFT field and final timestamp. Sealing must call this again.
inline std::string with_result_checksum(const std::string &line) {
    auto row=json::parse(line);if(row.type!='o')throw std::runtime_error("result is not a JSON object");
    if(!row.has("errors"))row.set("errors",json::parse("{\"invariant\":0}"));
    if(!row.has("os"))row.set("os",json::parse("{\"os\":"+quoted(ecm_program::host_os)+",\"architecture\":"+quoted(ecm_program::host_arch)+"}"));
    std::string target=row.get("exponent").text();
    if(target.empty()) {
        target=row.get("N").text();
        if(target.empty() && row.has("k")) {
            ecm_edwards::Integer n,k,f;
            mpz_ui_pow_ui(n.v,std::stoul(row.get("b").text()),std::stoul(row.get("n").text()));
            if(mpz_set_str(k.v,row.get("k").text().c_str(),10))throw std::runtime_error("invalid checksum target");
            mpz_mul(n.v,n.v,k.v);const auto c=std::stoll(row.get("c").text());
            if(c<0)mpz_sub_ui(n.v,n.v,static_cast<unsigned long>(-c));else mpz_add_ui(n.v,n.v,static_cast<unsigned long>(c));
            for(const auto &v:row.get("known-factors").items) {
                if(mpz_set_str(f.v,v.text().c_str(),10)||mpz_cmp_ui(f.v,1)<=0||!mpz_divisible_p(n.v,f.v))throw std::runtime_error("invalid checksum known factor");
                mpz_divexact(n.v,n.v,f.v);
            }
            target=decimal(n.v);
        }
        if(target.empty())throw std::runtime_error("missing checksum target");
        target="N="+target;
    }
    const auto &model=row.has("Edwards")?row.get("Edwards"):row.get("GMP-ECM-param3");
    const auto sigma=(row.has("Edwards")?"E":"")+model.get("sigma").text(row.get("sigma").text());
    const auto b1=row.get("b1").text(),b2=row.get("b2").text();
    const auto bound2=!b2.empty()&&std::stoull(b2)>std::stoull(b1)?b2:std::string{};
    const auto &program=row.get("program"),&os=row.get("os");
    const std::vector<std::string> slots={target,"ECM",json::joined(row.get("factors")),json::joined(row.get("known-factors")),b1,bound2,sigma,
        row.get("fft-length").text(),"invariant:"+row.get("errors").get("invariant").text("0"),
        program.get("name").text(),program.get("version").text(),"","",os.get("os").text(),os.get("architecture").text(),row.get("timestamp").text()};
    std::string canonical;for(size_t i=0;i<slots.size();++i){if(i)canonical+=';';canonical+=slots[i];}
    row.set("checksum",json::parse("{\"version\":1,\"checksum\":"+quoted(crc_hex(canonical))+"}"));
    return row.dump();
}
inline std::string submission_json(const Submission &r,const mpz_t n) {
    const bool hit=!r.factors.empty();
    if(!r.curves || r.b1<2 || (!hit && (r.stage || (r.stage1_only?r.b2!=r.b1:r.b2<=r.b1))) || (hit && r.stage!=1 && r.stage!=2) ||
       (r.stage1_only && (r.stage==2 || r.d || r.poly_size || r.stage2_fft_length)) || (!r.edwards && r.param!=0 && r.param!=2 && r.param!=3))
        throw std::runtime_error("inconsistent ECM submission result");
    if((r.d==0)!=(r.poly_size==0) || (r.d && (r.stage==1 || r.d<6 || r.d%2)))
        throw std::runtime_error("inconsistent Stage2 submission geometry");
    if(r.stage2_fft_length && (r.stage==1 || !r.d))throw std::runtime_error("inconsistent Stage2 FFT metadata");
    ecm_edwards::Integer f;
    for(const auto &text:r.factors)if(mpz_set_str(f.v,text.c_str(),10) || mpz_cmp_ui(f.v,1)<=0 ||
        mpz_cmp(f.v,n)>=0 || !mpz_divisible_p(n,f.v))throw std::runtime_error("invalid submission factor");
    if(!r.aid.empty() && (r.aid.size()!=32 || r.aid.find_first_not_of("0123456789abcdefABCDEF")!=r.aid.npos))
        throw std::runtime_error("invalid submission assignment id");
    const auto exponent=target_exponent(r,n);
    // Generic N follows a conservative zero-padded capacity reference. This is
    // protocol reporting metadata, independent of the engine's actual transform.
    const auto fft=reference_fft(exponent?exponent:2*mpz_sizeinbase(n,2)+128);
    std::ostringstream out;out<<"{\"status\":"<<quoted(hit?"F":"NF");
    if(r.kbnc && !exponent)out<<",\"k\":"<<normalized_integer(r.k)<<",\"b\":"<<r.b<<",\"n\":"<<r.exponent<<",\"c\":"<<r.c;
    else if(exponent)out<<",\"exponent\":"<<exponent;
    else {const auto value=decimal(n);out<<",\"N\":"<<(value.size()>2000?quoted(value):value);}
    out<<",\"worktype\":\"ECM\"";
    if(hit){out<<",\"factors\":[";for(size_t i=0;i<r.factors.size();++i){if(i)out<<',';out<<quoted(r.factors[i]);}out<<']';}
    out<<",\"b1\":"<<r.b1;
    if(r.stage!=1)out<<",\"b2\":"<<r.b2;
    if(r.d)out<<",\"d\":"<<r.d<<",\"poly-size\":"<<r.poly_size;
    if(r.stage2_fft_length)out<<",\"stage2-fft-length\":"<<r.stage2_fft_length;
    if(r.edwards){out<<",\"Edwards\":{";if(hit)out<<"\"sigma\":"<<r.sigma;out<<'}';}
    else if(r.param==3){out<<",\"GMP-ECM-param3\":{";if(hit)out<<"\"sigma\":"<<r.sigma;out<<'}';}
    else if(hit){out<<",\"sigma\":"<<r.sigma;if(r.param)out<<",\"param\":"<<r.param;}
    if(hit)out<<",\"stage\":"<<r.stage;
    out<<",\"curves\":"<<r.curves<<",\"fft-length\":"<<fft;
    out<<",\"program\":{\"name\":"<<quoted(ecm_program::name)<<",\"version\":"<<quoted(ecm_program::version);
#ifdef _WIN64
    out<<",\"port\":4";
#endif
    out<<"},\"timestamp\":"<<quoted(r.when.empty()?utc_timestamp():r.when);
    if(!r.known.empty()){out<<",\"known-factors\":[";for(size_t i=0;i<r.known.size();++i){if(i)out<<',';out<<quoted(r.known[i]);}out<<']';}
    if(!r.user.empty())out<<",\"user\":"<<quoted(r.user);
    if(!r.computer.empty())out<<",\"computer\":"<<quoted(r.computer);
    if(!r.aid.empty())out<<",\"aid\":"<<quoted(r.aid);
    out<<'}';const auto line=with_result_checksum(out.str());
    // An arbitrary long integer has no compact PrimeNet assignment identity.
    // Keep its exact local export rather than preventing Stage1 computation.
    if((exponent || r.kbnc) && line.size()>=2000)throw std::runtime_error("ECM submission exceeds Prime95 JSON message capacity");
    return line+'\n';
}
} // namespace ecm_results
