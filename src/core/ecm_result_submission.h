#pragma once
#include "ecm_edwards_curve.h"
#include "generated/p95_fft_reference.h"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ecm_results {
inline std::string quoted(const std::string &s) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<c;
    }
    out<<'"';return out.str();
}
inline std::string decimal(const mpz_t n) {
    std::vector<char> b(mpz_sizeinbase(n,10)+3);mpz_get_str(b.data(),10,n);return b.data();
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
    int stage=0;bool edwards=true;
    std::vector<std::string> factors,known;
    std::string user,computer,aid,when;
};
// Verify original M metadata against the actual modulus. Only exact Mersenne
// inputs are inferred; a cofactor needs its original exponent and known factors.
inline uint64_t target_exponent(const Submission &r,const mpz_t n) {
    ecm_edwards::Integer m,f;
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
inline std::string submission_json(const Submission &r,const mpz_t n) {
    const bool hit=!r.factors.empty();
    if(!r.curves || r.b1<2 || (!hit && (r.stage || r.b2<=r.b1)) || (hit && r.stage!=1 && r.stage!=2))
        throw std::runtime_error("inconsistent ECM submission result");
    ecm_edwards::Integer f;
    for(const auto &text:r.factors)if(mpz_set_str(f.v,text.c_str(),10) || mpz_cmp_ui(f.v,1)<=0 ||
        mpz_cmp(f.v,n)>=0 || !mpz_divisible_p(n,f.v))throw std::runtime_error("invalid submission factor");
    if(!r.aid.empty() && (r.aid.size()!=32 || r.aid.find_first_not_of("0123456789abcdefABCDEF")!=r.aid.npos))
        throw std::runtime_error("invalid submission assignment id");
    const auto exponent=target_exponent(r,n);
    // Generic N follows a conservative zero-padded capacity reference. This is
    // protocol reporting metadata; CPU/GPU arithmetic does not execute this FFT.
    const auto fft=reference_fft(exponent?exponent:2*mpz_sizeinbase(n,2)+128);
    std::ostringstream out;out<<"{\"status\":"<<quoted(hit?"F":"NF");
    if(exponent)out<<",\"exponent\":"<<exponent;
    else out<<",\"N\":"<<decimal(n);
    out<<",\"worktype\":\"ECM\"";
    if(hit){out<<",\"factors\":[";for(size_t i=0;i<r.factors.size();++i){if(i)out<<',';out<<quoted(r.factors[i]);}out<<']';}
    out<<",\"b1\":"<<r.b1;
    if(r.stage!=1)out<<",\"b2\":"<<r.b2;
    if(r.edwards){out<<",\"Edwards\":{";if(hit)out<<"\"sigma\":"<<r.sigma;out<<'}';}
    else if(hit)out<<",\"sigma\":"<<r.sigma;
    if(hit)out<<",\"stage\":"<<r.stage;
    out<<",\"curves\":"<<r.curves<<",\"fft-length\":"<<fft;
    // A stable numeric public version and platform code are separate from the
    // executable/source identities stored in the internal JSONL.
    out<<",\"program\":{\"name\":\"ECM-ELF\",\"version\":\"0.1\",\"build\":1,\"port\":4}"
       <<",\"timestamp\":"<<quoted(r.when.empty()?utc_timestamp():r.when);
    if(!r.known.empty()){out<<",\"known-factors\":[";for(size_t i=0;i<r.known.size();++i){if(i)out<<',';out<<quoted(r.known[i]);}out<<']';}
    if(!r.user.empty())out<<",\"user\":"<<quoted(r.user);
    if(!r.computer.empty())out<<",\"computer\":"<<quoted(r.computer);
    if(!r.aid.empty())out<<",\"aid\":"<<quoted(r.aid);
    out<<'}';const auto line=out.str();
    if(line.size()>=2000)throw std::runtime_error("ECM submission exceeds Prime95 JSON message capacity");
    return line+'\n';
}
} // namespace ecm_results
