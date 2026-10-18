#pragma once
#include "ecm_result_submission.h"

namespace ecm_results {
// Same canonical slots as the CPU Stage2 result path. Runtime metadata and
// personal identity are not included in the arithmetic-result checksum.
inline std::string internal_json(Submission r,const mpz_t n,uint64_t actual_fft,
                                 const std::string &source_build,const std::string &extra_fields={}) {
    const auto exponent=target_exponent(r,n);const bool hit=!r.factors.empty();
    if(r.when.empty())r.when=utc_timestamp();
    std::ostringstream out;out<<"{\"status\":"<<quoted(hit?"F":"NF");
    if(r.kbnc && !exponent)out<<",\"k\":"<<normalized_integer(r.k)<<",\"b\":"<<r.b<<",\"n\":"<<r.exponent<<",\"c\":"<<r.c;
    else if(exponent)out<<",\"exponent\":"<<exponent;
    else {const auto value=decimal(n);out<<",\"N\":"<<(value.size()>2000?quoted(value):value);}
    out<<",\"worktype\":\"ECM\",\"b1\":"<<r.b1;
    if(r.stage!=1)out<<",\"b2\":"<<r.b2;
    if(r.d)out<<",\"d\":"<<r.d<<",\"poly-size\":"<<r.poly_size;
    out<<",\"curves\":1";
    if(r.edwards)out<<",\"Edwards\":{\"sigma\":"<<r.sigma<<'}';
    else out<<",\"sigma\":"<<r.sigma<<",\"param\":"<<r.param;
    out<<",\"fft-length\":"<<actual_fft;
    if(hit){out<<",\"stage\":"<<r.stage<<",\"factors\":[";for(size_t i=0;i<r.factors.size();++i){if(i)out<<',';out<<quoted(r.factors[i]);}out<<']';}
    if(!r.known.empty()){out<<",\"known-factors\":[";for(size_t i=0;i<r.known.size();++i){if(i)out<<',';out<<quoted(r.known[i]);}out<<']';}
    out<<",\"errors\":{\"invariant\":0},\"program\":{\"name\":"<<quoted(ecm_program::name)<<",\"version\":"<<quoted(ecm_program::version)<<",\"port\":"<<quoted(std::string(ecm_program::host_os)+" "+ecm_program::host_arch)<<'}'
       <<",\"source-build\":"<<quoted(source_build)<<",\"os\":{\"os\":"<<quoted(ecm_program::host_os)<<",\"architecture\":"<<quoted(ecm_program::host_arch)<<"},\"timestamp\":"<<quoted(r.when);
    if(!r.user.empty())out<<",\"user\":"<<quoted(r.user);
    if(!r.computer.empty())out<<",\"computer\":"<<quoted(r.computer);
    if(!r.aid.empty())out<<",\"aid\":"<<quoted(r.aid);
    out<<extra_fields<<'}';
    return with_result_checksum(out.str())+'\n';
}
}
