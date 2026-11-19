#include "host_math.h"
#include <algorithm>
#include <future>
namespace ed_gpu_driver {
bool initialize(Batch &b,const mpz_t n,std::string &error,const ConstructionSink &publish,
                unsigned workers) {
    auto construct_range = [&](unsigned worker) {
        Integer d,px,py,a24,f;
        const auto first=uint32_t(uint64_t(b.count)*worker/workers);
        const auto last=uint32_t(uint64_t(b.count)*(worker+1)/workers);
        for (uint32_t i=first;i<last;++i) {
            const auto rc=construct(d.v,px.v,py.v,a24.v,f.v,b.sigmas[i],n);
            auto* s=b.state.data()+size_t(i)*9;
            if (rc==Setup::Ready) {
                mpz_set(s[0],d.v);mpz_set(s[1],px.v);mpz_set(s[2],py.v);
                mpz_set_ui(s[3],1);mpz_mul(s[4],px.v,py.v);mpz_mod(s[4],s[4],n);
            } else {
                // gcd=N witnesses mathematical degeneracy. Invalid inputs
                // and failed curve-equation checks don't carry that witness.
                if (rc==Setup::Degenerate && !construction_is_degenerate(rc,f.v,n))
                    return i;
                if (rc==Setup::Factor) {
                    b.hit[i]=1;
                    mpz_set(b.factors[i],f.v);
                }
                // Keep the original lane index with the same neutral state
                // used for a factor found during construction.
                mpz_set_ui(s[2],1);
                mpz_set_ui(s[3],1);
            }
            mpz_set_ui(s[6],1);mpz_set_ui(s[7],1);
        }
        return b.count;
    };
    std::vector<std::future<uint32_t>> jobs;
    for (unsigned worker=1;worker<workers;++worker)
        jobs.push_back(std::async(std::launch::async,construct_range,worker));
    auto invalid=construct_range(0);
    for (auto& job:jobs) invalid=std::min(invalid,job.get());
    if (invalid!=b.count) {
        error="invalid Atkin-Morain construction sigma="+std::to_string(b.sigmas[invalid]);
        return false;
    }
    if (publish) for (uint32_t i=0;i<b.count;++i)
        if (b.hit[i] && !publish(i,b.factors[i])) {
            error="cannot publish Edwards construction factor";
            return false;
        }
    return true;
}
}
