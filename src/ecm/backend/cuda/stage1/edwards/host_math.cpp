#include "host_math.h"
namespace ed_gpu_driver {
bool initialize(Batch &b,const mpz_t n,std::string &error,const ConstructionSink &publish) {
    Integer d,px,py,a24,f;
    for(uint32_t i=0;i<b.count;++i) {
        const auto rc=construct(d.v,px.v,py.v,a24.v,f.v,b.sigmas[i],n);
        auto *s=b.state.data()+size_t(i)*9;
        if(rc==Setup::Degenerate){error="degenerate Atkin-Morain sigma="+std::to_string(b.sigmas[i]);return false;}
        if(rc==Setup::Factor) {
            b.hit[i]=1;mpz_set(b.factors[i],f.v);
            if(publish && !publish(i,f.v)){error="cannot publish Edwards construction factor";return false;}
            // A harmless identity lane keeps batch indexing stable.
            mpz_set_ui(s[2],1);mpz_set_ui(s[3],1);
        } else {
            mpz_set(s[0],d.v);mpz_set(s[1],px.v);mpz_set(s[2],py.v);
            mpz_set_ui(s[3],1);mpz_mul(s[4],px.v,py.v);mpz_mod(s[4],s[4],n);
        }
        mpz_set_ui(s[6],1);mpz_set_ui(s[7],1);
    }
    return true;
}
}
