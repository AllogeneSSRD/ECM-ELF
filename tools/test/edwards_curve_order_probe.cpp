#include "../../src/ecm/curve/edwards.h"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: edwards_curve_order_probe PRIME SIGMA");
        ecm_edwards::Integer n,d,x,y,a24,f;
        if(mpz_set_str(n.v,argv[1],10))throw std::runtime_error("invalid modulus");
        const std::string text=argv[2];size_t end=0;
        const uint64_t sigma=std::stoull(text,&end);
        if(text.empty() || text[0]<'0' || text[0]>'9' || end!=text.size())throw std::runtime_error("invalid sigma");
        const auto status=ecm_edwards::construct(d.v,x.v,y.v,a24.v,f.v,sigma,n.v);
        if(status!=ecm_edwards::Setup::Ready){
            std::printf("STATUS|%s\n",status==ecm_edwards::Setup::Factor?"factor":"degenerate");
            gmp_printf("FACTOR|%Zd\n",f.v);return 0;
        }
        std::puts("STATUS|ready");
        gmp_printf("D|%Zd\nEDWARDS_X|%Zd\nEDWARDS_Y|%Zd\nA24|%Zd\n",d.v,x.v,y.v,a24.v);
        return 0;
    } catch(const std::exception &error) {
        std::fprintf(stderr,"%s\n",error.what());return 1;
    }
}
