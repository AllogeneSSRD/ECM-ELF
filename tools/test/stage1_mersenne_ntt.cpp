// End-to-end Suyama PARAM0 Stage1 driver and independent GMP oracle.
#include "../../src/ecm/stage1/montgomery/cuda/ntt/ladder.h"
#include "ecm/platform/program_version.h"
#include "ecm/stage1/scalar.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
struct Z {
    mpz_t v;
    Z() { mpz_init(v); }
    ~Z() { mpz_clear(v); }
    Z(const Z&)=delete;
};
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
std::string digits(mpz_srcptr z, int base) {
    std::string result(mpz_sizeinbase(z,base)+2,'\0');
    mpz_get_str(result.data(),base,z);
    result.resize(std::strlen(result.c_str()));
    return result;
}

void point_step(std::array<Z,6>& state, unsigned bit, mpz_srcptr n) {
    auto& xd=state[bit*2].v; auto& zd=state[bit*2+1].v;
    auto& xa=state[(1-bit)*2].v; auto& za=state[(1-bit)*2+1].v;
    Z a,b,c,d,aa,bb,e,da,cb,t;
    mpz_add(a.v,xd,zd); mpz_sub(b.v,xd,zd);
    mpz_add(c.v,xa,za); mpz_sub(d.v,xa,za);
    mpz_mul(da.v,d.v,a.v); mpz_mod(da.v,da.v,n);
    mpz_mul(cb.v,c.v,b.v); mpz_mod(cb.v,cb.v,n);
    mpz_add(t.v,da.v,cb.v); mpz_mul(t.v,t.v,t.v); mpz_mod(xa,t.v,n);
    mpz_sub(t.v,da.v,cb.v); mpz_mul(t.v,t.v,t.v); mpz_mod(t.v,t.v,n);
    mpz_mul(t.v,t.v,state[5].v); mpz_mod(za,t.v,n);
    mpz_mul(aa.v,a.v,a.v); mpz_mod(aa.v,aa.v,n);
    mpz_mul(bb.v,b.v,b.v); mpz_mod(bb.v,bb.v,n);
    mpz_mul(t.v,aa.v,bb.v); mpz_mod(xd,t.v,n);
    mpz_sub(e.v,aa.v,bb.v); mpz_mod(e.v,e.v,n);
    mpz_mul(t.v,state[4].v,e.v); mpz_add(t.v,t.v,bb.v); mpz_mod(t.v,t.v,n);
    mpz_mul(t.v,t.v,e.v); mpz_mod(zd,t.v,n);
}

bool inverse(mpz_ptr out, mpz_srcptr value, mpz_srcptr n, uint64_t sigma) {
    if (mpz_invert(out,value,n)) return true;
    mpz_gcd(out,value,n);
    if (mpz_cmp_ui(out,1)>0 && mpz_cmp(out,n)<0)
        std::printf("stage1_curve: sigma=%llu status=construction_factor factor=%s\n",
                    (unsigned long long)sigma,digits(out,10).c_str());
    else std::printf("stage1_curve: sigma=%llu status=degenerate_construction\n",(unsigned long long)sigma);
    return false;
}

bool init(std::array<Z,6>& s, uint64_t seed, mpz_srcptr n) {
    Z u,v,sigma,t,a,b,aa,bb,e,den,inv;
    mpz_import(sigma.v,1,1,sizeof(seed),0,0,&seed);
    mpz_mul(u.v,sigma.v,sigma.v); mpz_sub_ui(u.v,u.v,5);
    mpz_mul_ui(v.v,sigma.v,4);
    mpz_powm_ui(s[0].v,u.v,3,n); mpz_powm_ui(s[1].v,v.v,3,n);
    mpz_mul(den.v,s[0].v,v.v); mpz_mul_ui(den.v,den.v,16);
    if (!inverse(inv.v,den.v,n,seed)) return false;
    mpz_sub(t.v,v.v,u.v); mpz_powm_ui(s[4].v,t.v,3,n);
    mpz_mul_ui(t.v,u.v,3); mpz_add(t.v,t.v,v.v);
    mpz_mul(s[4].v,s[4].v,t.v); mpz_mul(s[4].v,s[4].v,inv.v); mpz_mod(s[4].v,s[4].v,n);
    mpz_sub_ui(t.v,s[4].v,1); mpz_mul(t.v,t.v,s[4].v);
    if (!inverse(inv.v,t.v,n,seed) || !inverse(inv.v,s[1].v,n,seed)) return false;
    mpz_mul(s[5].v,s[0].v,inv.v); mpz_mod(s[5].v,s[5].v,n);
    mpz_add(a.v,s[0].v,s[1].v); mpz_sub(b.v,s[0].v,s[1].v);
    mpz_mul(aa.v,a.v,a.v); mpz_mod(aa.v,aa.v,n);
    mpz_mul(bb.v,b.v,b.v); mpz_mod(bb.v,bb.v,n);
    mpz_sub(e.v,aa.v,bb.v);
    mpz_mul(s[2].v,aa.v,bb.v); mpz_mod(s[2].v,s[2].v,n);
    mpz_mul(t.v,s[4].v,e.v); mpz_add(t.v,t.v,bb.v);
    mpz_mul(s[3].v,e.v,t.v); mpz_mod(s[3].v,s[3].v,n);
    return true;
}

uint64_t num(const char* s, uint64_t lo, uint64_t hi) {
    size_t end=0;
    const std::string arg(s);
    if (arg.empty() || arg.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("expected unsigned decimal input");
    const auto x=std::stoull(arg,&end);
    if (x<lo || x>hi) throw std::invalid_argument("numeric argument out of range");
    return x;
}
void values(const char* name, const std::vector<double>& ms) {
    std::printf(" %s=",name);
    for (size_t i=0;i<ms.size();++i) std::printf("%s%.6f",i?",":"",ms[i]);
}

auto run_ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
                const std::vector<uint8_t>& bits, unsigned window, unsigned samples, bool weighted,
                unsigned length, unsigned tile = 0, unsigned dataflow = 0, unsigned resume = 0,
                bool square = false) {
    if (!length && !dataflow && !resume && !square)
        return ecm_mersenne_ntt::ladder(device,p,curves,initial,bits,window,samples,weighted);
    std::vector<uint32_t> packed((bits.size()+31)/32);
    for (size_t i=0; i<bits.size(); ++i) {
        const size_t bit=bits.size()-1-i;
        packed[bit/32] |= uint32_t(bits[i]) << (bit%32);
    }
    ecm_stage1::Stage1NttOptions options;
    options.length=length; options.tile=tile; options.window=window; options.weighted=weighted;
    ecm_mersenne_ntt::Control control;
    control.experimental_dataflow=dataflow;
    control.resume=resume;
    control.modular_square_probe=square;
    return ecm_mersenne_ntt::ladder(device,p,curves,initial,packed,unsigned(bits.size()),options,control,samples);
}

int check_carries(int device, unsigned p, bool weighted, unsigned length, unsigned dataflow, bool square) {
    // Deliberately non-curve states: stress the arithmetic independently of
    // Suyama's distribution (M-1+1, overflowing top limb, equal differences).
    constexpr unsigned count=4;
    const unsigned words=(p+31)/32;
    const size_t stride=size_t(words)*count;
    Z n;
    mpz_setbit(n.v,p); mpz_sub_ui(n.v,n.v,1);
    std::array<std::array<Z,6>,count> states;
    std::vector<uint32_t> input(6*stride);
    for (unsigned curve=0;curve<count;++curve) {
        for (unsigned slot=0;slot<6;++slot) {
            auto* z=states[curve][slot].v;
            if (curve==0) mpz_sub_ui(z,n.v,1);
            if (curve==1) { if (slot%2) mpz_set_ui(z,1); else mpz_sub_ui(z,n.v,1); }
            if (curve==2) { mpz_setbit(z,p-1); mpz_add_ui(z,z,slot); }
            if (curve==3) mpz_set_ui(z,slot%2);
            mpz_export(input.data()+slot*stride+curve*words,nullptr,-1,4,0,0,z);
        }
    }
    const std::vector<uint8_t> bits={0,1,0,0,1,1,1,0,0};
    const auto graph=run_ladder(device,p,count,input,bits,4,2,weighted,length,0,dataflow,0,square);
    const auto direct=run_ladder(device,p,count,input,bits,0,1,weighted,length,0,dataflow,0,square);
    if (graph.points!=direct.points) throw std::runtime_error("Graph/direct carry check differed");
    if (dataflow) {
        constexpr unsigned split=4;
        const std::vector<uint8_t> prefix(bits.begin(),bits.begin()+split);
        for (bool experimental_first : {false,true}) {
            const auto partial=run_ladder(device,p,count,input,prefix,4,1,weighted,length,0,
                                          experimental_first ? dataflow : 0,0,square);
            auto restarted=input;
            std::copy(partial.points.begin(),partial.points.end(),restarted.begin());
            const auto resumed=run_ladder(device,p,count,restarted,bits,4,1,weighted,length,0,
                                          experimental_first ? 0 : dataflow,split,square);
            if (resumed.points!=graph.points) throw std::runtime_error("cross-dataflow resume differed");
        }
    }
    for (unsigned curve=0;curve<count;++curve) {
        for (auto bit:bits) {
            if (square) {
                auto& x = states[curve][0].v;
                mpz_mul(x,x,x);
                mpz_mod(x,x,n.v);
            } else point_step(states[curve],bit,n.v);
        }
        Z got;
        for (unsigned slot=0;slot<4;++slot) {
            mpz_import(got.v,words,-1,4,0,0,graph.points.data()+slot*stride+curve*words);
            if (mpz_cmp(got.v,states[curve][slot].v)) throw std::runtime_error("GMP carry check differed");
        }
    }
    std::printf("stage1_ntt_%s: p=%u device=%d curves=%u bits=%zu bad=0 length=%u\n",
                square ? "squares" : "carries",p,device,count,bits.size(),graph.length);
    if (dataflow) std::printf("stage1_ntt_dataflow: requested=%u actual=%u\n",dataflow,graph.dataflow);
    return 0;
}
} // namespace

int main(int argc,char** argv) {
    const bool weighted = argc > 1 && std::strcmp(argv[1], "--weighted") == 0;
    if (weighted) { ++argv; --argc; }
    unsigned length=0;
    if (argc > 2 && !std::strcmp(argv[1],"--length")) {
        try { length=unsigned(num(argv[2],2,65536)); }
        catch (const std::exception& e) { std::fprintf(stderr,"stage1_ntt: %s\n",e.what()); return 2; }
        argv+=2; argc-=2;
    }
    unsigned dataflow=0;
    if (argc > 2 && !std::strcmp(argv[1],"--dataflow")) {
        try { dataflow=unsigned(num(argv[2],0,1023)); }
        catch (const std::exception& e) { std::fprintf(stderr,"stage1_ntt: %s\n",e.what()); return 2; }
        argv+=2; argc-=2;
    }
    const bool square = argc==4 && !std::strcmp(argv[1],"--check-squares");
    if (argc==4 && (square || !std::strcmp(argv[1],"--check-carries"))) {
        try { return check_carries(int(num(argv[2],0,63)),unsigned(num(argv[3],32,262144)), weighted,length,dataflow,square); }
        catch (const std::exception& e) { std::fprintf(stderr,"stage1_ntt: %s\n",e.what()); return 2; }
    }
    if (argc<9 || argc>10) {
        std::fprintf(stderr,"stage1_mersenne_ntt [--weighted] [--length N] [--dataflow 0..1023] device p B1 sigma curves samples tile window [print-x]\n"
                            "or use the same options with --check-carries|--check-squares device p\n"
                            "M=2^p-1, PARAM0, LCM, 1..%u curves, window=0 direct or 1..64 Graph\n",
                            ecm_stage1::stage1_ntt_max_curves);
        return 2;
    }
    try {
        const int device=int(num(argv[1],0,63));
        const unsigned p=unsigned(num(argv[2],32,262144)), b1=unsigned(num(argv[3],2,1000000));
        const uint64_t sigma=num(argv[4],6,UINT64_MAX);
        const unsigned curves=unsigned(num(argv[5],1,ecm_stage1::stage1_ntt_max_curves));
        const unsigned samples=unsigned(num(argv[6],1,20));
        const unsigned tile=unsigned(num(argv[7],1,12)), window=unsigned(num(argv[8],0,64));
        if (sigma>UINT64_MAX-(curves-1) || (argc==10 && std::strcmp(argv[9],"print-x")))
            throw std::invalid_argument("invalid sigma range or output option");
        _putenv_s("NTT_FUSE_T",std::to_string(tile).c_str());
        _putenv_s("NTT_FUSE_WARP_TAIL","1");
        _putenv_s("NTT_FUSE_M","4");
        _putenv_s("NTT_FUSE_COOP_OUTER","0");
        const auto setup=Clock::now();
        Z n, scalar;
        mpz_setbit(n.v,p); mpz_sub_ui(n.v,n.v,1);
        if (!ecm_build_lcm_exponent(scalar.v,b1,1)) throw std::runtime_error("LCM failed");
        std::vector<uint8_t> bits;
        for (size_t i=mpz_sizeinbase(scalar.v,2)-1;i--;) bits.push_back(uint8_t(mpz_tstbit(scalar.v,i)));
        const unsigned words=(p+31)/32;
        const size_t stride=size_t(words)*curves;
        std::vector<uint32_t> initial(6*stride);
        std::vector<uint64_t> active;
        for (unsigned lane=0;lane<curves;++lane) {
            std::array<Z,6> s;
            if (!init(s,sigma+lane,n.v)) continue;
            for (unsigned slot=0;slot<6;++slot)
                mpz_export(initial.data()+slot*stride+active.size()*words,nullptr,-1,4,0,0,s[slot].v);
            active.push_back(sigma+lane);
        }
        if (active.empty()) return 0;
        const size_t used=active.size()*words;
        std::vector<uint32_t> compact(6*used);
        for (unsigned j=0;j<6;++j) std::copy_n(initial.data()+j*stride,used,compact.data()+j*used);
        const double host_setup=seconds(setup);
        const auto run=run_ladder(device,p,unsigned(active.size()),compact,bits,window,samples,weighted,length,tile,dataflow);
        double oracle_seconds=0;
        for (size_t lane=0;lane<active.size();++lane) {
            std::array<Z,6> s;
            for (unsigned j=0;j<6;++j) mpz_import(s[j].v,words,-1,4,0,0,compact.data()+j*used+lane*words);
            const auto begin=Clock::now();
            for (auto bit:bits) point_step(s,bit,n.v);
            oracle_seconds+=seconds(begin);
            Z got,g,x;
            for (unsigned j=0;j<4;++j) {
                mpz_import(got.v,words,-1,4,0,0,run.points.data()+j*used+lane*words);
                if (mpz_cmp(got.v,s[j].v)) {
                    std::fprintf(stderr,"endpoint mismatch: lane=%zu slot=%u\n",lane,j);
                    return 3;
                }
            }
            mpz_gcd(g.v,s[1].v,n.v);
            const auto seed=(unsigned long long)active[lane];
            if (mpz_cmp(g.v,n.v)==0) std::printf("stage1_curve: sigma=%llu status=gcd_n\n",seed);
            else if (mpz_cmp_ui(g.v,1)>0) std::printf("stage1_curve: sigma=%llu status=factor factor=%s\n",seed,digits(g.v,10).c_str());
            else {
                std::printf("stage1_curve: sigma=%llu status=no_factor",seed);
                if (argc==10) {
                    if (!mpz_invert(x.v,s[1].v,n.v)) throw std::runtime_error("nonunit endpoint");
                    mpz_mul(x.v,x.v,s[0].v); mpz_mod(x.v,x.v,n.v); std::printf(" x=0x%s",digits(x.v,16).c_str());
                }
                std::puts("");
            }
        }
        std::printf("stage1_ntt: version=%s device=%d p=%u b1=%u requested_curves=%u curves=%zu bits=%zu"
                    " length=%u bpw=%u passes=%u tile=%u window=%u bad=0"
                    " host_setup_s=%.9f gpu_setup_s=%.9f readback_s=%.9f gmp_oracle_s=%.9f",
                    ecm_program::version,device,p,b1,curves,active.size(),bits.size(),run.length,run.digit_bits,
                    run.passes,run.tile,window,host_setup,run.setup_seconds,run.readback_seconds,oracle_seconds);
        values("event_ms",run.device_ms); values("wall_ms",run.wall_ms); std::puts("");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"stage1_ntt: %s\n",e.what()); return 2; }
}
