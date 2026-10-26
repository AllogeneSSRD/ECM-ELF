// Native GMP research probe: production Atkin-Morain vs GKL(iv).
// No production CLI/save dispatch is changed by linking this standalone probe.
#include "ecm/curve/gkl_iv.h"
#include "ecm/platform/program_version.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef ECM_FRIENDLY_WITH_STAGE2
#include "ecm/stage2/cuda/api.h"
#endif

using Integer=ecm_edwards::Integer;
using Setup=ecm_edwards::Setup;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double>(b-a).count();}
static void output(mpz_srcptr z){char *s=mpz_get_str(nullptr,10,z);std::cout<<s;void(*freefn)(void*,size_t);mp_get_memory_functions(nullptr,nullptr,&freefn);freefn(s,std::char_traits<char>::length(s)+1);}
struct Curve {
    Integer n,d,x,y,a24,factor;
    int a=1;
    Setup construct(unsigned family,uint64_t seed) {
        a=family==1?-1:1;
        if(family==1)return ecm_gkl_iv::construct(d.v,x.v,y.v,a24.v,factor.v,seed,n.v);
        if(family==2)return ecm_edwards::construct(d.v,x.v,y.v,a24.v,factor.v,seed,n.v);
        throw std::runtime_error("family must be 1 (GKL iv) or 2 (Atkin-Morain)");
    }
};
struct Point {Integer x,y,z,t,dt;};
struct Arithmetic {
    Curve &c;std::array<Integer,12> w;uint64_t multiplies=0,squares=0;
    explicit Arithmetic(Curve &curve):c(curve){}
    void reduce(mpz_t x){mpz_mod(x,x,c.n.v);}
    void mul(mpz_t r,mpz_srcptr x,mpz_srcptr y){++multiplies;mpz_mul(r,x,y);reduce(r);}
    void sqr(mpz_t r,mpz_srcptr x){++squares;mpz_mul(r,x,x);reduce(r);}
    void add(mpz_t r,mpz_srcptr x,mpz_srcptr y){mpz_add(r,x,y);reduce(r);}
    void sub(mpz_t r,mpz_srcptr x,mpz_srcptr y){mpz_sub(r,x,y);reduce(r);}
    void twice(mpz_t r){mpz_mul_2exp(r,r,1);reduce(r);}
    void copy(Point &p,const Point &q){mpz_set(p.x.v,q.x.v);mpz_set(p.y.v,q.y.v);mpz_set(p.z.v,q.z.v);mpz_set(p.t.v,q.t.v);mpz_set(p.dt.v,q.dt.v);}
    void identity(Point &p){mpz_set_ui(p.x.v,0);mpz_set_ui(p.y.v,1);mpz_set_ui(p.z.v,1);mpz_set_ui(p.t.v,0);mpz_set_ui(p.dt.v,0);}
    void cache(Point &p){mul(p.dt.v,c.d.v,p.t.v);if(c.a==-1)twice(p.dt.v);}
    void negative(Point &p,const Point &q){copy(p,q);mpz_neg(p.x.v,p.x.v);reduce(p.x.v);mpz_neg(p.t.v,p.t.v);reduce(p.t.v);mpz_neg(p.dt.v,p.dt.v);reduce(p.dt.v);}
    // HWCD extended doubling, 4M+4S for either sign of a.
    void dbl(Point &r,const Point &p) {
        auto A=w[0].v,B=w[1].v,C=w[2].v,D=w[3].v,E=w[4].v,F=w[5].v,G=w[6].v,H=w[7].v;
        sqr(A,p.x.v);sqr(B,p.y.v);sqr(C,p.z.v);twice(C);
        mpz_set(D,A);if(c.a==-1){mpz_neg(D,D);reduce(D);}
        add(E,p.x.v,p.y.v);sqr(E,E);sub(E,E,A);sub(E,E,B);
        add(G,D,B);sub(F,G,C);sub(H,D,B);
        mul(w[8].v,E,F);mul(w[9].v,G,H);mul(r.t.v,E,H);mul(r.z.v,F,G);
        mpz_set(r.x.v,w[8].v);mpz_set(r.y.v,w[9].v);
    }
    // Right-hand odd multiples cache d*T (a=1) or 2d*T (a=-1).
    // General a=1 addition is 9M; optimized a=-1 addition is 8M.
    void plus(Point &r,const Point &p,const Point &q) {
        auto A=w[0].v,B=w[1].v,C=w[2].v,D=w[3].v,E=w[4].v,F=w[5].v,G=w[6].v,H=w[7].v;
        if(c.a==-1) {
            sub(A,p.y.v,p.x.v);sub(H,q.y.v,q.x.v);mul(A,A,H);
            add(B,p.y.v,p.x.v);add(H,q.y.v,q.x.v);mul(B,B,H);
            mul(C,p.t.v,q.dt.v);mul(D,p.z.v,q.z.v);twice(D);
            sub(E,B,A);add(H,B,A);
        } else {
            mul(A,p.x.v,q.x.v);mul(B,p.y.v,q.y.v);mul(C,p.t.v,q.dt.v);mul(D,p.z.v,q.z.v);
            add(E,p.x.v,p.y.v);add(H,q.x.v,q.y.v);mul(E,E,H);sub(E,E,A);sub(E,E,B);sub(H,B,A);
        }
        sub(F,D,C);add(G,D,C);
        mul(w[8].v,E,F);mul(w[9].v,G,H);mul(r.t.v,E,H);mul(r.z.v,F,G);
        mpz_set(r.x.v,w[8].v);mpz_set(r.y.v,w[9].v);
    }
    bool valid(const Point &p) {
        if(!mpz_sgn(p.z.v))return false;
        sqr(w[0].v,p.x.v);sqr(w[1].v,p.y.v);sqr(w[2].v,p.z.v);
        if(c.a==-1)sub(w[3].v,w[1].v,w[0].v);else add(w[3].v,w[1].v,w[0].v);
        mul(w[3].v,w[3].v,w[2].v);sqr(w[2].v,w[2].v);sub(w[3].v,w[3].v,w[2].v);
        mul(w[0].v,w[0].v,w[1].v);mul(w[0].v,w[0].v,c.d.v);sub(w[3].v,w[3].v,w[0].v);
        mul(w[0].v,p.x.v,p.y.v);mul(w[1].v,p.z.v,p.t.v);sub(w[0].v,w[0].v,w[1].v);
        return !mpz_sgn(w[3].v) && !mpz_sgn(w[0].v);
    }
};
static std::vector<int> naf(mpz_srcptr scalar) {
    Integer s;mpz_set(s.v,scalar);std::vector<int> out;
    while(mpz_sgn(s.v)) {
        int d=0;
        if(mpz_odd_p(s.v)) {
            d=int(mpz_fdiv_ui(s.v,64));if(d>=32)d-=64;
            if(d>0)mpz_sub_ui(s.v,s.v,d);else mpz_add_ui(s.v,s.v,-d);
        }
        out.push_back(d);mpz_fdiv_q_2exp(s.v,s.v,1);
    }
    return out;
}
static void exponent(mpz_t s,unsigned b1,unsigned multiplier) {
    if(b1<2 || !multiplier)throw std::runtime_error("invalid B1/multiplier");
    std::vector<bool> prime(b1+1,true);mpz_set_ui(s,multiplier);
    for(unsigned p=2;p<=b1;++p)if(prime[p]) {
        unsigned power=p;while(power<=b1/p)power*=p;mpz_mul_ui(s,s,power);
        for(uint64_t j=uint64_t(p)*p;j<=b1;j+=p)prime[size_t(j)]=false;
    }
}
struct Timing {double dictionary=0,loop=0,check=0;uint64_t mul=0,sqr=0;};
static Timing multiply(Curve &c,const std::vector<int> &digits,Point &r,bool *valid=nullptr) {
    Arithmetic f(c);Point q,two;std::array<Point,16> positive,negative;
    auto start=Clock::now();mpz_set(q.x.v,c.x.v);mpz_set(q.y.v,c.y.v);mpz_set_ui(q.z.v,1);
    f.mul(q.t.v,q.x.v,q.y.v);f.cache(q);f.dbl(two,q);f.cache(two);f.copy(positive[0],q);
    for(unsigned i=1;i<16;++i){f.plus(positive[i],positive[i-1],two);f.cache(positive[i]);}
    for(unsigned i=0;i<16;++i)f.negative(negative[i],positive[i]);
    f.identity(r);auto prepared=Clock::now();f.multiplies=f.squares=0;
    for(auto i=digits.rbegin();i!=digits.rend();++i) {
        f.dbl(r,r);if(*i)f.plus(r,r,*i>0?positive[size_t(*i)/2]:negative[size_t(-*i)/2]);
    }
    auto done=Clock::now();const auto mul=f.multiplies,sqr=f.squares;
    // Exceptional/incomplete projective formulas are explicit failures, not hits.
    const bool ok=f.valid(r);
    if(valid)*valid=ok;
    else if(!ok)throw std::runtime_error("invalid projective endpoint; no success claim permitted");
    auto checked=Clock::now();return {seconds(start,prepared),seconds(prepared,done),seconds(done,checked),mul,sqr};
}
// An incomplete extended-Edwards formula can reach (0:0:0:0) on a small
// reduction even when the mathematical endpoint is the identity. Replay from
// the original verified point using the independent Montgomery X/Z ladder.
static void ladder(Curve &c,mpz_srcptr scalar,mpz_t result_x,mpz_t result_z) {
    Integer u,v,x0,z0,x1,z1,addx,addz,dx,dz;std::array<Integer,10>w;
    mpz_add_ui(u.v,c.y.v,1);mpz_mod(u.v,u.v,c.n.v);
    mpz_ui_sub(v.v,1,c.y.v);mpz_mod(v.v,v.v,c.n.v);
    if(!mpz_sgn(v.v)){mpz_set_ui(result_x,1);mpz_set_ui(result_z,0);return;}
    if(!mpz_sgn(u.v)){
        mpz_set_ui(result_x,mpz_even_p(scalar)?1:0);mpz_set_ui(result_z,mpz_even_p(scalar)?0:1);return;
    }
    mpz_set_ui(x0.v,1);mpz_set_ui(z0.v,0);mpz_set(x1.v,u.v);mpz_set(z1.v,v.v);
    auto mod=[&](mpz_t r){mpz_mod(r,r,c.n.v);};
    auto mul=[&](mpz_t r,mpz_srcptr a,mpz_srcptr b){mpz_mul(r,a,b);mod(r);};
    auto dbl=[&](mpz_srcptr x,mpz_srcptr z){
        mpz_add(w[0].v,x,z);mul(w[0].v,w[0].v,w[0].v);
        mpz_sub(w[1].v,x,z);mul(w[1].v,w[1].v,w[1].v);
        mpz_sub(w[2].v,w[0].v,w[1].v);mod(w[2].v);
        mul(dx.v,w[0].v,w[1].v);mul(dz.v,c.a24.v,w[2].v);
        mpz_add(dz.v,dz.v,w[1].v);mul(dz.v,dz.v,w[2].v);
    };
    for(size_t i=mpz_sizeinbase(scalar,2);i-->0;){
        mpz_add(w[0].v,x0.v,z0.v);mpz_sub(w[1].v,x0.v,z0.v);
        mpz_add(w[2].v,x1.v,z1.v);mpz_sub(w[3].v,x1.v,z1.v);
        mul(w[4].v,w[0].v,w[3].v);mul(w[5].v,w[1].v,w[2].v);
        mpz_add(w[6].v,w[4].v,w[5].v);mul(addx.v,w[6].v,w[6].v);mul(addx.v,addx.v,v.v);
        mpz_sub(w[6].v,w[4].v,w[5].v);mul(addz.v,w[6].v,w[6].v);mul(addz.v,addz.v,u.v);
        if(mpz_tstbit(scalar,i)){
            dbl(x1.v,z1.v);mpz_set(x1.v,dx.v);mpz_set(z1.v,dz.v);mpz_set(x0.v,addx.v);mpz_set(z0.v,addz.v);
        }else{
            dbl(x0.v,z0.v);mpz_set(x0.v,dx.v);mpz_set(z0.v,dz.v);mpz_set(x1.v,addx.v);mpz_set(z1.v,addz.v);
        }
    }
    if(!mpz_sgn(x0.v)&&!mpz_sgn(z0.v))throw std::runtime_error("invalid X/Z replay; no success claim permitted");
    mpz_set(result_x,x0.v);mpz_set(result_z,z0.v);
}
static int query(bool evaluate) {
    unsigned family,b1=0,multiplier=0;uint64_t seed;std::string n;
    while(std::cin>>family>>seed>>n) {
        if(evaluate && !(std::cin>>b1>>multiplier))throw std::runtime_error("incomplete evaluation input");
        Curve c;if(mpz_set_str(c.n.v,n.c_str(),10))throw std::runtime_error("invalid N");
        auto rc=c.construct(family,seed);std::cout<<family<<'|'<<seed<<'|'<<n<<'|'<<(rc==Setup::Ready?"ready":rc==Setup::Factor?"factor":"degenerate");
        for(mpz_srcptr z:{c.d.v,c.x.v,c.y.v,c.a24.v,c.factor.v}){std::cout<<'|';output(z);}
        if(evaluate && rc==Setup::Ready) {
            Integer s;exponent(s.v,b1,multiplier);Point r;bool valid;multiply(c,naf(s.v),r,&valid);
            Integer x,z;
            if(valid){mpz_add(x.v,r.z.v,r.y.v);mpz_mod(x.v,x.v,c.n.v);mpz_sub(z.v,r.z.v,r.y.v);mpz_mod(z.v,z.v,c.n.v);}
            else ladder(c,s.v,x.v,z.v);
            std::cout<<'|';output(x.v);std::cout<<'|';output(z.v);std::cout<<'|'<<(valid?0:1);
        }
        std::cout<<'\n';
    }
    return 0;
}
static int benchmark(unsigned bits,unsigned b1,unsigned curves,unsigned repeats) {
    Integer n,s;mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,bits);mpz_sub_ui(n.v,n.v,1);
    exponent(s.v,b1,48);const auto digits=naf(s.v);
    for(unsigned repeat=0;repeat<repeats;++repeat)for(unsigned k=0;k<2;++k) {
        const unsigned family=(repeat%2)?2-k:1+k;
        double setup=0,dictionary=0,loop=0,check=0,total=0;uint64_t mul=0,sqr=0;
        for(unsigned i=0;i<curves;++i) {
            Curve c;mpz_set(c.n.v,n.v);auto start=Clock::now();
            if(c.construct(family,i+1)!=Setup::Ready)throw std::runtime_error("non-ready benchmark construction");
            auto constructed=Clock::now();Point r;const auto t=multiply(c,digits,r);auto done=Clock::now();
            setup+=seconds(start,constructed);dictionary+=t.dictionary;loop+=t.loop;check+=t.check;
            total+=seconds(start,done);mul+=t.mul;sqr+=t.sqr;
        }
        std::cout<<std::setprecision(17)<<"{\"type\":\"ecm_friendly_cpu_benchmark\",\"program_version\":\""<<ecm_program::version
            <<"\",\"experiment_version\":\""<<ecm_gkl_iv::experiment_version<<"\",\"family\":"<<family
            <<",\"bits\":"<<bits<<",\"B1\":"<<b1<<",\"scalar_multiplier\":48,\"naf_w\":6,\"columns\":"<<digits.size()
            <<",\"curves\":"<<curves<<",\"repeat\":"<<repeat+1<<",\"setup_seconds\":"<<setup
            <<",\"dictionary_seconds\":"<<dictionary<<",\"loop_seconds\":"<<loop<<",\"check_seconds\":"<<check
            <<",\"total_seconds\":"<<total<<",\"loop_multiplications\":"<<mul<<",\"loop_squares\":"<<sqr<<",\"valid_endpoints\":"<<curves<<"}\n";
    }
    return 0;
}
#ifdef ECM_FRIENDLY_WITH_STAGE2
static std::string hex(mpz_srcptr z){char *s=mpz_get_str(nullptr,16,z);std::string r(s);void(*freefn)(void*,size_t);mp_get_memory_functions(nullptr,nullptr,&freefn);freefn(s,r.size()+1);return r;}
static void stage2_report(const char *json,void *context){*static_cast<std::string*>(context)=json;}
static int stage2() {
    unsigned family,b1,multiplier;uint64_t seed,b2,d;int device;std::string n;
    ecm_cuda_stage2_set_log_level(0);ecm_cuda_stage2_set_relocation(false);
    while(std::cin>>family>>seed>>n>>b1>>multiplier>>b2>>d>>device){
        Curve c;if(mpz_set_str(c.n.v,n.c_str(),10))throw std::runtime_error("invalid N");
        if(c.construct(family,seed)!=Setup::Ready)throw std::runtime_error("Stage2 test requires ready construction");
        Integer s,x,z,inv,g;exponent(s.v,b1,multiplier);Point r;bool valid;multiply(c,naf(s.v),r,&valid);
        if(valid){mpz_add(x.v,r.z.v,r.y.v);mpz_mod(x.v,x.v,c.n.v);mpz_sub(z.v,r.z.v,r.y.v);mpz_mod(z.v,z.v,c.n.v);}
        else ladder(c,s.v,x.v,z.v);
        if(!mpz_invert(inv.v,z.v,c.n.v))throw std::runtime_error("Stage1 hit/nonunit; do not count as a Stage2 test");
        mpz_mul(x.v,x.v,inv.v);mpz_mod(x.v,x.v,c.n.v);
        EcmStage2Metrics metrics;std::string result;
        const auto start=Clock::now();
        const int rc=ecm_cuda_stage2_run(hex(c.n.v).c_str(),hex(x.v).c_str(),seed,b1,b2,d,device,
            stage2_report,&result,0,&metrics,hex(c.a24.v).c_str());
        const auto end=Clock::now();
        if(rc || !metrics.clean || metrics.bad || result.empty())throw std::runtime_error("GPU Stage2 failed mandatory checks");
        std::cout<<std::setprecision(17)<<"FRIENDLY_GPU|{\"family\":"<<family<<",\"seed\":"<<seed
            <<",\"N\":\""<<n<<"\",\"B1\":"<<b1<<",\"B2\":"<<b2<<",\"multiplier\":"<<multiplier
            <<",\"d\":"<<metrics.d<<",\"p\":"<<metrics.p<<",\"giant_points\":"<<metrics.giant_points
            <<",\"selftest_cases\":"<<metrics.selftest_cases<<",\"checked\":"<<metrics.checked
            <<",\"bad\":"<<metrics.bad<<",\"engine_seconds\":"<<metrics.total_seconds
            <<",\"wall_seconds\":"<<seconds(start,end)<<","<<result<<"}\n";
    }
    return 0;
}
#endif
int main(int argc,char **argv) {
    try {
#ifdef ECM_FRIENDLY_WITH_STAGE2
        if(argc==2 && std::string(argv[1])=="--stage2")return stage2();
#endif
        if(argc==2 && std::string(argv[1])=="--construct")return query(false);
        if(argc==2 && std::string(argv[1])=="--evaluate")return query(true);
        if(argc==6 && std::string(argv[1])=="--benchmark")return benchmark(unsigned(std::stoul(argv[2])),unsigned(std::stoul(argv[3])),unsigned(std::stoul(argv[4])),unsigned(std::stoul(argv[5])));
        throw std::runtime_error("use --construct, --evaluate, or --benchmark bits B1 curves repeats");
    }catch(const std::exception &e){std::cerr<<"friendly_probe: "<<e.what()<<'\n';return 2;}
}
