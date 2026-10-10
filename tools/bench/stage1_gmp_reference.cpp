// Independent CPU point oracle for completed Stage1 cost benchmarks.
// Ordinary GMP residues, no production ECM/CUDA headers or kernel code.
#include <gmp.h>
#include <array>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Integer {
    mpz_t v;
    Integer(){mpz_init(v);}
    ~Integer(){mpz_clear(v);}
    Integer(const Integer&)=delete;
    Integer& operator=(const Integer&)=delete;
};
struct Point {Integer x,z;};
static uint64_t decimal(const char *text) {
    std::string s(text);uint64_t v=0;
    const auto r=std::from_chars(s.data(),s.data()+s.size(),v);
    if(s.empty() || r.ec!=std::errc{} || r.ptr!=s.data()+s.size())throw std::runtime_error("invalid unsigned decimal argument");
    return v;
}
static std::string hex(mpz_srcptr n) {
    std::string s(mpz_sizeinbase(n,16)+2,'\0');mpz_get_str(s.data(),16,n);s.resize(std::char_traits<char>::length(s.c_str()));return s;
}
static void set64(mpz_ptr n,uint64_t x){mpz_import(n,1,1,sizeof(x),0,0,&x);}
static void scalar_lcm(mpz_ptr result,unsigned bound) {
    // Segmented prime sieve and balanced product of maximal prime powers.
    // Peak sieve storage is one MiB; no repeated multiplication into a huge scalar.
    std::vector<unsigned> base;
    for(unsigned q=2;(uint64_t)q*q<=bound;++q) {
        bool prime=true;for(auto p:base){if((uint64_t)p*p>q)break;if(q%p==0){prime=false;break;}}
        if(prime)base.push_back(q);
    }
    std::array<Integer,32> bins;std::array<bool,32> occupied{};Integer product;
    constexpr unsigned block=1u<<20;
    for(uint64_t lo=2;lo<=bound;lo+=block) {
        const auto hi=std::min<uint64_t>((uint64_t)bound+1,lo+block);
        std::vector<unsigned char> marked((size_t)(hi-lo));
        for(auto p:base) {
            const auto first=std::max<uint64_t>((uint64_t)p*p,((lo+p-1)/p)*p);
            for(auto q=first;q<hi;q+=p)marked[(size_t)(q-lo)]=1;
        }
        for(uint64_t q=lo;q<hi;++q)if(!marked[(size_t)(q-lo)]) {
            auto power=q;while(power<=bound/q)power*=q;
            set64(product.v,power);size_t level=0;
            while(level<bins.size() && occupied[level]) {
                mpz_mul(product.v,product.v,bins[level].v);occupied[level++]=false;
            }
            if(level==bins.size())throw std::runtime_error("scalar product capacity exceeded");
            mpz_swap(product.v,bins[level].v);occupied[level]=true;
        }
    }
    mpz_set_ui(result,1);
    for(size_t level=bins.size();level--;)if(occupied[level])mpz_mul(result,result,bins[level].v);
}
struct Ladder {
    mpz_srcptr n,a24,xdiff;
    std::array<Integer,6> scratch;
    void dbl(Point &r,const Point &p) {
        auto a=scratch[0].v,b=scratch[1].v,e=scratch[2].v,w=scratch[3].v;
        mpz_add(a,p.x.v,p.z.v);mpz_mul(a,a,a);mpz_mod(a,a,n);
        mpz_sub(b,p.x.v,p.z.v);mpz_mul(b,b,b);mpz_mod(b,b,n);
        mpz_sub(e,a,b);mpz_mul(r.x.v,a,b);mpz_mod(r.x.v,r.x.v,n);
        mpz_mul(w,a24,e);mpz_add(w,w,b);mpz_mul(r.z.v,e,w);mpz_mod(r.z.v,r.z.v,n);
    }
    void add(Point &r,const Point &p,const Point &q) {
        auto a=scratch[0].v,b=scratch[1].v,c=scratch[2].v,d=scratch[3].v,e=scratch[4].v,f=scratch[5].v;
        mpz_add(a,p.x.v,p.z.v);mpz_sub(b,q.x.v,q.z.v);mpz_mul(a,a,b);mpz_mod(a,a,n);
        mpz_sub(c,p.x.v,p.z.v);mpz_add(d,q.x.v,q.z.v);mpz_mul(c,c,d);mpz_mod(c,c,n);
        mpz_add(e,a,c);mpz_sub(f,a,c);
        mpz_mul(r.x.v,e,e);mpz_mod(r.x.v,r.x.v,n);
        mpz_mul(r.z.v,f,f);mpz_mod(r.z.v,r.z.v,n);mpz_mul(r.z.v,r.z.v,xdiff);mpz_mod(r.z.v,r.z.v,n);
    }
};
static std::string point(mpz_srcptr n,mpz_srcptr scalar,uint64_t sigma) {
    Integer u,v,t,a24,den,inv,xdiff;
    set64(u.v,sigma);mpz_mul(u.v,u.v,u.v);mpz_sub_ui(u.v,u.v,5);
    set64(v.v,sigma);mpz_mul_ui(v.v,v.v,4);
    mpz_sub(t.v,v.v,u.v);mpz_powm_ui(a24.v,t.v,3,n);
    mpz_mul_ui(t.v,u.v,3);mpz_add(t.v,t.v,v.v);mpz_mul(a24.v,a24.v,t.v);
    mpz_powm_ui(den.v,u.v,3,n);mpz_mul(den.v,den.v,v.v);mpz_mul_ui(den.v,den.v,16);
    if(!mpz_invert(inv.v,den.v,n))throw std::runtime_error("nonunit curve denominator");
    mpz_mul(a24.v,a24.v,inv.v);mpz_mod(a24.v,a24.v,n);
    Point r0,r1,sum,doubled;
    mpz_powm_ui(r0.x.v,u.v,3,n);mpz_powm_ui(r0.z.v,v.v,3,n);
    if(!mpz_invert(inv.v,r0.z.v,n))throw std::runtime_error("nonunit initial point");
    mpz_mul(xdiff.v,r0.x.v,inv.v);mpz_mod(xdiff.v,xdiff.v,n);
    Ladder arithmetic{n,a24.v,xdiff.v};arithmetic.dbl(r1,r0);
    for(mp_bitcnt_t i=mpz_sizeinbase(scalar,2)-1;i--;) {
        arithmetic.add(sum,r0,r1);
        if(mpz_tstbit(scalar,i)) {
            arithmetic.dbl(doubled,r1);mpz_swap(r0.x.v,sum.x.v);mpz_swap(r0.z.v,sum.z.v);
            mpz_swap(r1.x.v,doubled.x.v);mpz_swap(r1.z.v,doubled.z.v);
        } else {
            arithmetic.dbl(doubled,r0);mpz_swap(r1.x.v,sum.x.v);mpz_swap(r1.z.v,sum.z.v);
            mpz_swap(r0.x.v,doubled.x.v);mpz_swap(r0.z.v,doubled.z.v);
        }
    }
    if(!mpz_invert(inv.v,r0.z.v,n))throw std::runtime_error("Stage1 point is not a unit");
    mpz_mul(t.v,r0.x.v,inv.v);mpz_mod(t.v,t.v,n);return hex(t.v);
}
int main(int argc,char **argv) {
    try {
        if(argc==3 && std::string(argv[1])=="--scalar") {
            const auto b1=decimal(argv[2]);if(b1<2 || b1>260000000)throw std::runtime_error("B1 must be 2..260000000");
            Integer s;scalar_lcm(s.v,(unsigned)b1);std::cout<<hex(s.v)<<'\n';return 0;
        }
        const bool literal=argc==7 && std::string(argv[1])=="--n";
        if(!literal && argc!=6)throw std::runtime_error("usage: stage1_gmp_reference PRIME_EXPONENT B1 SIGMA_FIRST CURVES lcm|choose12, or --n HEX_N B1 SIGMA_FIRST CURVES lcm|choose12");
        const unsigned offset=literal?1:0;
        const auto b1=decimal(argv[2+offset]),sigma=decimal(argv[3+offset]),curves=decimal(argv[4+offset]);
        const std::string mode=argv[5+offset];
        const unsigned allowed[]={107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213};
        if(b1<2 || b1>260000000 || sigma<6 || sigma>9007199254740991ull || !curves || curves>4096 ||
           curves-1>9007199254740991ull-sigma || (mode!="lcm" && mode!="choose12"))throw std::runtime_error("unsupported reference scope");
        const auto start=std::chrono::steady_clock::now();Integer n,s;
        if(literal) {
            const std::string text=argv[2];
            if(text.empty() || text.size()>4096 || text.find_first_not_of("0123456789abcdefABCDEF")!=text.npos ||
               mpz_set_str(n.v,text.c_str(),16) || mpz_cmp_ui(n.v,3)<=0 || !mpz_odd_p(n.v) ||
               mpz_sizeinbase(n.v,2)>16384)throw std::runtime_error("target must be an odd hexadecimal integer >3 with at most 16384 bits");
        } else {
            const auto p=decimal(argv[1]);
            if(std::find(std::begin(allowed),std::end(allowed),p)==std::end(allowed))throw std::runtime_error("unsupported reference scope");
            mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,(mp_bitcnt_t)p);mpz_sub_ui(n.v,n.v,1);
        }
        scalar_lcm(s.v,(unsigned)b1);if(mode=="choose12")mpz_mul_ui(s.v,s.v,12);
        std::cout<<"{\"type\":\"stage1_reference\",\"bits\":"<<mpz_sizeinbase(n.v,2)<<",\"b1\":"<<b1<<",\"exponent\":\""<<mode
            <<"\",\"n_hex\":\""<<hex(n.v)<<"\",\"sigma_first\":"<<sigma<<",\"curves\":"<<curves
            <<",\"scalar_bits\":"<<mpz_sizeinbase(s.v,2)<<"}\n"<<std::flush;
        for(uint64_t i=0;i<curves;++i) {
            const auto x=point(n.v,s.v,sigma+i);
            std::cout<<"{\"type\":\"point\",\"sigma\":"<<sigma+i<<",\"x_hex\":\""<<x<<"\"}\n"<<std::flush;
        }
        std::cout<<"{\"type\":\"complete\",\"curves\":"<<curves<<",\"algorithm\":\"plain_gmp_ladder\",\"seconds\":"
            <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
    return 0;
}
