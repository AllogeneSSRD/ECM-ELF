#include "group_order.h"
#include "../curve/ecm_edwards_curve.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ecm_group_order {
Integer::Integer() { mpz_init(v); }
Integer::Integer(uint64_t n) { mpz_init(v); mpz_import(v, 1, 1, sizeof(n), 0, 0, &n); }
Integer::Integer(const std::string &s) : Integer() {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || mpz_set_str(v,s.c_str(),10))
        throw std::invalid_argument("expected an unsigned decimal integer");
}
Integer::Integer(const Integer &n) { mpz_init_set(v,n.v); }
Integer::Integer(Integer &&n) noexcept : Integer() { mpz_swap(v,n.v); }
Integer &Integer::operator=(const Integer &n) { if(this!=&n)mpz_set(v,n.v);return *this; }
Integer &Integer::operator=(Integer &&n) noexcept { if(this!=&n)mpz_swap(v,n.v);return *this; }
Integer::~Integer() { mpz_clear(v); }
std::string Integer::str() const {
    std::string s(mpz_sizeinbase(v,10)+2,'\0');
    mpz_get_str(&s[0],10,v);s.resize(s.find('\0'));return s;
}
const char *name(Status s) {
    switch(s){case Status::Complete:return "complete";case Status::Partial:return "partial";
    case Status::ResourceLimit:return "resource_limit";default:return "invalid_input";}
}
const char *name(Primality p) {
    return p==Primality::Proven?"proven":p==Primality::Probable?"probable":"composite";
}
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
Integer z(long n) { Integer r;mpz_set_si(r.v,n);return r; }
Integer operator+(const Integer&a,const Integer&b){Integer r;mpz_add(r.v,a.v,b.v);return r;}
Integer operator-(const Integer&a,const Integer&b){Integer r;mpz_sub(r.v,a.v,b.v);return r;}
Integer operator*(const Integer&a,const Integer&b){Integer r;mpz_mul(r.v,a.v,b.v);return r;}
Integer operator/(const Integer&a,const Integer&b){Integer r;mpz_fdiv_q(r.v,a.v,b.v);return r;}
Integer operator+(const Integer&a,long b){return a+z(b);}
Integer operator-(const Integer&a,long b){return a-z(b);}
Integer operator*(const Integer&a,long b){return a*z(b);}
Integer mod(const Integer&a,const Integer&n){Integer r;mpz_mod(r.v,a.v,n.v);return r;}
Integer inv(const Integer&a,const Integer&n){Integer r;if(!mpz_invert(r.v,a.v,n.v))throw std::invalid_argument("noninvertible curve denominator");return r;}
Integer power(const Integer&a,unsigned long e){Integer r;mpz_pow_ui(r.v,a.v,e);return r;}
Integer gcd(const Integer&a,const Integer&b){Integer r;mpz_gcd(r.v,a.v,b.v);return r;}
bool eq(const Integer&a,const Integer&b){return mpz_cmp(a.v,b.v)==0;}
bool zero(const Integer&a){return mpz_sgn(a.v)==0;}
uint64_t u64(const Integer&a){
    if(mpz_sgn(a.v)<0 || mpz_sizeinbase(a.v,2)>64)throw std::invalid_argument("integer does not fit uint64");
    uint64_t n=0;size_t count=0;mpz_export(&n,&count,1,sizeof(n),0,0,a.v);return n;
}
struct Exhausted:std::runtime_error { using std::runtime_error::runtime_error; };
struct Context {
    const Limits &limits;
    Statistics statistics;
    Clock::time_point start=Clock::now();
    gmp_randstate_t random;
    explicit Context(const Limits&l):limits(l) {
        if(!l.milliseconds || !l.max_bits || l.max_bits>65536 || l.trial_limit>1000000 || l.baby_steps>10000000)
            throw std::invalid_argument("invalid resource limits");
        gmp_randinit_mt(random);gmp_randseed_ui(random,0xEC0F17);
    }
    ~Context(){gmp_randclear(random);}
    void check() const {if(elapsed(start)>=limits.milliseconds)throw Exhausted("time budget exhausted");}
    void size(const Integer&n)const{if(mpz_sizeinbase(n.v,2)>limits.max_bits)throw Exhausted("input bit budget exceeded");}
    Integer draw(const Integer&n){Integer r;mpz_urandomm(r.v,random,n.v);return r;}
};
std::vector<uint32_t> primes(uint32_t limit){
    std::vector<bool> sieve(size_t(limit)+1,true);std::vector<uint32_t> out;
    for(uint32_t p=2;p<=limit;++p)if(sieve[p]){
        out.push_back(p);if(uint64_t(p)*p<=limit)for(uint64_t j=uint64_t(p)*p;j<=limit;j+=p)sieve[size_t(j)]=false;
    }return out;
}
void record_factor(Factorization&out,const Integer&q,uint32_t e,Primality p){
    auto it=std::find_if(out.factors.begin(),out.factors.end(),[&](const PrimePower&v){return eq(v.prime,q);});
    if(it==out.factors.end())out.factors.push_back({q,e,p});else it->exponent+=e;
    Integer divisor=power(q,e);mpz_divexact(out.cofactor.v,out.cofactor.v,divisor.v);
}
Integer rho(const Integer&n,Context&ctx){
    Integer x,y,ys,c,prod,d,t;
    auto advance=[&](Integer&v){
        if(ctx.statistics.rho_iterations>=ctx.limits.rho_iterations)throw Exhausted("rho iteration budget exhausted");
        ++ctx.statistics.rho_iterations;
        if(!(ctx.statistics.rho_iterations&127))ctx.check();
        mpz_mul(v.v,v.v,v.v);mpz_add(v.v,v.v,c.v);mpz_mod(v.v,v.v,n.v);
    };
    for(unsigned attempt=0;attempt<32;++attempt){
        y=ctx.draw(n-1)+1;c=ctx.draw(n-1)+1;uint64_t r=1;mpz_set_ui(d.v,1);
        while(mpz_cmp_ui(d.v,1)==0){
            x=y;for(uint64_t j=0;j<r;++j)advance(y);
            for(uint64_t k=0;k<r && mpz_cmp_ui(d.v,1)==0;k+=64){
                ys=y;mpz_set_ui(prod.v,1);
                for(uint64_t j=0;j<std::min<uint64_t>(64,r-k);++j){
                    advance(y);mpz_sub(t.v,x.v,y.v);mpz_mul(prod.v,prod.v,t.v);mpz_mod(prod.v,prod.v,n.v);
                }mpz_gcd(d.v,prod.v,n.v);
            }
            if(r>std::numeric_limits<uint64_t>::max()/2)throw Exhausted("rho cycle size exceeded");
            r*=2;
        }
        if(eq(d,n))do{advance(ys);mpz_sub(t.v,x.v,ys.v);mpz_gcd(d.v,t.v,n.v);}while(mpz_cmp_ui(d.v,1)==0);
        if(mpz_cmp_ui(d.v,1)>0 && mpz_cmp(d.v,n.v)<0)return d;
    }throw Exhausted("rho restart budget exhausted");
}
void split(const Integer&n,uint32_t multiplicity,Factorization&out,Context&ctx){
    ctx.check();if(mpz_cmp_ui(n.v,1)==0)return;
    const auto primality=classify_prime(n);
    if(primality!=Primality::Composite){record_factor(out,n,multiplicity,primality);return;}
    if(mpz_perfect_power_p(n.v)){
        Integer root;
        for(unsigned long e=2;e<=mpz_sizeinbase(n.v,2);++e){
            ctx.check();if(mpz_root(root.v,n.v,e)){split(root,multiplicity*uint32_t(e),out,ctx);return;}
        }
    }
    Integer divisor=rho(n,ctx);split(divisor,multiplicity,out,ctx);split(n/divisor,multiplicity,out,ctx);
}
Factorization factor(const Integer&n,Context&ctx){
    Factorization out;out.cofactor=n;const auto start=Clock::now();
    if(mpz_cmp_ui(n.v,1)<0){out.reason="factorization requires a positive integer";return out;}
    try{
        ctx.size(n);Integer work=n;
        for(uint32_t q:primes(ctx.limits.trial_limit)){
            ctx.check();uint32_t e=0;
            while(mpz_divisible_ui_p(work.v,q)){mpz_divexact_ui(work.v,work.v,q);++e;}
            if(e)record_factor(out,Integer(q),e,Primality::Proven);
            if(mpz_cmp_ui(work.v,1)==0)break;
        }
        split(work,1,out,ctx);out.status=Status::Complete;
    }catch(const Exhausted&e){out.status=Status::Partial;out.reason=e.what();}
    std::sort(out.factors.begin(),out.factors.end(),[](const PrimePower&a,const PrimePower&b){return mpz_cmp(a.prime.v,b.prime.v)<0;});
    out.all_proven=out.status==Status::Complete && std::all_of(out.factors.begin(),out.factors.end(),[](const PrimePower&p){return p.primality==Primality::Proven;});
    out.statistics.factor_ms=elapsed(start);return out;
}

Point affine_add(const Curve&c,const Point&P,const Point&Q,Context*ctx=nullptr){
    if(ctx && !(++ctx->statistics.point_additions&127))ctx->check();
    if(P.infinity)return Q;if(Q.infinity)return P;
    const Integer&p=c.prime;Integer slope;
    if(eq(P.x,Q.x)){
        if(zero(mod(P.y+Q.y,p)))return {};
        slope=mod((P.x*P.x*3+c.a)*inv(P.y*2,p),p);
    }else slope=mod((Q.y-P.y)*inv(Q.x-P.x,p),p);
    Point R;R.infinity=false;R.x=mod(slope*slope-P.x-Q.x,p);R.y=mod(slope*(P.x-R.x)-P.y,p);return R;
}
// Jacobian scalar multiplication keeps inverses out of the bit loop.
Point scalar_mul(const Curve&c,const Point&P,const Integer&n,Context*ctx=nullptr){
    if(mpz_sgn(n.v)<0)throw std::invalid_argument("negative scalar");
    if(P.infinity || zero(n))return {};
    Integer X,Y,Z,p=c.prime;mpz_set_ui(Y.v,1);
    auto reduce=[&](const Integer&v){return mod(v,p);};
    for(size_t bits=mpz_sizeinbase(n.v,2);bits-- >0;){
        if(ctx && !(++ctx->statistics.point_additions&127))ctx->check();
        if(!zero(Z)){
            Integer xx=reduce(X*X),yy=reduce(Y*Y),yyyy=reduce(yy*yy),zz=reduce(Z*Z);
            Integer S=reduce(X*yy*4),M=reduce(xx*3+c.a*reduce(zz*zz));
            Integer nx=reduce(M*M-S*2),ny=reduce(M*(S-nx)-yyyy*8);
            Z=reduce(Y*Z*2);X=std::move(nx);Y=std::move(ny);
        }
        if(!mpz_tstbit(n.v,static_cast<mp_bitcnt_t>(bits)))continue;
        if(zero(Z)){X=P.x;Y=P.y;mpz_set_ui(Z.v,1);continue;}
        Integer zz=reduce(Z*Z),U=reduce(P.x*zz),V=reduce(P.y*Z*zz),H=reduce(U-X),J=reduce(V-Y);
        if(zero(H)){
            if(zero(J)){
                // Same point: the generic addition denominator vanishes.
                Point same=affine_add(c,P,P,ctx);
                if(same.infinity){mpz_set_ui(Z.v,0);}else{X=same.x;Y=same.y;mpz_set_ui(Z.v,1);}
            }else mpz_set_ui(Z.v,0);
            continue;
        }
        Integer hh=reduce(H*H),hhh=reduce(hh*H),K=reduce(X*hh);
        Integer nx=reduce(J*J-hhh-K*2),ny=reduce(J*(K-nx)-Y*hhh);
        Z=reduce(Z*H);X=std::move(nx);Y=std::move(ny);
    }
    if(zero(Z))return {};
    Integer iz=inv(Z,p),iz2=reduce(iz*iz);Point R;R.infinity=false;
    R.x=reduce(X*iz2);R.y=reduce(Y*iz2*iz);return R;
}
Integer square_root(const Integer&n,const Integer&p,Context*ctx=nullptr){
    Integer a=mod(n,p);if(zero(a))return a;
    if(mpz_legendre(a.v,p.v)!=1)throw std::invalid_argument("not a quadratic residue");
    Integer q=p-1,t,x,c,b,tmp;
    if(mpz_fdiv_ui(p.v,4)==3){Integer e=(p+1)/z(4);mpz_powm(x.v,a.v,e.v,p.v);return x;}
    const auto s=mpz_scan1(q.v,0);mpz_fdiv_q_2exp(q.v,q.v,s);
    Integer nonresidue(2);while(mpz_legendre(nonresidue.v,p.v)!=-1){if(ctx)ctx->check();nonresidue=nonresidue+1;}
    mpz_powm(c.v,nonresidue.v,q.v,p.v);Integer e=(q+1)/z(2);mpz_powm(x.v,a.v,e.v,p.v);mpz_powm(t.v,a.v,q.v,p.v);
    mp_bitcnt_t m=s;
    while(mpz_cmp_ui(t.v,1)!=0){
        if(ctx)ctx->check();tmp=t;mp_bitcnt_t i=0;
        while(mpz_cmp_ui(tmp.v,1)!=0 && i<m){tmp=mod(tmp*tmp,p);++i;}
        if(i>=m)throw std::invalid_argument("invalid prime field during square root");
        b=c;for(mp_bitcnt_t k=0;k<m-i-1;++k)b=mod(b*b,p);
        x=mod(x*b,p);c=mod(b*b,p);t=mod(t*c,p);m=i;
    }return x;
}
Curve short_model(const Integer&p,const Integer&a2,const Integer&a4,const Point&P){
    Curve c;c.prime=p;Integer third=inv(z(3),p),shift=mod(a2*third,p);
    c.a=mod(a4-a2*shift,p);c.b=mod(shift*shift*shift*2-a4*shift,p);
    c.point=P;if(!P.infinity)c.point.x=mod(P.x+shift,p);
    if(zero(mod(power(c.a,3)*4+power(c.b,2)*27,p)))throw std::invalid_argument("singular curve");
    if(!point_on_curve(c,c.point))throw std::invalid_argument("base point mapping failed");return c;
}
std::string key(const Point&P,const Integer&p){
    if(P.infinity)return std::string(1,'I');
    const size_t bytes=(mpz_sizeinbase(p.v,2)+7)/8;
    std::string out(1+2*bytes,'\0');out[0]='P';size_t count=0;
    const size_t xb=(mpz_sizeinbase(P.x.v,2)+7)/8,yb=(mpz_sizeinbase(P.y.v,2)+7)/8;
    mpz_export(&out[1+bytes-xb],&count,1,1,1,0,P.x.v);
    mpz_export(&out[1+2*bytes-yb],&count,1,1,1,0,P.y.v);return out;
}
Integer exact_point_order(const Curve&c,const Point&P,const Integer&M,const std::vector<PrimePower>&parts,Context&ctx){
    if(!scalar_mul(c,P,M,&ctx).infinity)throw std::invalid_argument("known multiple does not annihilate point");
    Integer n=M;
    for(const auto&f:parts)for(uint32_t e=0;e<f.exponent;++e){
        ctx.check();Integer candidate=n/f.prime;
        if(scalar_mul(c,P,candidate,&ctx).infinity)n=std::move(candidate);else break;
    }
    if(!scalar_mul(c,P,n,&ctx).infinity)throw std::logic_error("point order verification failed");
    for(const auto&f:parts)if(mpz_divisible_p(n.v,f.prime.v) && scalar_mul(c,P,n/f.prime,&ctx).infinity)
        throw std::logic_error("point order minimality failed");
    return n;
}
Integer interval_multiple(const Curve&c,const Point&P,const Integer&lower,const Integer&upper,Context&ctx){
    Integer width=upper-lower,root;mpz_sqrt(root.v,width.v);root=root+1;
    if(mpz_cmp(root.v,Integer(ctx.limits.baby_steps).v)>0)throw Exhausted("baby-step table budget exceeded");
    const uint64_t m=u64(root);std::unordered_map<std::string,uint64_t> baby;
    baby.reserve(size_t(m));Point R;
    for(uint64_t j=0;j<m;++j){
        ctx.check();auto inserted=baby.emplace(key(R,c.prime),j);++ctx.statistics.baby_steps;
        // First repetition in jP is the exact point order, also a valid multiple.
        if(!inserted.second)return Integer(j-inserted.first->second);
        R=affine_add(c,R,P,&ctx);
    }
    Point step=scalar_mul(c,P,root,&ctx),giant=scalar_mul(c,P,lower,&ctx);
    if(!step.infinity)step.y=mod(z(0)-step.y,c.prime);
    if(!giant.infinity)giant.y=mod(z(0)-giant.y,c.prime);
    Integer base=lower;
    for(uint64_t i=0;i<=u64(width/root);++i){
        ctx.check();++ctx.statistics.giant_steps;
        auto it=baby.find(key(giant,c.prime));
        if(it!=baby.end()){
            Integer candidate=base+Integer(it->second);
            if(mpz_cmp(candidate.v,upper.v)<=0)return candidate;
        }
        giant=affine_add(c,giant,step,&ctx);base=base+root;
    }throw std::logic_error("no annihilator in Hasse interval");
}
// Combined divisibility constraints from E and its quadratic twist.
bool unique_order(Integer&out,const Integer&D,const Integer&T,const Integer&p,const Integer&lower,const Integer&upper){
    Integer total=(p+1)*2,g=gcd(D,T);
    if(!mpz_divisible_p(total.v,g.v))throw std::logic_error("inconsistent twist orders");
    Integer modulus=D*(T/g),residue(0),tg=T/g;
    if(mpz_cmp_ui(tg.v,1)>0)residue=mod(D*mod((total/g)*inv(D/g,tg),tg),modulus);
    Integer k;mpz_cdiv_q(k.v,(lower-residue).v,modulus.v);Integer candidate=residue+k*modulus;
    if(mpz_cmp(candidate.v,upper.v)>0)throw std::logic_error("no group order satisfies point constraints");
    if(mpz_cmp((candidate+modulus).v,upper.v)<=0)return false;out=std::move(candidate);return true;
}
Integer count(const Curve&curve,Context&ctx){
    const auto&p=curve.prime;
    if(mpz_cmp(p.v,Integer(ctx.limits.enumeration_limit).v)<=0){
        uint64_t n=1;Integer x(0);
        for(uint64_t i=0;i<u64(p);++i){ctx.check();x=Integer(i);Integer rhs=mod(x*x*x+curve.a*x+curve.b,p);n+=1+mpz_legendre(rhs.v,p.v);}
        return Integer(n);
    }
    Integer h;mpz_sqrt(h.v,(p*4).v);Integer lower=p+1-h,upper=p+1+h;
    // Reject impossible allocations before random point generation.
    Integer needed;mpz_sqrt(needed.v,(upper-lower).v);needed=needed+1;
    if(mpz_cmp(needed.v,Integer(ctx.limits.baby_steps).v)>0)throw Exhausted("baby-step table budget exceeded");
    Integer ns(2);while(mpz_legendre(ns.v,p.v)!=-1){ctx.check();ns=ns+1;}
    Curve twist;twist.prime=p;twist.a=mod(curve.a*ns*ns,p);twist.b=mod(curve.b*ns*ns*ns,p);
    Integer divisors[2]={Integer(1),Integer(1)},result;
    for(uint32_t attempt=0;attempt<ctx.limits.random_points;++attempt){
        ctx.check();unsigned side=attempt%2;const Curve&c=side?twist:curve;Point P;
        if(!attempt && !curve.point.infinity)P=curve.point;
        else for(;;){
            ctx.check();P.infinity=false;P.x=ctx.draw(p);Integer rhs=mod(P.x*P.x*P.x+c.a*P.x+c.b,p);
            if(mpz_legendre(rhs.v,p.v)>=0){P.y=square_root(rhs,p,&ctx);break;}
        }
        ++ctx.statistics.sampled_points;
        Integer M=interval_multiple(c,P,lower,upper,ctx);Factorization f=factor(M,ctx);
        if(f.status!=Status::Complete)throw Exhausted("point-annihilator factorization incomplete: "+f.reason);
        // A probable factor must never certify a unique #E through an inexact divisor.
        if(!f.all_proven)throw Exhausted("group-count certificate needs proven annihilator factors");
        Integer d=exact_point_order(c,P,M,f.factors,ctx);
        mpz_lcm(divisors[side].v,divisors[side].v,d.v);
        if(unique_order(result,divisors[0],divisors[1],p,lower,upper))return result;
    }throw Exhausted("random-point budget exhausted");
}
Factorization supplied_factorization(const Integer&M,const std::vector<PrimePower>&parts,Context&ctx){
    Factorization out;out.cofactor=M;Integer product(1);
    for(const auto&f:parts){
        ctx.check();ctx.size(f.prime);
        if(!f.exponent || f.exponent>ctx.limits.max_bits || mpz_cmp_ui(f.prime.v,2)<0)
            throw std::invalid_argument("invalid supplied prime power");
        if(std::any_of(out.factors.begin(),out.factors.end(),[&](const PrimePower&q){return eq(q.prime,f.prime);}))
            throw std::invalid_argument("duplicate supplied prime");
        auto kind=classify_prime(f.prime);if(kind==Primality::Composite)throw std::invalid_argument("composite supplied factor");
        Integer divisor=power(f.prime,f.exponent);product=product*divisor;
        if(mpz_cmp(product.v,M.v)>0)throw std::invalid_argument("supplied factors exceed multiple");
        out.factors.push_back({f.prime,f.exponent,kind});
    }
    if(!eq(product,M))throw std::invalid_argument("supplied factor product mismatch");
    std::sort(out.factors.begin(),out.factors.end(),[](const PrimePower&a,const PrimePower&b){return mpz_cmp(a.prime.v,b.prime.v)<0;});
    mpz_set_ui(out.cofactor.v,1);out.status=Status::Complete;
    out.all_proven=std::all_of(out.factors.begin(),out.factors.end(),[](const PrimePower&p){return p.primality==Primality::Proven;});return out;
}
void analyze_impl(Analysis&out,const Curve&c,Context&ctx,const Integer*known,const std::vector<PrimePower>*parts){
    ctx.size(c.prime);ctx.size(c.a);ctx.size(c.b);ctx.size(c.point.x);ctx.size(c.point.y);
    if(mpz_cmp_ui(c.prime.v,3)<=0)throw std::invalid_argument("modulus must be a prime >3");
    out.modulus_primality=classify_prime(c.prime);
    if(out.modulus_primality==Primality::Composite)throw std::invalid_argument("composite modulus");
    if(zero(mod(power(c.a,3)*4+power(c.b,2)*27,c.prime)))throw std::invalid_argument("singular curve");
    if(!point_on_curve(c,c.point))throw std::invalid_argument("point is not on curve");
    if(parts && !known)throw std::invalid_argument("factorization requires a known multiple");
    if(known){
        ctx.size(*known);if(mpz_cmp_ui(known->v,1)<0)throw std::invalid_argument("known multiple must be positive");out.multiple=*known;
    }else{
        const auto start=Clock::now();
        try{out.group_order=count(c,ctx);}catch(...){ctx.statistics.count_ms=elapsed(start);throw;}
        ctx.statistics.count_ms=elapsed(start);out.multiple=out.group_order;out.group_order_computed=true;
    }
    const auto factor_start=Clock::now();
    out.multiple_factorization=parts?supplied_factorization(out.multiple,*parts,ctx):factor(out.multiple,ctx);
    ctx.statistics.factor_ms=elapsed(factor_start);
    if(out.multiple_factorization.status!=Status::Complete){out.status=Status::Partial;out.reason=out.multiple_factorization.reason;return;}
    const auto point_start=Clock::now();
    try{out.point_order=exact_point_order(c,c.point,out.multiple,out.multiple_factorization.factors,ctx);}
    catch(...){ctx.statistics.point_ms=elapsed(point_start);throw;}
    ctx.statistics.point_ms=elapsed(point_start);out.point_minimality_checked=true;
    for(const auto&f:out.multiple_factorization.factors){
        Integer work=out.point_order;uint32_t e=0;
        while(mpz_divisible_p(work.v,f.prime.v)){mpz_divexact(work.v,work.v,f.prime.v);++e;}
        if(e)out.point_factorization.push_back({f.prime,e,f.primality});
    }
    out.all_proven=out.modulus_primality==Primality::Proven && out.multiple_factorization.all_proven;
    out.status=Status::Complete;
}
} // namespace

Primality classify_prime(const Integer&n){
    if(mpz_cmp_ui(n.v,2)<0 || mpz_probab_prime_p(n.v,32)==0)return Primality::Composite;
    if(mpz_sizeinbase(n.v,2)>64)return Primality::Probable;
    // First 12 prime bases suffice below 2^64 (Sorenson/Webster, arXiv:1509.00864).
    Integer d=n-1,x,a;const auto s=mpz_scan1(d.v,0);mpz_fdiv_q_2exp(d.v,d.v,s);
    for(unsigned base:{2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u}){
        if(mpz_cmp_ui(n.v,base)<=0)continue;a=Integer(base);mpz_powm(x.v,a.v,d.v,n.v);
        if(mpz_cmp_ui(x.v,1)==0 || eq(x,n-1))continue;
        bool pass=false;for(mp_bitcnt_t r=1;r<s;++r){x=mod(x*x,n);if(eq(x,n-1)){pass=true;break;}}
        if(!pass)return Primality::Composite;
    }return Primality::Proven;
}
Factorization factor_integer(const Integer&n,const Limits&limits){
    try{Context ctx(limits);Factorization out=factor(n,ctx);ctx.statistics.total_ms=elapsed(ctx.start);ctx.statistics.factor_ms=out.statistics.factor_ms;out.statistics=ctx.statistics;return out;}
    catch(const std::invalid_argument&e){Factorization out;out.cofactor=n;out.reason=e.what();return out;}
}
bool point_on_curve(const Curve&c,const Point&P){
    if(mpz_cmp_ui(c.prime.v,3)<=0)return false;
    if(P.infinity)return true;
    if(mpz_sgn(P.x.v)<0 || mpz_sgn(P.y.v)<0 || mpz_cmp(P.x.v,c.prime.v)>=0 || mpz_cmp(P.y.v,c.prime.v)>=0)return false;
    return zero(mod(P.y*P.y-P.x*P.x*P.x-c.a*P.x-c.b,c.prime));
}
Point multiply(const Curve&c,const Point&P,const Integer&scalar){return scalar_mul(c,P,scalar);}

Curve construct_curve(const CurveRequest&r){
    const Integer&p=r.prime;
    if(mpz_cmp_ui(p.v,3)<=0 || classify_prime(p)==Primality::Composite)throw std::invalid_argument("modulus must be a prime >3");
    if(!r.sigma)throw std::invalid_argument("sigma must be nonzero uint64");
    Integer A,B,x,y,s(r.sigma);Point P;
    if(r.family==Family::Edwards){
        if(r.parameter!=-1)throw std::invalid_argument("Edwards does not have a Montgomery parameter");
        Integer d,ex,ey,a24,f;
        if(ecm_edwards::construct(d.v,ex.v,ey.v,a24.v,f.v,r.sigma,p.v)!=ecm_edwards::Setup::Ready)
            throw std::invalid_argument("degenerate Edwards construction");
        A=mod((d+1)*2*inv(z(1)-d,p),p);B=mod(z(4)*inv(z(1)-d,p),p);
        if(eq(ey,z(1)))return short_model(p,mod(B*A,p),mod(B*B,p),P);
        P.infinity=false;
        if(eq(ey,p-1)){P.x=Integer(0);P.y=Integer(0);}
        else{Integer u=mod((ey+1)*inv(z(1)-ey,p),p);P.x=mod(B*u,p);P.y=mod(B*B*u*inv(ex,p),p);}
    }else{
        if(r.parameter<0 || r.parameter>3)throw std::invalid_argument("Montgomery parameter must be 0..3");
        x=Integer(2);y=Integer(1);
        if(r.parameter==0){
            if(r.sigma<6)throw std::invalid_argument("Suyama sigma must be >=6");
            Integer u=mod(s*s-5,p),v=mod(s*4,p);
            A=mod((u*3+v)*power(v-u,3)*inv(power(u,3)*v*4,p)-2,p);
            x=mod(power(mod(u*inv(v,p),p),3),p);B=mod(x*(x*(x+A)+1),p);
        }else if(r.parameter==1 || r.parameter==3){
            A=mod((r.parameter==1?s*s:s)*4*inv(power(z(2),r.parameter==1?64:32),p)-2,p);B=mod(A*4+10,p);
        }else{
            Curve aux;aux.prime=p;aux.a=Integer(0);aux.b=Integer(36);aux.point.infinity=false;
            aux.point.x=mod(z(-3),p);aux.point.y=mod(z(3),p);
            Point t=scalar_mul(aux,aux.point,s);if(t.infinity)throw std::invalid_argument("auxiliary point at infinity");
            Integer x3=mod((t.x*3+t.y+6)*inv((t.y-3)*2,p),p);
            A=mod((z(1)-power(x3,4)*3-x3*x3*6)*inv(power(x3,3)*4,p),p);
            y=mod(A*4+10,p);B=inv(y,p);
        }
        if(zero(B) || zero(mod(A*A-4,p)))throw std::invalid_argument("singular Montgomery curve");
        P.infinity=false;P.x=mod(B*x,p);P.y=mod(B*B*y,p);
    }
    return short_model(p,mod(B*A,p),mod(B*B,p),P);
}
Analysis analyze_curve(const Curve&c,const Limits&limits,const Integer*known,const std::vector<PrimePower>*parts){
    Analysis out;
    try{Context ctx(limits);
        try{analyze_impl(out,c,ctx,known,parts);}catch(const Exhausted&e){out.status=Status::ResourceLimit;out.reason=e.what();}
        catch(const std::invalid_argument&e){out.status=Status::InvalidInput;out.reason=e.what();}
        ctx.statistics.total_ms=elapsed(ctx.start);out.statistics=ctx.statistics;
    }catch(const std::invalid_argument&e){out.reason=e.what();}return out;
}
Analysis analyze(const CurveRequest&r,const Limits&limits,const Integer*known,const std::vector<PrimePower>*parts){
    Analysis out;
    try{Context ctx(limits);
        try{
            ctx.size(r.prime);const auto start=Clock::now();Curve c=construct_curve(r);ctx.statistics.construction_ms=elapsed(start);
            analyze_impl(out,c,ctx,known,parts);
        }catch(const Exhausted&e){out.status=Status::ResourceLimit;out.reason=e.what();}
        catch(const std::invalid_argument&e){out.status=Status::InvalidInput;out.reason=e.what();}
        ctx.statistics.total_ms=elapsed(ctx.start);out.statistics=ctx.statistics;
    }catch(const std::invalid_argument&e){out.reason=e.what();}return out;
}
} // namespace ecm_group_order
