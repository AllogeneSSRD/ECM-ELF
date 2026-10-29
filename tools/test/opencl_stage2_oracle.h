#pragma once
// Independent GMP fixtures, never linked into the native OpenCL operator library.
#include <gmp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace stage2_oracle {
using Word=std::uint64_t;
struct Big {
    mpz_t v;
    Big(){mpz_init(v);}explicit Big(Word x):Big(){mpz_import(v,1,-1,8,0,0,&x);}
    Big(const Big &x):Big(){mpz_set(v,x.v);}Big(Big &&x)noexcept:Big(){mpz_swap(v,x.v);}
    Big& operator=(const Big &x){mpz_set(v,x.v);return *this;}
    ~Big(){mpz_clear(v);}
};
using Poly=std::vector<Big>;
inline double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline Big add(const Big&a,const Big&b,const Big&n){Big r;mpz_add(r.v,a.v,b.v);mpz_mod(r.v,r.v,n.v);return r;}
inline Big sub(const Big&a,const Big&b,const Big&n){Big r;mpz_sub(r.v,a.v,b.v);mpz_mod(r.v,r.v,n.v);return r;}
inline Big mul(const Big&a,const Big&b,const Big&n){Big r;mpz_mul(r.v,a.v,b.v);mpz_mod(r.v,r.v,n.v);return r;}
inline std::vector<Word> flatten(const Poly &p,std::size_t nw){std::vector<Word> out(p.size()*nw);for(std::size_t i=0;i<p.size();++i){std::size_t count=0;if(mpz_sizeinbase(p[i].v,2)>nw*64)throw std::runtime_error("oracle limb overflow");mpz_export(out.data()+i*nw,&count,-1,8,0,0,p[i].v);}return out;}
inline Big import(const std::vector<Word> &a){Big n;mpz_import(n.v,a.size(),-1,8,0,0,a.data());return n;}
inline std::string text(const Big &n){std::string s(mpz_sizeinbase(n.v,10)+2,'\0');mpz_get_str(s.data(),10,n.v);s.resize(s.find('\0'));return s;}
inline Poly multiply(const Poly&a,const Poly&b,const Big&n){Poly c(a.size()+b.size()-1);for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<b.size();++j)c[i+j]=add(c[i+j],mul(a[i],b[j],n),n);return c;}
inline Poly roots(const Poly &r,const Big &n){Poly f{Big(1)};for(const auto &x:r)f=multiply(f,{sub(Big(0),x,n),Big(1)},n);return f;}
inline Poly remainder(Poly a,const Poly&b,const Big&n){
    if(b.size()<2 || mpz_cmp_ui(b.back().v,1))throw std::runtime_error("oracle divisor not monic");
    while(a.size()>=b.size()){auto q=a.back();const auto at=a.size()-b.size();for(std::size_t i=0;i<b.size();++i)a[at+i]=sub(a[at+i],mul(q,b[i],n),n);a.pop_back();}
    a.resize(b.size()-1);return a;
}
inline Big evaluate(const Poly&a,const Big&x,const Big&n){Big r;for(std::size_t i=a.size();i>0;--i)r=add(mul(r,x,n),a[i-1],n);return r;}
inline Poly inverse(const Poly&a,std::size_t count,const Big&n){Poly g(count);g[0]=Big(1);for(std::size_t i=1;i<count;++i){Big sum;for(std::size_t j=1;j<=i && j<a.size();++j)sum=add(sum,mul(a[j],g[i-j],n),n);g[i]=sub(Big(0),sum,n);}return g;}
inline Big gcd(const Big&a,const Big&n){Big g;mpz_gcd(g.v,a.v,n.v);return g;}
inline bool unit(const Big&a,const Big&n){return mpz_cmp_ui(gcd(a,n).v,1)==0;}
inline Big invert(const Big&a,const Big&n){Big inv;if(!mpz_invert(inv.v,a.v,n.v))throw std::runtime_error("fixture nonunit denominator");return inv;}
struct Point {Big x,z;};
inline Point twice(const Point&p,const Big&a24,const Big&n){auto a=add(p.x,p.z,n),b=sub(p.x,p.z,n);a=mul(a,a,n);b=mul(b,b,n);auto e=sub(a,b,n);return {mul(a,b,n),mul(e,add(b,mul(a24,e,n),n),n)};}
inline Point sum(const Point&p,const Point&q,const Point&diff,const Big&n){auto da=mul(sub(q.x,q.z,n),add(p.x,p.z,n),n),cb=mul(add(q.x,q.z,n),sub(p.x,p.z,n),n);
    auto plus=add(da,cb,n),minus=sub(da,cb,n);return {mul(diff.z,mul(plus,plus,n),n),mul(diff.x,mul(minus,minus,n),n)};}
inline Point ladder(const Point&p,const Big&scalar,const Big&a24,const Big&n){if(!mpz_sgn(scalar.v))return {Big(1),Big(0)};
    Point r=p,s=twice(p,a24,n);for(std::size_t i=mpz_sizeinbase(scalar.v,2)-1;i>0;--i){auto added=sum(r,s,p,n);if(mpz_tstbit(scalar.v,mp_bitcnt_t(i-1))){s=twice(s,a24,n);r=added;}else{r=twice(r,a24,n);s=added;}}return r;}
inline Big affine(const Point&p,const Big&n){return mul(p.x,invert(p.z,n),n);}
struct Curve {Big modulus,point_modulus,a24,exponent;Point stage1;Poly babies,giants;unsigned b1=0,b2=0,d=0,sigma=0;};
// Suyama param0 matches ecm_mont_cpu.h. This GMP fixture keeps projective
// differences (both Xdiff and Zdiff), independently of GPU normalization.
inline Curve curve(const Big &n,unsigned sigma,unsigned b1,unsigned b2,unsigned d){
    if(b1<5 || b2<=b1 || d<2 || d%2)throw std::invalid_argument("fixture bounds");
    Curve c;c.modulus=n;c.point_modulus=n;c.b1=b1;c.b2=b2;c.d=d;c.sigma=sigma;
    Big s(sigma);auto u=sub(mul(s,s,n),Big(5),n),v=mul(Big(4),s,n),u3=mul(mul(u,u,n),u,n),v3=mul(mul(v,v,n),v,n);
    auto diff=sub(v,u,n);auto numerator=mul(mul(mul(diff,diff,n),diff,n),add(mul(Big(3),u,n),v,n),n);
    c.a24=mul(numerator,invert(mul(Big(16),mul(u3,v,n),n),n),n);
    const auto a=sub(mul(Big(4),c.a24,n),Big(2),n);
    if(!unit(sub(mul(a,a,n),Big(4),n),n))throw std::runtime_error("fixture singular curve");
    c.exponent=Big(1);for(unsigned i=2;i<=b1;++i)mpz_lcm_ui(c.exponent.v,c.exponent.v,i);
    c.stage1=ladder({u3,v3},c.exponent,c.a24,n);
    if(!unit(c.stage1.z,n))throw std::runtime_error("fixture Stage1 already hits");
    for(unsigned j=1;j<=d/2;++j)if(std::gcd(j,d)==1)c.babies.push_back(affine(ladder(c.stage1,Big(j),c.a24,n),n));
    // Same conservative giant range as CUDA: i=1..floor(B2/D)+2.
    for(unsigned i=1;i<=b2/d+2;++i)c.giants.push_back(affine(ladder(c.stage1,Big(Word(i)*d),c.a24,n),n));
    return c;
}
inline Poly direct_values(const Poly&babies,const Poly&giants,const Big&n){Poly out(babies.size(),Big(1));for(std::size_t i=0;i<babies.size();++i)for(const auto &g:giants)out[i]=mul(out[i],sub(babies[i],g,n),n);return out;}
inline Big factor(const Poly&v,const Big&n){Big product(1);for(const auto &x:v)product=mul(product,x,n);return gcd(product,n);}
inline bool prime(unsigned p){if(p<2)return false;for(unsigned d=2;d<=p/d;++d)if(p%d==0)return false;return true;}
}
